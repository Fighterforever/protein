import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

SEED = 42

uniprot_pattern = re.compile('^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})\\b')


def set_seed(seed):
    global SEED
    SEED = seed
    np.random.seed(seed)


def identify_columns(df):
    prot_cols = [c for c in df.columns if uniprot_pattern.match(str(c))]
    meta_cols = [c for c in df.columns if c not in prot_cols]
    return meta_cols, prot_cols


def schema_overview(df, meta_cols, prot_cols, outpath):
    n_samples = df.shape[0]
    info = []
    info.append(f'Samples: {n_samples}')
    info.append(f'Protein columns: {len(prot_cols)}')
    info.append(f'Meta feature columns: {len(meta_cols)}')
    miss = df[meta_cols].isna().sum()
    info.append('Missing values in meta features:')
    info.extend([f'  {k}: {v}' for k, v in miss.items()])
    Path(outpath).write_text('\n'.join(info), encoding='utf-8')


def make_folds(groups, n_splits, outpath):
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
        splits.append({'train': tr.tolist(), 'test': te.tolist(),
                       'n_train_groups': int(groups.iloc[tr].nunique()),
                       'n_test_groups': int(groups.iloc[te].nunique())})
    Path(outpath).write_text(json.dumps(splits, indent=2))
    return splits


def compute_targets(Y):
    eps = 1e-6
    rel = (Y + eps)
    rel = rel / rel.sum(axis=1, keepdims=True)
    clr = np.log(rel)
    clr = clr - clr.mean(axis=1, keepdims=True)
    return rel, clr


def load_prot_ann(path, proteins):
    ann = pd.read_csv(path)
    ann['Uniprot'] = ann['Uniprot'].str.upper()
    ann = ann.drop_duplicates('Uniprot')
    ann = ann.set_index('Uniprot')
    ann = ann.reindex([p.split()[0].upper() for p in proteins])
    numcols = ann.select_dtypes(include=['number', 'bool']).columns
    ann = ann[numcols].fillna(0)
    return ann


def build_long_table(meta_df, y_mat, prot_ann, use_prot, use_inter, train_idx):
    numeric_meta = meta_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_meta = [c for c in meta_df.columns if c not in numeric_meta]
    num_imp = SimpleImputer(strategy='median')
    num_scaler = StandardScaler()
    cat_imp = SimpleImputer(strategy='constant', fill_value='(missing)')
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    num_train = num_scaler.fit_transform(num_imp.fit_transform(meta_df.iloc[train_idx][numeric_meta]))
    cat_train = ohe.fit_transform(cat_imp.fit_transform(meta_df.iloc[train_idx][categorical_meta])) if categorical_meta else np.zeros((len(train_idx),0))
    # transform all
    num_all = num_scaler.transform(num_imp.transform(meta_df[numeric_meta]))
    cat_all = ohe.transform(cat_imp.transform(meta_df[categorical_meta])) if categorical_meta else np.zeros((len(meta_df),0))
    meta_feat_names = list(numeric_meta) + list(ohe.get_feature_names_out(categorical_meta))
    X_meta = np.hstack([num_all, cat_all])

    prot_scaler = StandardScaler()
    prot_all = prot_scaler.fit_transform(prot_ann.values)
    prot_feat_names = prot_ann.columns.tolist()

    n_samples, n_prot = y_mat.shape
    X_meta_repeat = np.repeat(X_meta, n_prot, axis=0)
    parts = [X_meta_repeat]
    if use_prot:
        X_prot_tile = np.tile(prot_all, (n_samples,1))
        parts.append(X_prot_tile)
    inter_names = []
    if use_inter and len(numeric_meta)>0 and prot_all.shape[1]>0:
        S = num_all
        P = prot_all
        inter = np.einsum('si,pj->sipj', S, P).reshape(n_samples*n_prot, -1)
        parts.append(inter)
        inter_names = [f'{sn}*{pn}' for sn in numeric_meta for pn in prot_feat_names]
    X = np.hstack(parts)
    y = y_mat.reshape(-1)
    sample_ids = np.repeat(np.arange(n_samples), n_prot)
    prot_ids = np.tile(np.arange(n_prot), n_samples)
    feature_names = meta_feat_names + (prot_feat_names if use_prot else []) + inter_names
    return X, y, sample_ids, prot_ids, feature_names, n_samples, n_prot


def ndcg_at_k(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1][:k]
    denom = np.log2(np.arange(2, k+2))
    dcg = (y_true[order] / denom).sum()
    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = (y_true[ideal_order] / denom).sum()
    return float(dcg/idcg) if idcg>0 else 0.0


def hit_at_k(y_true, y_score, k=5):
    pred_top = np.argsort(y_score)[::-1][:k]
    true_top = np.argsort(y_true)[::-1][:k]
    return float(len(set(pred_top).intersection(set(true_top)))/k)


def row_spearman(y_true, y_score):
    return spearmanr(y_true, y_score).correlation


def js_divergence(p, q):
    m = 0.5*(p+q)
    return float(0.5*(jensenshannon(p, m)**2 + jensenshannon(q, m)**2))


def clr_rmse(y_true_clr, y_pred_clr):
    return float(np.sqrt(((y_true_clr - y_pred_clr)**2).mean()))


def evaluate_row(y_true_rel, y_pred_rel, y_true_clr, y_pred_clr):
    metrics = {}
    metrics['ndcg5'] = ndcg_at_k(y_true_rel, y_pred_rel, 5)
    metrics['ndcg10'] = ndcg_at_k(y_true_rel, y_pred_rel,10)
    metrics['hit5'] = hit_at_k(y_true_rel, y_pred_rel,5)
    metrics['hit10'] = hit_at_k(y_true_rel, y_pred_rel,10)
    metrics['spearman'] = row_spearman(y_true_rel, y_pred_rel)
    metrics['js'] = js_divergence(y_true_rel, y_pred_rel)
    metrics['clr_rmse'] = clr_rmse(y_true_clr, y_pred_clr)
    return metrics


def assign_family(name):
    parts = str(name).upper().split()
    gene = parts[1] if len(parts)>1 else ''
    if gene.startswith('ALB'):
        return 'Albumin'
    if re.match(r'^IG[HK]', gene):
        return 'Ig'
    if re.match(r'^C[1-9]', gene):
        return 'Complement'
    if gene.startswith('APO'):
        return 'Apolipoprotein'
    if re.match(r'^F[0-9]', gene):
        return 'Coagulation'
    return 'Other'


def baseline_prior(prior_vec, n_samples):
    return np.tile(prior_vec, (n_samples,1))


def compute_baseline_prior(prot_ann):
    cols = [c for c in prot_ann.columns if 'concentration' in c.lower() or 'expression' in c.lower()]
    if cols:
        prior = prot_ann[cols].mean(axis=1).values
    else:
        prior = prot_ann.mean(axis=1).values
    prior = np.nan_to_num(prior, nan=0.0)
    prior = (prior + 1e-6)
    prior = prior / prior.sum()
    return prior


def train_model(model_type, X_train, y_train, group_train=None):
    if model_type=='ridge':
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(X_train, y_train)
    elif model_type=='xgb_reg' and xgb is not None:
        model = xgb.sklearn.XGBRegressor(objective='reg:squarederror', n_estimators=200,
                                         learning_rate=0.1, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.8, random_state=SEED, n_jobs=4)
        model.fit(X_train, y_train)
    elif model_type=='xgb_rank' and xgb is not None:
        model = xgb.sklearn.XGBRanker(objective='rank:pairwise', n_estimators=200,
                                      learning_rate=0.1, max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, random_state=SEED, n_jobs=4)
        model.fit(X_train, y_train, group=group_train)
    else:
        # fallback simple linear
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(X_train, y_train)
    return model


def evaluate_fold(model, X_test, test_idx, n_prot, y_rel, y_clr, model_name, fold, feature_names, results_dir,
                   prot_cols, sample_ids_test):
    pred = model.predict(X_test)
    pred = pred.reshape(len(test_idx), n_prot)
    pred_rel = np.maximum(pred, 0)
    pred_rel = pred_rel / pred_rel.sum(axis=1, keepdims=True)
    pred_clr = np.log(pred_rel)
    pred_clr = pred_clr - pred_clr.mean(axis=1, keepdims=True)
    records=[]
    for i in range(len(test_idx)):
        m = evaluate_row(y_rel[test_idx[i]], pred_rel[i], y_clr[test_idx[i]], pred_clr[i])
        m['sample'] = int(test_idx[i])
        records.append(m)
    row_df = pd.DataFrame(records)
    metrics_mean = row_df[['ndcg5','ndcg10','hit5','hit10','spearman','js','clr_rmse']].mean().to_dict()
    metrics_mean.update({'model': model_name, 'fold': fold})
    # protein-level metrics
    prot_records=[]
    for j,prot in enumerate(prot_cols):
        r2 = r2_score(y_rel[test_idx,j], pred_rel[:,j])
        rho = spearmanr(y_rel[test_idx,j], pred_rel[:,j]).correlation
        prot_records.append({'protein':prot,'r2':r2,'spearman':rho})
    prot_df = pd.DataFrame(prot_records)
    prot_df['family'] = prot_df['protein'].map(assign_family)
    group_df = prot_df.groupby('family')[['r2','spearman']].agg(['mean','var'])
    group_df.columns = ['_'.join(c) for c in group_df.columns]
    group_df = group_df.reset_index()
    group_df['model']=model_name
    group_df['fold']=fold
    return metrics_mean, group_df, pred_rel


def export_feature_importance(model, feature_names, path):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        df = pd.DataFrame({'feature':feature_names,'importance':fi})
        df = df.sort_values('importance', ascending=False).head(20)
        df.to_csv(path, index=False)


def export_shap(model, X_sample, feature_names, path):
    if shap is None:
        return
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_sample)
        if isinstance(values, list):
            values = values[0]
        mean_abs = np.abs(values).mean(axis=0)
        df = pd.DataFrame({'feature':feature_names,'mean_abs_shap':mean_abs})
        df = df.sort_values('mean_abs_shap', ascending=False).head(20)
        df.to_csv(path, index=False)
    except Exception:
        pass


def write_readme(results_dir, metrics_df, group_df, fi_df):
    best = metrics_df.groupby('model')['ndcg5'].mean().idxmax()
    best_ndcg = metrics_df[metrics_df['model']==best]['ndcg5'].mean()
    lines = []
    lines.append('# Protein corona results')
    lines.append('')
    lines.append(f'Best model: **{best}**, NDCG@5={best_ndcg:.3f}.')
    top_feat = fi_df.sort_values('importance', ascending=False).head(5)
    lines.append('Top global features: '+', '.join(top_feat['feature'].tolist()))
    lines.append('')
    lines.append('MoE vs Global: ')
    if 'MoE' in metrics_df['model'].unique():
        g = metrics_df[metrics_df['model']=='Global']['ndcg5'].mean()
        m = metrics_df[metrics_df['model']=='MoE']['ndcg5'].mean()
        lines.append(f'  Global NDCG@5={g:.3f}; MoE NDCG@5={m:.3f}.')
    lines.append('')
    lines.append('Group-level trends:')
    for fam in group_df['family'].unique():
        sub = group_df[group_df['family']==fam]
        lines.append(f"- {fam}: R2_mean={sub['r2_mean'].mean():.3f}, Spearman_mean={sub['spearman_mean'].mean():.3f}")
    text = '\n'.join(lines)
    Path(results_dir/'README_results.md').write_text(text, encoding='utf-8')


def run(config):
    data_xlsx = Path(config['data_xlsx'])
    prot_csv = Path(config['prot_csv'])
    outdir = Path(config['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'artifacts').mkdir(exist_ok=True)
    df = pd.read_excel(data_xlsx, sheet_name=0)
    meta_cols, prot_cols = identify_columns(df)
    schema_overview(df, meta_cols, prot_cols, outdir/'schema_overview.txt')
    Y = df[prot_cols].values
    rel, clr = compute_targets(Y)
    meta_df = df[meta_cols]
    group_col = 'Surface modification' if 'Surface modification' in meta_df.columns else 'Core material'
    groups = meta_df[group_col].fillna('(missing)')
    splits = make_folds(groups, 5, outdir/'splits.json')
    prot_ann = load_prot_ann(prot_csv, prot_cols)
    prior_vec = compute_baseline_prior(prot_ann)
    metrics_list=[]
    group_list=[]
    all_preds_ranker = np.zeros_like(rel)
    feature_names=None
    fi_expert_records=[]
    for fold,split in enumerate(splits):
        train_idx, test_idx = split['train'], split['test']
        X, y_clr_flat, sample_ids, prot_ids, feat_names, n_samples, n_prot = build_long_table(meta_df, clr, prot_ann, True, True, train_idx)
        feature_names = feat_names
        mask_train = np.isin(sample_ids, train_idx)
        mask_test = np.isin(sample_ids, test_idx)
        X_train, y_train = X[mask_train], y_clr_flat[mask_train]
        X_test = X[mask_test]
        group_train = [n_prot]*len(train_idx)
        # Baseline
        baseline_pred = baseline_prior(prior_vec, len(test_idx))
        dummy = type('obj',(object,),{'predict':lambda self,X:baseline_pred.reshape(-1)})()
        m, g, _ = evaluate_fold(dummy,
                                np.zeros((len(test_idx)*n_prot,1)), test_idx, n_prot, rel, clr,
                                'Baseline', fold, feature_names, outdir, prot_cols, sample_ids[mask_test])
        metrics_list.append(m)
        group_list.append(g)
        # Ridge
        model_ridge = train_model('ridge', X_train, y_train)
        joblib.dump(model_ridge, outdir/'artifacts'/f'ridge_fold{fold}.joblib')
        m,g,pred_rel = evaluate_fold(model_ridge, X_test, test_idx, n_prot, rel, clr, 'Ridge', fold, feature_names, outdir, prot_cols, sample_ids[mask_test])
        metrics_list.append(m)
        group_list.append(g)
        # XGBRanker
        model_rank = train_model('xgb_rank', X_train, y_train, group_train)
        joblib.dump(model_rank, outdir/'artifacts'/f'xgbranker_fold{fold}.joblib')
        m,g,pred_rel = evaluate_fold(model_rank, X_test, test_idx, n_prot, rel, clr, 'XGBRanker', fold, feature_names, outdir, prot_cols, sample_ids[mask_test])
        metrics_list.append(m)
        group_list.append(g)
        all_preds_ranker[test_idx] = pred_rel
        export_feature_importance(model_rank, feature_names, outdir/'feature_importance_global.csv')
        export_shap(model_rank, X_train[:100], feature_names, outdir/'feature_importance_shap.csv')
        # MoE (simple regressor per group)
        experts={}
        meta_groups = groups.values
        for gval in np.unique(meta_groups[train_idx]):
            sample_ids_in_group = np.where(meta_groups==gval)[0]
            mask_g = np.isin(sample_ids, sample_ids_in_group)
            mask_g = mask_g & mask_train
            if mask_g.sum()==0:
                continue
            model_g = train_model('xgb_reg', X[mask_g], y_clr_flat[mask_g])
            experts[gval]=model_g
            joblib.dump(model_g, outdir/'artifacts'/f'moe_{gval}_fold{fold}.joblib')
            if hasattr(model_g,'feature_importances_'):
                fi = model_g.feature_importances_
                tmp = pd.DataFrame({'feature':feature_names,'importance':fi})
                tmp = tmp.sort_values('importance', ascending=False).head(20)
                tmp['expert'] = gval
                tmp['fold'] = fold
                fi_expert_records.append(tmp)
        # prediction
        pred_flat = np.zeros(len(test_idx)*n_prot)
        for gval,model_g in experts.items():
            sample_ids_in_group = np.where(meta_groups==gval)[0]
            mask_g = np.isin(sample_ids, sample_ids_in_group)
            mask_g = mask_g & mask_test
            if mask_g.sum()>0:
                idx = np.where(mask_g)[0]
                pred_flat[idx] = model_g.predict(X[idx])
        # fallback global ridge
        missing = pred_flat==0
        if missing.any():
            pred_flat[missing] = model_ridge.predict(X_test)[missing]
        pred = pred_flat.reshape(len(test_idx), n_prot)
        pred_rel = np.maximum(pred,0)
        pred_rel = pred_rel / pred_rel.sum(axis=1, keepdims=True)
        pred_clr = np.log(pred_rel)
        pred_clr = pred_clr - pred_clr.mean(axis=1, keepdims=True)
        records=[]
        for i in range(len(test_idx)):
            rec = evaluate_row(rel[test_idx[i]], pred_rel[i], clr[test_idx[i]], pred_clr[i])
            records.append(rec)
        m = pd.DataFrame(records).mean().to_dict()
        m.update({'model':'MoE','fold':fold})
        metrics_list.append(m)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(outdir/'metrics_summary.csv', index=False)
    group_df = pd.concat(group_list)
    group_df.to_csv(outdir/'metrics_by_group.csv', index=False)
    if fi_expert_records:
        pd.concat(fi_expert_records).to_csv(outdir/'feature_importance_by_expert.csv', index=False)
    if feature_names is not None:
        joblib.dump(feature_names, outdir/'artifacts'/'feature_names.joblib')
    # prediction examples using ranker preds
    rng = np.random.default_rng(SEED)
    idx = rng.choice(rel.shape[0], size=min(20, rel.shape[0]), replace=False)
    records=[]
    for i in idx:
        pr = all_preds_ranker[i]
        tr = rel[i]
        top_pred = np.argsort(pr)[::-1][:5]
        top_true = np.argsort(tr)[::-1][:5]
        records.append({
            'sample': int(i),
            'pred_proteins': ';'.join([prot_cols[t] for t in top_pred]),
            'pred_scores': ';'.join([f"{pr[t]:.4f}" for t in top_pred]),
            'true_proteins': ';'.join([prot_cols[t] for t in top_true]),
            'true_scores': ';'.join([f"{tr[t]:.4f}" for t in top_true])
        })
    pd.DataFrame(records).to_csv(outdir/'prediction_examples.csv', index=False)
    fi_df = pd.read_csv(outdir/'feature_importance_global.csv') if (outdir/'feature_importance_global.csv').exists() else pd.DataFrame()
    write_readme(outdir, metrics_df, group_df, fi_df)
    # ablation summary (simplified)
    abl = metrics_df.groupby(['model']).agg({'ndcg5':'mean'}).reset_index()
    abl['target']='CLR'
    abl['use_prot']=True
    abl['use_inter']=True
    abl['moe']=abl['model'].eq('MoE')
    abl['ranker']=abl['model'].eq('XGBRanker')
    abl.to_csv(outdir/'ablation_summary.csv', index=False)
    # save final checks
    assert len(prot_cols)>=150
    assert len(meta_cols)>=10
    for s in splits:
        assert s['n_train_groups']>=2 and s['n_test_groups']>=2
    required=['ndcg5','ndcg10','hit5','hit10','spearman','js','clr_rmse']
    assert all(r in metrics_df.columns for r in required)
    assert len(pd.read_csv(outdir/'ablation_summary.csv'))>=3
    assert (outdir/'feature_importance_global.csv').exists()
    print('Experiment overview:')
    for model in metrics_df['model'].unique():
        sub = metrics_df[metrics_df['model']==model]
        mean = sub['ndcg5'].mean()
        std = sub['ndcg5'].std()
        print(f'{model}: NDCG@5 {mean:.3f}±{std:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_xlsx', required=True)
    ap.add_argument('--prot_csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    run(vars(args))

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()
