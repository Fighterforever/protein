"""
蛋白对相似度预测 - 多模型对比训练

支持三种模型：MLP、XGBoost、Random Forest
支持两种任务：回归（预测相似度）、分类（预测是否相似）

输出指标（对标文献）：
- 回归：MSE, MAE, R², Pearson
- 分类：AUC, Accuracy, Precision, Recall, F1

文献对比基准：
- Nanoscale Advances 2025: R² = 0.45~0.88 (Random Forest 回归)
- ACS Nano 2025 (UC Riverside): AUC > 0.85 (XGBoost/LightGBM 分类)
- ACS Nano 2025 (UC Berkeley): AUC = 0.97, Accuracy = 92% (XGBoost 分类)
"""
import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import pearsonr

# 可选依赖
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    HAS_RF = True
except ImportError:
    HAS_RF = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("model-comparison")


# ============== 数据加载 ==============

def load_embeddings(out_dir: Path) -> Tuple[List[str], np.ndarray]:
    """加载并标准化 embedding"""
    seq_path = out_dir / "emb_seq_esm2_t33.csv"
    le100_path = out_dir / "emb_struc_le100.csv"
    le21_path = out_dir / "emb_struc_le21.csv"

    if not seq_path.exists():
        raise FileNotFoundError(f"缺少序列嵌入文件：{seq_path}")

    df = pd.read_csv(seq_path)
    for extra in [le100_path, le21_path]:
        if extra.exists():
            df = df.merge(pd.read_csv(extra), on="protein_id", how="left")

    emb_cols = [c for c in df.columns if c.startswith(("esm2_", "le100_", "le21_"))]
    df = df.sort_values("protein_id").reset_index(drop=True)
    df[emb_cols] = df[emb_cols].fillna(0.0)

    scaler = StandardScaler()
    emb_matrix = scaler.fit_transform(df[emb_cols].values.astype(np.float32))
    protein_ids = df["protein_id"].astype(str).tolist()

    return protein_ids, emb_matrix


def load_labels(labels_path: Path, label_col: str) -> pd.DataFrame:
    """加载标签"""
    if labels_path.suffix == ".parquet":
        df = pd.read_parquet(labels_path)
    else:
        df = pd.read_csv(labels_path)
    
    required_cols = {"protein_i", "protein_j", label_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"标签文件缺少列：{required_cols - set(df.columns)}")
    
    df = df.rename(columns={label_col: "label"})
    return df


def prepare_data(
    protein_ids: List[str],
    emb_matrix: np.ndarray,
    label_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """准备训练/验证/测试数据"""
    pid_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
    
    # 过滤有效蛋白对
    valid_mask = (
        label_df["protein_i"].isin(pid_to_idx) & 
        label_df["protein_j"].isin(pid_to_idx)
    )
    label_df = label_df[valid_mask].reset_index(drop=True)
    
    # 移除自身配对
    label_df = label_df[label_df["protein_i"] != label_df["protein_j"]].reset_index(drop=True)
    
    LOGGER.info(f"有效蛋白对数量：{len(label_df)}")
    
    # 构建特征矩阵
    n_samples = len(label_df)
    n_features = emb_matrix.shape[1] * 2
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = label_df["label"].values.astype(np.float32)
    
    for i, row in label_df.iterrows():
        idx_i = pid_to_idx[row["protein_i"]]
        idx_j = pid_to_idx[row["protein_j"]]
        X[i] = np.concatenate([emb_matrix[idx_i], emb_matrix[idx_j]])
    
    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=seed
    )
    
    LOGGER.info(f"数据集大小：train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ============== 评估指标 ==============

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """回归任务评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson 相关系数
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        pearson, _ = pearsonr(y_true, y_pred)
    else:
        pearson = 0.0
    
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson": float(pearson),
    }


def evaluate_classification(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """分类任务评估指标"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def convert_to_classification(
    y: np.ndarray, 
    threshold: float = 0.3,
) -> np.ndarray:
    """将回归标签转换为分类标签
    
    threshold: 丰度相似度阈值，大于此值视为"相似"
    文献中通常用 0.3 或 0.5 作为阈值
    """
    return (y >= threshold).astype(int)


# ============== MLP 模型 ==============

class PairwiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    data: Dict[str, np.ndarray],
    hidden_dims: List[int] = [1024, 512, 128],
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    task: str = "regression",
    device: str = "cpu",
) -> Tuple[Any, Dict]:
    """训练 MLP 模型"""
    if not HAS_TORCH:
        raise ImportError("需要安装 PyTorch")
    
    device = torch.device(device)
    
    X_train = torch.FloatTensor(data["X_train"])
    X_val = torch.FloatTensor(data["X_val"])
    X_test = torch.FloatTensor(data["X_test"])
    
    if task == "classification":
        y_train = torch.FloatTensor(convert_to_classification(data["y_train"]))
        y_val_cls = convert_to_classification(data["y_val"])
        y_test_cls = convert_to_classification(data["y_test"])
    else:
        y_train = torch.FloatTensor(data["y_train"])
    
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = PairwiseMLP(input_dim, hidden_dims, dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    best_val_metric = float("-inf") if task == "classification" else float("-inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # 初始化
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        train_loss = total_loss / len(train_dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val.to(device)).cpu().numpy()
        
        if task == "classification":
            val_pred_proba = 1 / (1 + np.exp(-val_output))  # sigmoid
            val_metrics = evaluate_classification(y_val_cls, val_pred_proba)
            val_metric = val_metrics["AUC"]
            is_better = val_metric > best_val_metric
        else:
            val_metrics = evaluate_regression(y_val, val_output)
            val_metric = -val_metrics["MSE"]  # 负 MSE，越大越好
            is_better = val_metric > best_val_metric
        
        if is_better:
            best_val_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        
        if epoch % 10 == 0:
            if task == "classification":
                LOGGER.info(f"Epoch {epoch}/{epochs} - loss={train_loss:.4f}, val_AUC={val_metrics['AUC']:.4f}")
            else:
                LOGGER.info(f"Epoch {epoch}/{epochs} - loss={train_loss:.4f}, val_R2={val_metrics['R2']:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    model.eval()
    
    # 测试集评估
    with torch.no_grad():
        test_output = model(X_test.to(device)).cpu().numpy()
    
    if task == "classification":
        test_pred_proba = 1 / (1 + np.exp(-test_output))
        test_metrics = evaluate_classification(y_test_cls, test_pred_proba)
        # 同时计算回归指标
        test_metrics.update({f"Reg_{k}": v for k, v in evaluate_regression(y_test, test_output).items()})
    else:
        test_metrics = evaluate_regression(y_test, test_output)
        # 同时计算分类指标（用阈值 0.3）
        y_test_cls = convert_to_classification(y_test)
        test_pred_proba = (test_output - test_output.min()) / (test_output.max() - test_output.min() + 1e-8)
        test_metrics.update({f"Cls_{k}": v for k, v in evaluate_classification(y_test_cls, test_pred_proba).items()})
    
    return model, {
        "model": "MLP",
        "task": task,
        "test_metrics": test_metrics,
        "history": history,
        "best_val_metric": float(best_val_metric),
        "test_actual": y_test.tolist(),
        "test_predicted": test_output.tolist(),
    }


# ============== XGBoost 模型 ==============

def train_xgboost(
    data: Dict[str, np.ndarray],
    task: str = "regression",
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[Any, Dict]:
    """训练 XGBoost 模型"""
    if not HAS_XGBOOST:
        raise ImportError("需要安装 xgboost: pip install xgboost")
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    if task == "classification":
        y_train_cls = convert_to_classification(y_train)
        y_val_cls = convert_to_classification(y_val)
        y_test_cls = convert_to_classification(y_test)
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            eval_metric="auc",
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(
            X_train, y_train_cls,
            eval_set=[(X_val, y_val_cls)],
            verbose=False,
        )
        
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_classification(y_test_cls, test_pred_proba)
        
        # 同时计算回归指标（用概率作为预测值）
        test_metrics.update({f"Reg_{k}": v for k, v in evaluate_regression(y_test, test_pred_proba).items()})
        
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        test_pred = model.predict(X_test)
        test_metrics = evaluate_regression(y_test, test_pred)
        
        # 同时计算分类指标
        y_test_cls = convert_to_classification(y_test)
        test_pred_proba = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8)
        test_metrics.update({f"Cls_{k}": v for k, v in evaluate_classification(y_test_cls, test_pred_proba).items()})
        
        test_output = test_pred  # For saving
    
    return model, {
        "model": "XGBoost",
        "task": task,
        "test_metrics": test_metrics,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "test_actual": y_test.tolist(),
        "test_predicted": test_output.tolist() if task == "regression" else test_pred_proba.tolist(),
    }


# ============== Random Forest 模型 ==============

def train_random_forest(
    data: Dict[str, np.ndarray],
    task: str = "regression",
    n_estimators: int = 200,
    max_depth: int = 15,
    seed: int = 42,
) -> Tuple[Any, Dict]:
    """训练 Random Forest 模型"""
    if not HAS_RF:
        raise ImportError("需要安装 scikit-learn")
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    if task == "classification":
        y_train_cls = convert_to_classification(y_train)
        y_test_cls = convert_to_classification(y_test)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_cls)
        
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_classification(y_test_cls, test_pred_proba)
        
        # 同时计算回归指标
        test_metrics.update({f"Reg_{k}": v for k, v in evaluate_regression(y_test, test_pred_proba).items()})
        
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        test_pred = model.predict(X_test)
        test_metrics = evaluate_regression(y_test, test_pred)
        
        # 同时计算分类指标
        y_test_cls = convert_to_classification(y_test)
        test_pred_proba = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8)
        test_metrics.update({f"Cls_{k}": v for k, v in evaluate_classification(y_test_cls, test_pred_proba).items()})
        
        test_output = test_pred  # For saving
    
    return model, {
        "model": "RandomForest",
        "task": task,
        "test_metrics": test_metrics,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        "test_actual": y_test.tolist(),
        "test_predicted": test_output.tolist() if task == "regression" else test_pred_proba.tolist(),
    }


# ============== 主函数 ==============

def parse_args():
    parser = argparse.ArgumentParser(description="多模型对比训练")
    parser.add_argument("--out_dir", type=Path, default=Path("ProDESIGN-LE/out"))
    parser.add_argument("--labels", type=Path, default=Path("ProDESIGN-LE/out/pairwise_abundance_similarity_long.csv"))
    parser.add_argument("--label_col", type=str, default="abundance_similarity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50, help="MLP 训练轮数")
    parser.add_argument("--cls_threshold", type=float, default=0.3, help="分类阈值")
    return parser.parse_args()


def main():
    args = parse_args()
    
    LOGGER.info("=" * 60)
    LOGGER.info("蛋白对相似度预测 - 多模型对比")
    LOGGER.info("=" * 60)
    
    # 加载数据
    LOGGER.info("加载 embedding...")
    protein_ids, emb_matrix = load_embeddings(args.out_dir)
    LOGGER.info(f"蛋白数量：{len(protein_ids)}，embedding 维度：{emb_matrix.shape[1]}")
    
    LOGGER.info("加载标签...")
    label_df = load_labels(args.labels, args.label_col)
    
    LOGGER.info("准备数据...")
    data = prepare_data(protein_ids, emb_matrix, label_df, seed=args.seed)
    
    # 标签分布统计
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    y_cls = convert_to_classification(y_all, args.cls_threshold)
    LOGGER.info(f"标签分布：相似={y_cls.sum()} ({y_cls.mean()*100:.1f}%), 不相似={len(y_cls)-y_cls.sum()} ({(1-y_cls.mean())*100:.1f}%)")
    
    results = []
    
    # ========== 回归任务 ==========
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("任务 1: 回归（预测丰度相似度）")
    LOGGER.info("=" * 60)
    
    # MLP 回归
    if HAS_TORCH:
        LOGGER.info("\n[1/3] 训练 MLP (回归)...")
        _, mlp_reg_result = train_mlp(data, task="regression", epochs=args.epochs)
        results.append(mlp_reg_result)
        LOGGER.info(f"MLP 回归测试集: R²={mlp_reg_result['test_metrics']['R2']:.4f}, Pearson={mlp_reg_result['test_metrics']['Pearson']:.4f}")
    
    # XGBoost 回归
    if HAS_XGBOOST:
        LOGGER.info("\n[2/3] 训练 XGBoost (回归)...")
        _, xgb_reg_result = train_xgboost(data, task="regression")
        results.append(xgb_reg_result)
        LOGGER.info(f"XGBoost 回归测试集: R²={xgb_reg_result['test_metrics']['R2']:.4f}, Pearson={xgb_reg_result['test_metrics']['Pearson']:.4f}")
    
    # Random Forest 回归
    if HAS_RF:
        LOGGER.info("\n[3/3] 训练 Random Forest (回归)...")
        _, rf_reg_result = train_random_forest(data, task="regression")
        results.append(rf_reg_result)
        LOGGER.info(f"Random Forest 回归测试集: R²={rf_reg_result['test_metrics']['R2']:.4f}, Pearson={rf_reg_result['test_metrics']['Pearson']:.4f}")
    
    # ========== 分类任务 ==========
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info(f"任务 2: 分类（阈值={args.cls_threshold}）")
    LOGGER.info("=" * 60)
    
    # MLP 分类
    if HAS_TORCH:
        LOGGER.info("\n[1/3] 训练 MLP (分类)...")
        _, mlp_cls_result = train_mlp(data, task="classification", epochs=args.epochs)
        results.append(mlp_cls_result)
        LOGGER.info(f"MLP 分类测试集: AUC={mlp_cls_result['test_metrics']['AUC']:.4f}, Accuracy={mlp_cls_result['test_metrics']['Accuracy']:.4f}")
    
    # XGBoost 分类
    if HAS_XGBOOST:
        LOGGER.info("\n[2/3] 训练 XGBoost (分类)...")
        _, xgb_cls_result = train_xgboost(data, task="classification")
        results.append(xgb_cls_result)
        LOGGER.info(f"XGBoost 分类测试集: AUC={xgb_cls_result['test_metrics']['AUC']:.4f}, Accuracy={xgb_cls_result['test_metrics']['Accuracy']:.4f}")
    
    # Random Forest 分类
    if HAS_RF:
        LOGGER.info("\n[3/3] 训练 Random Forest (分类)...")
        _, rf_cls_result = train_random_forest(data, task="classification")
        results.append(rf_cls_result)
        LOGGER.info(f"Random Forest 分类测试集: AUC={rf_cls_result['test_metrics']['AUC']:.4f}, Accuracy={rf_cls_result['test_metrics']['Accuracy']:.4f}")
    
    # ========== 结果汇总 ==========
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("结果汇总")
    LOGGER.info("=" * 60)
    
    print("\n### 回归任务指标 ###")
    print(f"{'模型':<15} {'MSE':<10} {'MAE':<10} {'R²':<10} {'Pearson':<10}")
    print("-" * 55)
    for r in results:
        if r["task"] == "regression":
            m = r["test_metrics"]
            print(f"{r['model']:<15} {m['MSE']:<10.4f} {m['MAE']:<10.4f} {m['R2']:<10.4f} {m['Pearson']:<10.4f}")
    
    print("\n### 分类任务指标 ###")
    print(f"{'模型':<15} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 65)
    for r in results:
        if r["task"] == "classification":
            m = r["test_metrics"]
            print(f"{r['model']:<15} {m['AUC']:<10.4f} {m['Accuracy']:<10.4f} {m['Precision']:<10.4f} {m['Recall']:<10.4f} {m['F1']:<10.4f}")
    
    print("\n### 文献对比基准 ###")
    print("Nanoscale Advances 2025 (Random Forest 回归): R² = 0.45~0.88")
    print("ACS Nano 2025 UC Riverside (XGBoost 分类): AUC > 0.85")
    print("ACS Nano 2025 UC Berkeley (XGBoost 分类): AUC = 0.97, Accuracy = 92%")
    
    # 保存结果
    output_path = args.out_dir / "model_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    LOGGER.info(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

