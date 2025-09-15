import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pc_run import (
    identify_columns,
    compute_targets,
    ndcg_at_k,
    hit_at_k,
    row_spearman,
    js_divergence,
    clr_rmse,
    evaluate_row,
)


def test_identify_columns_splits_meta_and_prot():
    df = pd.DataFrame(
        {
            'Sample': [1, 2],
            'P12345': [0.1, 0.2],
            'O65432': [0.3, 0.4],
            'meta': ['a', 'b'],
        }
    )
    meta_cols, prot_cols = identify_columns(df)
    assert set(meta_cols) == {'Sample', 'meta'}
    assert set(prot_cols) == {'P12345', 'O65432'}


def test_compute_targets_returns_rel_and_clr():
    Y = np.array([[1, 1, 2], [0, 3, 3]], dtype=float)
    rel, clr = compute_targets(Y)
    assert rel.shape == Y.shape
    np.testing.assert_allclose(rel.sum(axis=1), np.ones(Y.shape[0]))
    np.testing.assert_allclose(clr.mean(axis=1), np.zeros(Y.shape[0]), atol=1e-9)


def test_eval_metrics_individual_functions():
    y_true = np.array([3, 2, 1, 0, 0])
    y_pred = np.array([3, 1, 0, 2, 0])
    assert ndcg_at_k(y_true, y_true, k=5) == 1.0
    assert ndcg_at_k(y_true, y_pred, k=3) < 1.0
    assert hit_at_k(y_true, y_true, k=3) == 1.0
    assert hit_at_k(y_true, y_pred, k=3) < 1.0
    assert np.isclose(row_spearman(y_true, y_true), 1.0)


def test_evaluate_row_matches_components():
    counts_true = np.arange(1, 11, dtype=float)
    counts_pred = counts_true[::-1]
    rel_true, clr_true = compute_targets(counts_true.reshape(1, -1))
    rel_pred, clr_pred = compute_targets(counts_pred.reshape(1, -1))
    metrics = evaluate_row(rel_true[0], rel_pred[0], clr_true[0], clr_pred[0])
    assert metrics['js'] == js_divergence(rel_true[0], rel_pred[0])
    assert metrics['clr_rmse'] == clr_rmse(clr_true[0], clr_pred[0])
    assert metrics['spearman'] == row_spearman(rel_true[0], rel_pred[0])
    assert 0.0 <= metrics['ndcg5'] <= 1.0
    assert 0.0 <= metrics['hit5'] <= 1.0
