"""
Compute pairwise similarity labels for all proteins.

支持两种模式：
- sequence: 序列一致性（默认）
- abundance: 丰度模式相关系数（何学长要求的"功能相似度"）

输出：
1. 长表 `pairwise_{mode}_similarity_long.csv`
2. 矩阵 `pairwise_{mode}_similarity.csv`
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from Bio import pairwise2
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


# ============== 序列相似度模式 ==============

def load_sequences(index_path: Path) -> Tuple[List[str], List[str]]:
    """Read protein_id 和 sequence 列，过滤缺失序列。"""
    df = pd.read_csv(index_path)
    df = df[df["sequence"].notna()].copy()
    df = df.sort_values("protein_id").reset_index(drop=True)
    protein_ids = df["protein_id"].astype(str).tolist()
    sequences = df["sequence"].astype(str).tolist()
    if not protein_ids:
        raise ValueError("未在 index 表中找到可用的序列。")
    return protein_ids, sequences


def compute_identity(seq_a: str, seq_b: str) -> float:
    """使用全局比对得到序列一致性。"""
    if not HAS_BIOPYTHON:
        raise ImportError("需要安装 biopython 才能计算序列一致性。")
    if not seq_a or not seq_b:
        return float("nan")
    # match=2, mismatch=-1, gap_open=-2, gap_extend=-0.5
    alignment = pairwise2.align.globalms(
        seq_a,
        seq_b,
        2,
        -1,
        -2,
        -0.5,
        one_alignment_only=True,
    )[0]
    aligned_a, aligned_b = alignment.seqA, alignment.seqB
    matches = sum(aa == bb for aa, bb in zip(aligned_a, aligned_b))
    alignment_len = len(aligned_a)
    return matches / alignment_len if alignment_len else 0.0


def compute_sequence_matrix(
    protein_ids: List[str], sequences: List[str]
) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
    """生成序列一致性的对称矩阵与长表条目。"""
    n = len(protein_ids)
    matrix = np.zeros((n, n), dtype=np.float32)
    long_entries: List[Tuple[str, str, float]] = []

    for i in tqdm(range(n), desc="Computing pairwise sequence identity"):
        for j in range(i, n):
            identity = compute_identity(sequences[i], sequences[j])
            matrix[i, j] = identity
            matrix[j, i] = identity
            long_entries.append((protein_ids[i], protein_ids[j], identity))
            if i != j:
                long_entries.append((protein_ids[j], protein_ids[i], identity))
    return matrix, long_entries


# ============== 丰度模式相似度模式 ==============

def extract_uniprot_ac(protein_name: str) -> Optional[str]:
    """从蛋白名称中提取 UniProt accession code。"""
    uniprot_pattern = r'^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$'
    match = re.match(uniprot_pattern, protein_name)
    if match:
        return match.group()
    parts = protein_name.split()
    if parts:
        candidate = parts[0]
        match = re.match(uniprot_pattern, candidate)
        if match:
            return match.group()
    return None


def load_abundance_vectors(imputed_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    从 imputed.xlsx 加载每个蛋白的丰度向量。
    
    返回：
    - protein_ids: 蛋白 ID 列表
    - abundance_matrix: shape (n_proteins, n_conditions)，每行是一个蛋白在所有条件下的丰度
    """
    df = pd.read_excel(imputed_path)
    
    # 识别蛋白列
    protein_columns = []
    protein_ids = []
    for col in df.columns:
        uniprot_ac = extract_uniprot_ac(col)
        if uniprot_ac:
            protein_columns.append(col)
            protein_ids.append(uniprot_ac)
    
    if not protein_columns:
        raise ValueError("未在 imputed.xlsx 中找到蛋白列。")
    
    # 提取丰度矩阵：每列是一个蛋白，每行是一个条件
    abundance_df = df[protein_columns].copy()
    # 转置：每行是一个蛋白，每列是一个条件
    abundance_matrix = abundance_df.values.T.astype(np.float32)
    
    # 处理缺失值：用列均值填充
    for i in range(abundance_matrix.shape[0]):
        row = abundance_matrix[i]
        mask = np.isnan(row)
        if mask.any():
            mean_val = np.nanmean(row)
            abundance_matrix[i, mask] = mean_val if not np.isnan(mean_val) else 0.0
    
    # 按 protein_id 排序
    sorted_indices = np.argsort(protein_ids)
    protein_ids = [protein_ids[i] for i in sorted_indices]
    abundance_matrix = abundance_matrix[sorted_indices]
    
    print(f"加载了 {len(protein_ids)} 个蛋白，每个蛋白有 {abundance_matrix.shape[1]} 个条件的丰度值。")
    return protein_ids, abundance_matrix


def compute_abundance_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个丰度向量的 Pearson 相关系数。"""
    # 过滤掉两边都是 NaN 的位置
    mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
    if mask.sum() < 2:
        return 0.0
    a = vec_a[mask]
    b = vec_b[mask]
    
    # 计算 Pearson 相关系数
    if a.std() == 0 or b.std() == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_abundance_matrix(
    protein_ids: List[str], abundance_matrix: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
    """生成丰度模式相似度的对称矩阵与长表条目。"""
    n = len(protein_ids)
    matrix = np.zeros((n, n), dtype=np.float32)
    long_entries: List[Tuple[str, str, float]] = []

    for i in tqdm(range(n), desc="Computing pairwise abundance similarity"):
        for j in range(i, n):
            sim = compute_abundance_similarity(abundance_matrix[i], abundance_matrix[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
            long_entries.append((protein_ids[i], protein_ids[j], sim))
            if i != j:
                long_entries.append((protein_ids[j], protein_ids[i], sim))
    return matrix, long_entries


# ============== 保存输出 ==============

def save_outputs(
    protein_ids: List[str],
    matrix: np.ndarray,
    long_entries: List[Tuple[str, str, float]],
    out_dir: Path,
    mode: str,
    label_col: str,
) -> None:
    """保存矩阵和长表。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存矩阵
    matrix_df = pd.DataFrame(matrix, index=protein_ids, columns=protein_ids)
    matrix_path = out_dir / f"pairwise_{mode}_similarity.csv"
    matrix_df.to_csv(matrix_path, index=True)

    # 保存长表
    long_df = pd.DataFrame(
        long_entries, columns=["protein_i", "protein_j", label_col]
    )
    if HAS_PARQUET:
        long_path = out_dir / f"pairwise_{mode}_similarity.parquet"
        long_df.to_parquet(long_path, index=False)
        long_path_str = str(long_path)
    else:
        long_path = out_dir / f"pairwise_{mode}_similarity_long.csv"
        long_df.to_csv(long_path, index=False)
        long_path_str = str(long_path)

    # 简要统计
    stats = {
        "mode": mode,
        "proteins": len(protein_ids),
        "rows_matrix": len(matrix_df),
        "rows_long": len(long_df),
        "long_path": long_path_str,
        "matrix_path": str(matrix_path),
        "parquet_enabled": HAS_PARQUET,
        "similarity_mean": float(matrix[np.triu_indices(len(protein_ids), k=1)].mean()),
        "similarity_std": float(matrix[np.triu_indices(len(protein_ids), k=1)].std()),
    }
    stats_path = out_dir / f"pairwise_{mode}_similarity.json"
    pd.Series(stats).to_json(stats_path, indent=2)
    
    print(f"\n=== {mode.upper()} 模式输出 ===")
    print(f"矩阵: {matrix_path}")
    print(f"长表: {long_path_str}")
    print(f"统计: {stats_path}")
    print(f"相似度均值: {stats['similarity_mean']:.4f}, 标准差: {stats['similarity_std']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pairwise similarity labels (sequence or abundance mode)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequence", "abundance"],
        default="abundance",
        help="相似度计算模式：sequence（序列一致性）或 abundance（丰度模式相关系数）。",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("ProDESIGN-LE/out/protein_index.csv"),
        help="包含 protein_id 和 sequence 的 CSV（sequence 模式需要）。",
    )
    parser.add_argument(
        "--imputed",
        type=Path,
        default=Path("imputed.xlsx"),
        help="imputed.xlsx 宽表路径（abundance 模式需要）。",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("ProDESIGN-LE/out"),
        help="输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.mode == "sequence":
        protein_ids, sequences = load_sequences(args.index)
        matrix, long_entries = compute_sequence_matrix(protein_ids, sequences)
        save_outputs(protein_ids, matrix, long_entries, args.out_dir, "sequence", "sequence_identity")
    
    elif args.mode == "abundance":
        protein_ids, abundance_matrix = load_abundance_vectors(args.imputed)
        matrix, long_entries = compute_abundance_matrix(protein_ids, abundance_matrix)
        save_outputs(protein_ids, matrix, long_entries, args.out_dir, "abundance", "abundance_similarity")


if __name__ == "__main__":
    main()
