"""
未知蛋白功能推断流程

实现何学长的核心想法：
- 输入：一个未知蛋白的序列/结构（或 embedding）
- 输出：功能标签预测（通过找到与它"丰度模式相似"的已知蛋白）

流程：
1. 加载未知蛋白的 ESM2+LE embedding（或直接提供）
2. 用 pairwise 模型与所有已知蛋白配对打分
3. 取 Top-K 最相似的已知蛋白
4. 读取这些已知蛋白的功能标签（Protein class, Biological process 等）
5. 投票/加权得出未知蛋白的功能预测
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("infer-function")


class PairwiseMLP(nn.Module):
    """与 train_pairwise.py 中的模型结构一致"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_embeddings(out_dir: Path) -> Tuple[List[str], np.ndarray, StandardScaler]:
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
        else:
            LOGGER.warning("未找到结构嵌入文件：%s，按 0 填充", extra.name)

    emb_cols = [c for c in df.columns if c.startswith(("esm2_", "le100_", "le21_"))]
    df = df.sort_values("protein_id").reset_index(drop=True)
    df[emb_cols] = df[emb_cols].fillna(0.0)

    scaler = StandardScaler()
    emb_matrix = scaler.fit_transform(df[emb_cols].values.astype(np.float32))
    protein_ids = df["protein_id"].astype(str).tolist()

    return protein_ids, emb_matrix, scaler


def load_annotations(ann_path: Path) -> pd.DataFrame:
    """加载蛋白功能注释"""
    df = pd.read_csv(ann_path)
    # 识别 UniProt ID 列
    if "Uniprot" in df.columns:
        df = df.rename(columns={"Uniprot": "protein_id"})
    elif "protein_id" not in df.columns:
        # 尝试从第一列提取
        first_col = df.columns[0]
        df["protein_id"] = df[first_col].astype(str)
    return df


def load_model(
    model_path: Path, metrics_path: Path, device: torch.device
) -> Tuple[PairwiseMLP, Dict]:
    """加载训练好的模型"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    input_dim = metrics["input_dim"]
    hidden_dims = metrics["hidden_dims"]
    dropout = metrics.get("dropout", 0.1)
    
    model = PairwiseMLP(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, metrics


def compute_similarity_scores(
    query_emb: np.ndarray,
    known_embs: np.ndarray,
    model: PairwiseMLP,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """计算查询蛋白与所有已知蛋白的相似度分数"""
    n_known = known_embs.shape[0]
    scores = np.zeros(n_known, dtype=np.float32)
    
    for i in range(0, n_known, batch_size):
        end = min(i + batch_size, n_known)
        batch_embs = known_embs[i:end]
        
        # 拼接 query 和 batch
        query_repeated = np.tile(query_emb, (end - i, 1))
        features = np.concatenate([query_repeated, batch_embs], axis=1).astype(np.float32)
        
        with torch.no_grad():
            features_t = torch.from_numpy(features).to(device)
            batch_scores = model(features_t).cpu().numpy()
        
        scores[i:end] = batch_scores
    
    return scores


def infer_function(
    scores: np.ndarray,
    protein_ids: List[str],
    annotations: pd.DataFrame,
    top_k: int = 5,
    annotation_cols: Optional[List[str]] = None,
) -> Dict:
    """基于相似度分数推断功能"""
    if annotation_cols is None:
        annotation_cols = ["Protein class", "Biological process", "Molecular function"]
    
    # 获取 Top-K 最相似的蛋白
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_proteins = [(protein_ids[i], float(scores[i])) for i in top_indices]
    
    # 收集 Top-K 蛋白的功能标签
    results = {
        "top_similar_proteins": top_proteins,
        "function_predictions": {},
    }
    
    for col in annotation_cols:
        if col not in annotations.columns:
            continue
        
        # 收集所有 Top-K 蛋白的该列标签
        all_labels = []
        for idx in top_indices:
            pid = protein_ids[idx]
            ann_row = annotations[annotations["protein_id"] == pid]
            if not ann_row.empty:
                label = ann_row[col].values[0]
                if pd.notna(label) and str(label).strip():
                    # 处理多值标签（用逗号分隔）
                    for sub_label in str(label).split(","):
                        sub_label = sub_label.strip()
                        if sub_label:
                            all_labels.append(sub_label)
        
        # 统计标签频率
        if all_labels:
            label_counts = Counter(all_labels)
            results["function_predictions"][col] = {
                "top_labels": label_counts.most_common(5),
                "all_labels": all_labels,
            }
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer protein function from pairwise similarity.")
    parser.add_argument(
        "--query_protein",
        type=str,
        default=None,
        help="查询蛋白的 UniProt ID（必须在已知蛋白列表中）",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("ProDESIGN-LE/out"),
        help="包含 embedding 和模型的目录",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="abundance",
        help="模型后缀（如 abundance 或 sequence）",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("ProDESIGN-LE/out/protein_index.csv"),
        help="蛋白功能注释文件",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="返回 Top-K 最相似的蛋白",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示模式：随机选择一个蛋白作为未知蛋白",
    )
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not args.cpu
        else torch.device("cpu")
    )
    LOGGER.info(f"使用设备: {device}")
    
    # 加载 embedding
    LOGGER.info("加载 embedding...")
    protein_ids, emb_matrix, scaler = load_embeddings(args.out_dir)
    LOGGER.info(f"已加载 {len(protein_ids)} 个蛋白的 embedding")
    
    # 加载模型
    model_path = args.out_dir / f"pairwise_{args.model_suffix}_model.pt"
    metrics_path = args.out_dir / f"pairwise_{args.model_suffix}_metrics.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    
    LOGGER.info(f"加载模型: {model_path}")
    model, metrics = load_model(model_path, metrics_path, device)
    
    # 加载注释
    LOGGER.info(f"加载注释: {args.annotations}")
    annotations = load_annotations(args.annotations)
    
    # 确定查询蛋白
    if args.demo or args.query_protein is None:
        # 演示模式：随机选择一个蛋白
        np.random.seed(42)
        query_idx = np.random.randint(len(protein_ids))
        query_protein = protein_ids[query_idx]
        LOGGER.info(f"演示模式：随机选择蛋白 {query_protein} 作为未知蛋白")
    else:
        query_protein = args.query_protein
        if query_protein not in protein_ids:
            raise ValueError(f"蛋白 {query_protein} 不在已知蛋白列表中")
        query_idx = protein_ids.index(query_protein)
    
    # 获取查询蛋白的 embedding
    query_emb = emb_matrix[query_idx]
    
    # 计算与所有蛋白的相似度
    LOGGER.info("计算相似度分数...")
    scores = compute_similarity_scores(query_emb, emb_matrix, model, device)
    
    # 将查询蛋白自身的分数设为负无穷（排除自己）
    scores[query_idx] = float("-inf")
    
    # 推断功能
    LOGGER.info(f"推断功能（Top-{args.top_k}）...")
    results = infer_function(
        scores,
        protein_ids,
        annotations,
        top_k=args.top_k,
        annotation_cols=["Protein class", "Biological process", "Molecular function", "Disease involvement"],
    )
    
    # 添加查询蛋白信息
    results["query_protein"] = query_protein
    results["query_protein_annotations"] = {}
    query_ann = annotations[annotations["protein_id"] == query_protein]
    if not query_ann.empty:
        for col in ["Protein class", "Biological process", "Molecular function", "Disease involvement"]:
            if col in query_ann.columns:
                val = query_ann[col].values[0]
                if pd.notna(val):
                    results["query_protein_annotations"][col] = str(val)
    
    # 输出结果
    print("\n" + "=" * 60)
    print(f"查询蛋白: {query_protein}")
    print("=" * 60)
    
    if results["query_protein_annotations"]:
        print("\n【真实标签（用于验证）】")
        for k, v in results["query_protein_annotations"].items():
            print(f"  {k}: {v[:100]}..." if len(str(v)) > 100 else f"  {k}: {v}")
    
    print(f"\n【Top-{args.top_k} 最相似的已知蛋白】")
    for pid, score in results["top_similar_proteins"]:
        print(f"  {pid}: 相似度 = {score:.4f}")
    
    print("\n【功能预测（基于 Top-K 投票）】")
    for col, pred in results["function_predictions"].items():
        print(f"\n  {col}:")
        for label, count in pred["top_labels"]:
            print(f"    - {label}: {count} 票")
    
    print("\n" + "=" * 60)
    
    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"结果已保存到: {args.output}")
    else:
        default_output = args.out_dir / f"infer_function_{query_protein}.json"
        with open(default_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"结果已保存到: {default_output}")


if __name__ == "__main__":
    main()

