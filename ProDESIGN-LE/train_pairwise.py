"""
Train a pairwise similarity regression model using concatenated protein embeddings.

支持两种标签模式：
- sequence_identity: 序列一致性
- abundance_similarity: 丰度模式相关系数（何学长要求的"功能相似度"）

输入：
- ESM2 序列嵌入 (`emb_seq_esm2_t33.csv`)
- LE100 (`emb_struc_le100.csv`) 与 LE21 (`emb_struc_le21.csv`)（若存在）
- 标签文件（可自定义标签列名）

输出：
- 模型权重 (`pairwise_similarity_model.pt`)
- 训练指标 (`pairwise_similarity_metrics.json`)
- 配置文件 (`pairwise_similarity_config.json`)
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("pairwise-train")


def load_embeddings(out_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    """合并序列与结构嵌入，返回 protein_id 顺序、矩阵与列名。"""
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

    return protein_ids, emb_matrix, emb_cols


def load_labels(labels_path: Path, label_col: str) -> pd.DataFrame:
    """读取 pairwise 标签表。"""
    if labels_path.suffix == ".parquet":
        df = pd.read_parquet(labels_path)
    else:
        df = pd.read_csv(labels_path)
    
    required_cols = {"protein_i", "protein_j", label_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"标签文件缺少列：{required_cols - set(df.columns)}")
    
    # 统一重命名标签列为 "label"
    df = df.rename(columns={label_col: "label"})
    return df


def split_pairs(
    pairs: pd.DataFrame,
    protein_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_pairs: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按蛋白划分训练/验证/测试集，若验证集过小则退化为行级拆分。"""
    rng = np.random.default_rng(seed)
    pid_arr = np.array(protein_ids)
    permuted = pid_arr[rng.permutation(len(pid_arr))]

    train_count = int(len(permuted) * train_ratio)
    val_count = int(len(permuted) * val_ratio)

    train_ids = set(permuted[:train_count])
    val_ids = set(permuted[train_count : train_count + val_count])
    test_ids = set(permuted[train_count + val_count :])

    def assign_split(row: pd.Series) -> str:
        i_id, j_id = row["protein_i"], row["protein_j"]
        if i_id in train_ids and j_id in train_ids:
            return "train"
        if i_id in val_ids and j_id in val_ids:
            return "val"
        if i_id in test_ids and j_id in test_ids:
            return "test"
        return "drop"

    pairs = pairs.copy()
    pairs["split"] = pairs.apply(assign_split, axis=1)

    train_df = pairs[pairs["split"] == "train"].reset_index(drop=True)
    val_df = pairs[pairs["split"] == "val"].reset_index(drop=True)
    test_df = pairs[pairs["split"] == "test"].reset_index(drop=True)

    if len(val_df) < min_pairs or len(test_df) < min_pairs:
        LOGGER.warning(
            "按蛋白划分后的验证/测试样本太少 (val=%d, test=%d)，改用行级随机拆分。",
            len(val_df),
            len(test_df),
        )
        shuffled = pairs[pairs["split"] != "drop"].sample(
            frac=1.0, random_state=seed
        )
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        train_df = shuffled.iloc[:train_end].reset_index(drop=True)
        val_df = shuffled.iloc[train_end:val_end].reset_index(drop=True)
        test_df = shuffled.iloc[val_end:].reset_index(drop=True)

    LOGGER.info(
        "数据集规模：train=%d, val=%d, test=%d (总计=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(train_df) + len(val_df) + len(test_df),
    )
    return train_df, val_df, test_df


class PairwiseProteinDataset(Dataset):
    """按需拼接 Embedding 的 Dataset。"""

    def __init__(
        self,
        df: pd.DataFrame,
        protein_ids: List[str],
        emb_matrix: np.ndarray,
    ):
        self.df = df.reset_index(drop=True)
        self.pid_to_idx: Dict[str, int] = {pid: idx for idx, pid in enumerate(protein_ids)}
        self.emb_matrix = emb_matrix
        
        # 过滤掉不在 embedding 中的蛋白对
        valid_mask = self.df["protein_i"].isin(self.pid_to_idx) & self.df["protein_j"].isin(self.pid_to_idx)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            LOGGER.warning(f"过滤掉 {n_invalid} 个无效蛋白对（蛋白 ID 不在 embedding 中）")
            self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        i_vec = self.emb_matrix[self.pid_to_idx[row["protein_i"]]]
        j_vec = self.emb_matrix[self.pid_to_idx[row["protein_j"]]]
        features = np.concatenate((i_vec, j_vec)).astype(np.float32)
        label = np.float32(row["label"])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


class PairwiseMLP(nn.Module):
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            preds = model(features)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(labels.cpu().numpy())
    preds_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    mse = float(np.mean((preds_all - targets_all) ** 2))
    mae = float(np.mean(np.abs(preds_all - targets_all)))
    if preds_all.std() == 0 or targets_all.std() == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds_all, targets_all)[0, 1])
    return {"mse": mse, "mae": mae, "pearson": corr}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise similarity model.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("ProDESIGN-LE/out"),
        help="输出目录（也是嵌入所在目录）。",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ProDESIGN-LE/out/pairwise_abundance_similarity_long.csv"),
        help="pairwise 标签文件（CSV 或 Parquet）。",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="abundance_similarity",
        help="标签列名（如 sequence_identity 或 abundance_similarity）。",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="abundance",
        help="模型输出文件后缀（如 abundance 或 sequence）。",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=[1024, 512, 128]
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    LOGGER.info(f"使用标签文件: {args.labels}")
    LOGGER.info(f"标签列: {args.label_col}")

    protein_ids, emb_matrix, emb_cols = load_embeddings(args.out_dir)
    label_df = load_labels(args.labels, args.label_col)
    
    # 过滤标签中不在 embedding 中的蛋白
    valid_pids = set(protein_ids)
    label_df = label_df[
        label_df["protein_i"].isin(valid_pids) & label_df["protein_j"].isin(valid_pids)
    ].reset_index(drop=True)
    LOGGER.info(f"有效标签数: {len(label_df)}")
    
    train_df, val_df, test_df = split_pairs(
        label_df,
        protein_ids,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    train_dataset = PairwiseProteinDataset(train_df, protein_ids, emb_matrix)
    val_dataset = PairwiseProteinDataset(val_df, protein_ids, emb_matrix)
    test_dataset = PairwiseProteinDataset(test_df, protein_ids, emb_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    input_dim = emb_matrix.shape[1] * 2
    model = PairwiseMLP(input_dim, args.hidden_dims, args.dropout)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not args.cpu
        else torch.device("cpu")
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f, val_mse=%.4f, val_mae=%.4f, val_pearson=%.3f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["mse"],
            val_metrics["mae"],
            val_metrics["pearson"],
        )
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("训练失败：未获得有效模型。")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    LOGGER.info(
        "测试集指标：mse=%.4f, mae=%.4f, pearson=%.3f",
        test_metrics["mse"],
        test_metrics["mae"],
        test_metrics["pearson"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"pairwise_{args.model_suffix}_model.pt"
    torch.save(best_state, model_path)

    metrics = {
        "test": test_metrics,
        "best_val_mse": best_val,
        "history": history,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "label_col": args.label_col,
        "labels_file": str(args.labels),
    }

    metrics_path = args.out_dir / f"pairwise_{args.model_suffix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "embedding_columns": emb_cols,
        "labels_file": str(args.labels),
        "label_col": args.label_col,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    config_path = args.out_dir / f"pairwise_{args.model_suffix}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    LOGGER.info("模型保存在 %s，指标保存在 %s", model_path, metrics_path)


if __name__ == "__main__":
    main()
