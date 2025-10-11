## Protein Corona Embedding & Prediction

本项目提供一条可复现实验数据到蛋白表示与下游分析的完整流水线：
- 基于 ESM2 的蛋白序列嵌入
- 基于 ProDESIGN-LE 的蛋白结构局部环境嵌入（LE100/LE21）
- 自动抓取 UniProt 序列与 AlphaFold PDB
- 多源特征合并、长表构建与快速评估
- 从既有产物直接“生成统计”而无需重算（gen-stats）

目录中 `ProDESIGN-LE/` 既包含原始 ProDESIGN-LE 能力（序列设计/evaluator/预处理）也包含本项目适配后的嵌入流水线脚本。


### 目录结构
- `ProDESIGN-LE/`：序列/结构嵌入与 ProDESIGN-LE 模型代码
  - `pc_embed_pipeline.py`：嵌入流水线 CLI（index/fetch/seq-embed/str-embed/merge/build-long/quick-eval/gen-stats）
  - `get_LE.py`：从 PDB 计算 LE100/LE21（需 `best.pkl`）
  - `pe/`：几何/数据/模型模块
  - `pdbs/`：AlphaFold 下载/自带的 PDB 文件
  - `out/`：流水线输出目录
- `imputed.xlsx`：蛋白冠宽表（已预处理）
- `imputed_prot_ann.csv`：Mapping 到 HPA 的蛋白注释
- `bootstrap.sh`：一键环境与流水线脚本生成
- `Data_Exploration.ipynb`：早期探索性分析
- `NOTE.md`：蛋白冠背景与蛋白清单


### 环境要求
- Python 3.9+（推荐 3.10）
- 可选 GPU（CUDA）
- 结构嵌入需 `best.pkl`（ProDESIGN-LE 预训练权重）

安装依赖（推荐使用虚拟环境）：

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r ProDESIGN-LE/requirements_extra.txt
```

`requirements_extra.txt` 已包含：torch、fair-esm、biopython、pandas、numpy、pyarrow、tqdm、scikit-learn、openpyxl、requests、einops。


### 快速开始
若你已在仓库根目录：

```bash
# 1) 生成蛋白索引（从 imputed.xlsx 提取 UniProt 并与注释表合并）
python ProDESIGN-LE/pc_embed_pipeline.py index \
  --imputed imputed.xlsx \
  --ann imputed_prot_ann.csv \
  --out_dir ProDESIGN-LE/out

# 2) 抓取序列（优先从本地 FASTA，其次 UniProt REST）
python ProDESIGN-LE/pc_embed_pipeline.py fetch-seq \
  --out_dir ProDESIGN-LE/out

# 3) 计算序列嵌入（ESM2 t33 650M）
python ProDESIGN-LE/pc_embed_pipeline.py seq-embed \
  --out_dir ProDESIGN-LE/out

# 4) 计算结构嵌入（需 best.pkl；自动下载 AlphaFold v4 PDB）
python ProDESIGN-LE/pc_embed_pipeline.py str-embed \
  --pdb_dir ProDESIGN-LE/pdbs \
  --out_dir ProDESIGN-LE/out

# 5) 合并嵌入与注释（标准化数值列）
python ProDESIGN-LE/pc_embed_pipeline.py merge \
  --out_dir ProDESIGN-LE/out \
  --ann imputed_prot_ann.csv

# 6) 构建长表（宽转长，拼接嵌入）
python ProDESIGN-LE/pc_embed_pipeline.py build-long \
  --imputed imputed.xlsx \
  --out_dir ProDESIGN-LE/out

# 7) 快速评估（占位计算，便于后续替换为真实评估）
python ProDESIGN-LE/pc_embed_pipeline.py quick-eval \
  --out_dir ProDESIGN-LE/out

# 8) （新增）从既有产物生成统计 JSON（无需重算嵌入）
python ProDESIGN-LE/pc_embed_pipeline.py gen-stats \
  --pdb_dir ProDESIGN-LE/pdbs \
  --out_dir ProDESIGN-LE/out
```

一键执行（可选跳过结构嵌入）：

```bash
# 全流程
python ProDESIGN-LE/pc_embed_pipeline.py all \
  --imputed imputed.xlsx \
  --ann imputed_prot_ann.csv \
  --pdb_dir ProDESIGN-LE/pdbs \
  --out_dir ProDESIGN-LE/out

# 无 best.pkl 时跳过结构嵌入
python ProDESIGN-LE/pc_embed_pipeline.py all \
  --imputed imputed.xlsx \
  --ann imputed_prot_ann.csv \
  --skip-str-embed \
  --out_dir ProDESIGN-LE/out
```


### 主要输出
- `out/protein_index.csv`：索引与注释
- `out/sequences.fasta`：FASTA 序列
- `out/emb_seq_esm2_t33.csv`：ESM2 序列嵌入（1280 维）
- `out/emb_struc_le100.csv`：LE100（100 维）
- `out/emb_struc_le21.csv`：LE21（21 维）
- `out/protein_embed_merged.parquet`：合并后的特征表
- `out/long_task_table.parquet`：长表
- `out/stats_seq.json`、`out/stats_le.json`：从既有产物汇总的统计（gen-stats 生成）

示例统计（当前仓库数据）：
- `stats_seq.json`：`total_proteins=179`、`embedded_proteins=179`、`embedding_dim=1280`、`model=esm2_t33_650M_UR50D`，`sequence_length_stats`（mean≈552.03，median=444，min=83，max=4563）。
- `stats_le.json`：`total_proteins=179`、`le100_proteins=179`、`le21_proteins=179`、`downloaded_pdbs=175`、`missing_pdbs=["P08519","P04114","P49908","P22352"]`。

说明：上述 4 个 UniProt 号在 AlphaFold 页面/文件接口均 404，不提供模型，无法下载与计算 LE。


### 常见问题（FAQ）
- 找不到 `best.pkl`
  - 仅影响结构嵌入。可先加 `--skip-str-embed` 跑通其它步骤与 gen-stats。
- `einops` 或 `torch` 报错
  - 已在 `requirements_extra.txt` 中列出，确保虚拟环境激活后安装。
- CUDA OOM
  - 序列嵌入自动降批次；长序列采用滑窗计算 CLS 表示。
- AlphaFold PDB 下载失败
  - 网络/速率限制或该条目无公开模型；最终缺失项会记录到 `stats_le.json` 与合并表 `le_missing`。
- 临时 PDB 主链文件
  - `get_LE.py` 会在 `Temp/` 下写入主链 PDB（若目录不存在请手动创建）。


### 再现性建议
- 固定 Python/依赖版本并使用虚拟环境。
- 大规模计算建议分步执行并缓存中间产物（index、fetch-seq、seq-embed、str-embed）。


### 引用
- ESM2: Rao et al., Facebook AI Research（fair-esm）
- ProDESIGN-LE: Accurate and efficient protein sequence design through learning concise local environment of residues（Bioinformatics, 2023）
- 蛋白注释：The Human Protein Atlas
- 结构来源：AlphaFold Protein Structure Database


### 许可证
若未声明，默认以本仓库随附的 LICENSE 为准；如未包含 LICENSE，请在对外发布前补充适用的开源协议。


