# 蛋白序列与结构 Embedding 工作总结

## 1. 任务背景

根据小组分工，本人负责"蛋白序列、结构 embedding"部分。该任务的目标是将179个蛋白冠相关蛋白转换为机器学习可用的高维特征向量，为后续的随机森林预测模型提供输入特征。

蛋白质的结构特征对蛋白冠形成具有重要影响，包括：
- 疏水/亲水区域分布
- 电荷分布
- 柔性结构域的存在
- 二级结构特征（α螺旋、β折叠等）

通过深度学习模型生成的蛋白嵌入能够有效捕获这些结构-功能关系，为预测蛋白冠组成提供关键特征。

## 2. 技术方案

### 2.1 序列 Embedding：ESM2

**模型选择**：ESM2-t33-650M（Meta AI）
- ESM2 应该是目前性能最优的蛋白质语言模型之一
- 在 UniRef50 数据集上训练
- 33层Transformer，650M参数量

**技术细节**：
- 嵌入维度：1280维
- 处理策略：
  - 短序列（≤1022氨基酸）：直接嵌入
  - 长序列（>1022氨基酸）：滑动窗口平均
- GPU加速：自动检测CUDA并使用GPU
- 内存管理：自适应批处理，自动处理CUDA OOM

### 2.2 结构 Embedding：ProDESIGN-LE

**模型选择**：ProDESIGN-LE（Local Environment）
- 基于Transformer的蛋白质结构编码器
- 能够提取蛋白质局部环境特征
- 来源：https://github.com/bigict/ProDESIGN-LE

**技术细节**：
- 输入：蛋白质3D结构（PDB格式）
- 输出：
  - **LE100**：100维特征向量（高层抽象特征，logits层）
  - **LE21**：21维特征向量（低层局部环境特征，features层）
- 结构来源：AlphaFold v4 预测结构（自动下载）
- 模型权重：best.pkl（需预先准备）

### 2.3 自动化流水线

开发了端到端的自动化流水线 `pc_embed_pipeline.py`，包含以下功能模块：

1. **index**：从宽表提取UniProt ID并与注释数据合并
2. **fetch-seq**：从UniProt REST API批量获取FASTA序列
3. **seq-embed**：使用ESM2生成序列嵌入
4. **str-embed**：自动下载PDB并计算结构嵌入
5. **merge**：合并多源嵌入与元数据，标准化数值特征
6. **build-long**：构建长表格式（每行=一个样本）
7. **quick-eval**：快速评估，目前是占位
8. **gen-stats**：从现有输出生成统计摘要

## 3. 实施成果

### 3.1 数据覆盖情况

**序列嵌入完成情况**：
```json
{
  "total_proteins": 179,
  "embedded_proteins": 179,
  "embedding_dim": 1280,
  "model": "esm2_t33_650M_UR50D",
  "sequence_length_stats": {
    "mean": 552.03,
    "median": 444.0,
    "min": 83,
    "max": 4563
  }
}
```
 **100% 覆盖**

**结构嵌入完成情况**：
```json
{
  "total_proteins": 179,
  "le100_proteins": 179,
  "le21_proteins": 179,
  "downloaded_pdbs": 175,
  "missing_pdbs": [
    "P08519",  // Apolipoprotein(a)
    "P04114",  // Apolipoprotein B-100
    "P49908",  // Selenoprotein P
    "P22352"   // Glutathione peroxidase 3
  ]
}
```
 **97.8% 覆盖**（175/179，4个蛋白AlphaFold数据库无结构）

### 3.2 输出文件说明

所有输出文件位于 `ProDESIGN-LE/out/` 目录：

| 文件名 | 说明 | 格式 | 维度/行数 |
|--------|------|------|-----------|
| `protein_index.csv` | 蛋白元数据索引（含HPA注释） | CSV | 179行 × 88列 |
| `sequences.fasta` | 蛋白序列（FASTA格式） | FASTA | 179条序列 |
| `emb_seq_esm2_t33.csv` | ESM2序列嵌入 | CSV | 179行 × 1281列 |
| `emb_struc_le100.csv` | LE100结构嵌入 | CSV | 179行 × 101列 |
| `emb_struc_le21.csv` | LE21结构嵌入 | CSV | 179行 × 22列 |
| `protein_embed_merged.parquet` | 合并表（嵌入+元数据） | Parquet | 179行 × 1490列 |
| `long_task_table.parquet` | 长表（样本级） | Parquet | 待定 × N列 |
| `eval_summary.csv` | 评估摘要（占位） | CSV | 待定 |
| `stats_seq.json` | 序列嵌入统计 | JSON | - |
| `stats_le.json` | 结构嵌入统计 | JSON | - |

### 3.3 重点


1. 已完成的嵌入自动跳过，避免重复计算
4. 无需手动下载PDB，流水线自动从AlphaFold抓取
5. 数值特征自动标准化（Z-score

## 4. 使用方法

### 4.1 环境准备

```bash
# 克隆仓库
git clone https://github.com/Fighterforever/protein.git
cd protein

# 创建环境
python -m venv .venv
source .venv/bin/activate  # 我是macOS

# 安装依赖
pip install -r ProDESIGN-LE/requirements_extra.txt

# 准备权重文件（需提前获取）
# 将 best.pkl 放置在 ProDESIGN-LE/ 目录下
```

### 4.2 一键运行

```bash
# 完整流水线（从原始数据到合并表）
python ProDESIGN-LE/pc_embed_pipeline.py all \
  --imputed imputed.xlsx \
  --ann imputed_prot_ann.csv \
  --pdb_dir ProDESIGN-LE/pdbs \
  --out_dir ProDESIGN-LE/out

# 生成统计摘要（从已有输出）
python ProDESIGN-LE/pc_embed_pipeline.py gen-stats \
  --pdb_dir ProDESIGN-LE/pdbs \
  --out_dir ProDESIGN-LE/out
```

### 4.3 分步执行

如需细粒度控制，可分步执行各模块（详见 `README.md`）。

## 5. 下游对接说明

### 5.1 文件

**用于随机森林建模**：
- **主表**：`protein_embed_merged.parquet`
  - 包含179个蛋白的完整特征（嵌入 + 元数据）
  - 可直接用于蛋白级别的分析

- **样本表**：`long_task_table.parquet`
  - 宽转长格式，每行为一个实验样本
  - 包含纳米粒特征 + 蛋白特征 + 目标变量
  - 适合直接输入随机森林

### 5.2 特征

**序列嵌入**（1280维）：
- 列名格式：`esm2_0`, `esm2_1`, ..., `esm2_1279`
- 特点：高维、信息密集、已标准化
- 建议：可考虑PCA降维至50-100维

**结构嵌入 LE100**（100维）：
- 列名格式：`le100_0`, `le100_1`, ..., `le100_99`
- 特点：高层抽象特征，反映全局结构模式

**结构嵌入 LE21**（21维）：
- 列名格式：`le21_0`, `le21_1`, ..., `le21_20`
- 特点：局部环境特征，维度较低

**元数据特征**（88列）：
- 包含血液浓度、组织表达、亚细胞定位等生物学信息
- 已自动标准化数值列

### 5.3 缺失值处理

4个结构缺失的蛋白（P08519, P04114, P49908, P22352）：
- LE100/LE21嵌入已填充为**全0向量**
- 序列嵌入完整
- 建议：RF建模时可保留（树模型对缺失值鲁棒）或单独标记

## 6. 技术栈与依赖

**核心库**：
- PyTorch 2.0+（深度学习框架）
- fair-esm（Meta ESM2模型）
- Biopython（序列处理）
- pandas、numpy（数据处理）
- pyarrow（高效表格存储）
- scikit-learn（标准化）

**外部API**：
- UniProt REST API（序列获取）
- AlphaFold Protein Structure Database（结构下载）

## 7. 已知问题与限制

4个蛋白AlphaFold无预测结构（均为大型/特殊蛋白）


## 8. 项目仓库

**GitHub**: https://github.com/Fighterforever/protein

仓库包含：
- 完整代码与流水线脚本
- 输入数据（imputed.xlsx, imputed_prot_ann.csv）
- 输出示例（out/ 目录）
- 详细使用文档（README.md）

**最新发布**：v0.1.0
- Release页面：https://github.com/Fighterforever/protein/releases/tag/v0.1.0

## 9. 参考文献

1. **ESM2**: Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model[J]. Science, 2023, 379(6637): 1123-1130.

2. **ProDESIGN-LE**: Zheng Z, Deng Y, Xue D, et al. Structure-informed Language Models Are Protein Designers[C]//International Conference on Machine Learning. PMLR, 2023: 42400-42416.

3. **AlphaFold Database**: Varadi M, Anyango S, Deshpande M, et al. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models[J]. Nucleic acids research, 2022, 50(D1): D439-D444.

