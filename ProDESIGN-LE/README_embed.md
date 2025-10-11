# Protein Embedding Pipeline

This pipeline generates protein embeddings for structure-function analysis using ESM2 (sequence) and ProDESIGN-LE (structure) models.

## Prerequisites

- `git` and Python 3.9+
- CUDA (optional, for GPU acceleration)
- `best.pkl` weights file (for structure embedding)

## Quick Start

### One-click bootstrap

```bash
bash bootstrap.sh
```

### Manual installation

1. Clone repository:
```bash
git clone https://github.com/bigict/ProDESIGN-LE.git
cd ProDESIGN-LE
```

2. Create environment:
```bash
# Using conda (recommended)
conda create -n pcembed python=3.10
conda activate pcembed

# Or using venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # If exists
pip install -r requirements_extra.txt
```

4. Copy data files:
```bash
cp ../imputed.xlsx .
cp ../imputed_prot_ann.csv .
cp ../get_LE.py .
```

5. Obtain `best.pkl` and place in repository root.

## Usage

### Environment activation

```bash
# Conda
conda activate pcembed

# Venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Individual steps

```bash
# 1) Create protein index
python pc_embed_pipeline.py index --imputed imputed.xlsx --ann imputed_prot_ann.csv --out_dir out

# 2) Fetch protein sequences
python pc_embed_pipeline.py fetch-seq --out_dir out --fasta_dir fastas

# 3) Generate sequence embeddings
python pc_embed_pipeline.py seq-embed --out_dir out

# 4) Generate structure embeddings (requires best.pkl)
python pc_embed_pipeline.py str-embed --pdb_dir pdbs --out_dir out

# 5) Merge all embeddings
python pc_embed_pipeline.py merge --out_dir out --ann imputed_prot_ann.csv

# 6) Build long-format task table
python pc_embed_pipeline.py build-long --imputed imputed.xlsx --out_dir out

# 7) Quick evaluation (optional)
python pc_embed_pipeline.py quick-eval --out_dir out
```

### One-click pipeline

```bash
# Full pipeline with structure embedding
python pc_embed_pipeline.py all --imputed imputed.xlsx --ann imputed_prot_ann.csv --pdb_dir pdbs --out_dir out

# Skip structure embedding (if best.pkl is missing)
python pc_embed_pipeline.py all --imputed imputed.xlsx --ann imputed_prot_ann.csv --skip-str-embed --out_dir out

# (New) Generate stats from existing outputs
python pc_embed_pipeline.py gen-stats --pdb_dir pdbs --out_dir out
```

## Input Files

- `imputed.xlsx`: Protein abundance data (wide format)
- `imputed_prot_ann.csv`: Protein annotations
- `get_LE.py`: Structure embedding script (provided with ProDESIGN-LE)
- `best.pkl`: Pre-trained weights (required for structure embedding)

## Output Files

- `out/protein_index.csv`: Protein metadata index
- `out/sequences.fasta`: Protein sequences in FASTA format
- `out/emb_seq_esm2_t33.csv`: ESM2 sequence embeddings (1280D)
- `out/emb_struc_le100.csv`: LE100 structure embeddings (100D)
- `out/emb_struc_le21.csv`: LE21 structure embeddings (21D)
- `out/protein_embed_merged.parquet`: Combined embeddings
- `out/long_task_table.parquet`: Long-format task table

## Important Notes

1. **Protein IDs**: The pipeline uses UniProt accession codes as protein identifiers. These are extracted from column names in the input data.

2. **Structure Embedding**: 
   - Requires `best.pkl` weights file
   - Downloads PDB structures from AlphaFold v4
   - Uses position arguments when calling `get_LE.py`

3. **Memory Usage**:
   - Sequence embedding uses GPU if available
   - Adaptive batch size handles CUDA OOM errors
   - Long sequences (>1022 aa) use sliding window approach

4. **Error Handling**:
   - Pipeline continues even if structure embedding fails
   - Missing PDBs are logged but don't stop execution
   - Failed UniProt requests are logged and skipped

## Troubleshooting

1. **Missing `best.pkl`**: The pipeline will skip structure embedding but continue with sequence embeddings. Obtain the weights file from the ProDESIGN-LE project.

2. **CUDA OOM**: The pipeline automatically reduces batch size when GPU memory is insufficient.

3. **Network Issues**: The pipeline retries failed downloads and logs errors for manual inspection.

4. **Large Datasets**: For very large datasets, consider running individual steps separately and monitoring memory usage.
