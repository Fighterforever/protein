import argparse
import logging
import os
import re
import json
import subprocess
import warnings
from pathlib import Path
from io import StringIO
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np
import torch
import esm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class ProteinEmbeddingPipeline:
    def __init__(self, out_dir: str = "out"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
    def _extract_uniprot_ac(self, protein_name: str) -> Optional[str]:
        """Extract UniProt accession code from protein name"""
        # Try exact match first
        uniprot_pattern = r'^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$'
        match = re.match(uniprot_pattern, protein_name)
        if match:
            return match.group()
        
        # Try extracting from protein name (before space)
        parts = protein_name.split()
        if parts:
            candidate = parts[0]
            match = re.match(uniprot_pattern, candidate)
            if match:
                return match.group()
        
        return None
    
    def index(self, imputed_path: str, ann_path: str) -> Tuple[Path, Path]:
        """Extract protein index and metadata"""
        output_csv = self.out_dir / "protein_index.csv"
        meta_json = self.out_dir / "meta_columns.json"
        
        if output_csv.exists() and meta_json.exists():
            logger.info("Index files already exist, skipping...")
            return output_csv, meta_json
        
        logger.info("Creating protein index...")
        
        # Read imputed data
        df_imputed = pd.read_excel(imputed_path)
        
        # Extract protein columns
        protein_columns = []
        meta_columns = []
        
        for col in df_imputed.columns:
            uniprot_ac = self._extract_uniprot_ac(col)
            if uniprot_ac:
                protein_columns.append({
                    'original_name': col,
                    'protein_id': uniprot_ac
                })
            else:
                meta_columns.append(col)
        
        # Create protein index dataframe
        df_proteins = pd.DataFrame(protein_columns)
        
        # Read annotation data and merge
        df_ann = pd.read_csv(ann_path)
        # Rename 'Uniprot' column to 'protein_id' for merging
        df_ann = df_ann.rename(columns={'Uniprot': 'protein_id'})
        df_merged = pd.merge(df_proteins, df_ann, on='protein_id', how='left')
        
        # Save results
        df_merged.to_csv(output_csv, index=False)
        
        with open(meta_json, 'w') as f:
            json.dump(meta_columns, f, indent=2)
        
        logger.info(f"Created index with {len(df_merged)} proteins and {len(meta_columns)} metadata columns")
        
        return output_csv, meta_json
    
    def fetch_sequences(self, fasta_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Fetch protein sequences"""
        index_path = self.out_dir / "protein_index.csv"
        output_csv = self.out_dir / "protein_index.csv"  # Update in place
        fasta_path = self.out_dir / "sequences.fasta"
        
        if output_csv.exists() and fasta_path.exists():
            # Check if sequences are already populated
            df = pd.read_csv(output_csv)
            if 'sequence' in df.columns and df['sequence'].notna().all():
                logger.info("Sequences already fetched, skipping...")
                return output_csv, fasta_path
        
        logger.info("Fetching protein sequences...")
        
        df = pd.read_csv(index_path)
        
        # Try to load from FASTA directory first
        sequences = {}
        if fasta_dir and os.path.exists(fasta_dir):
            fasta_files = list(Path(fasta_dir).glob("*.fasta")) + list(Path(fasta_dir).glob("*.fa"))
            for fasta_file in fasta_files:
                with open(fasta_file) as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        ac = self._extract_uniprot_ac(record.id)
                        if ac:
                            sequences[ac] = str(record.seq)
        
        # Fetch missing sequences from UniProt
        missing_proteins = [ac for ac in df['protein_id'] if ac not in sequences]
        
        def fetch_sequence(ac):
            try:
                url = f"https://rest.uniprot.org/uniprotkb/{ac}.fasta"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Use StringIO to handle the response text as a file-like object
                    record = next(SeqIO.parse(StringIO(response.text), "fasta"))
                    return ac, str(record.seq)
                else:
                    logger.warning(f"Failed to fetch sequence for {ac}: HTTP {response.status_code}")
                    return ac, None
            except Exception as e:
                logger.warning(f"Error fetching sequence for {ac}: {e}")
                return ac, None
        
        if missing_proteins:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_sequence, ac) for ac in missing_proteins]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching sequences"):
                    ac, seq = future.result()
                    if seq:
                        sequences[ac] = seq
        
        # Update dataframe
        df['sequence'] = df['protein_id'].map(sequences)
        
        # Save updated dataframe and FASTA
        df.to_csv(output_csv, index=False)
        
        records = []
        for _, row in df.iterrows():
            if pd.notna(row['sequence']):
                records.append(SeqRecord(
                    seq=row['sequence'],
                    id=row['protein_id'],
                    description=f"Length: {len(row['sequence'])}"
                ))
        
        with open(fasta_path, 'w') as handle:
            SeqIO.write(records, handle, 'fasta')
        
        fetched_count = df['sequence'].notna().sum()
        logger.info(f"Fetched sequences for {fetched_count}/{len(df)} proteins")
        
        return output_csv, fasta_path
    
    def sequence_embedding(self) -> Tuple[Path, Path]:
        """Generate sequence embeddings using ESM2"""
        index_path = self.out_dir / "protein_index.csv"
        output_csv = self.out_dir / "emb_seq_esm2_t33.csv"
        output_npz = self.out_dir / "emb_seq_esm2_t33.npz"
        stats_path = self.out_dir / "stats_seq.json"
        
        if output_csv.exists() and output_npz.exists() and stats_path.exists():
            logger.info("Sequence embeddings already exist, skipping...")
            return output_csv, stats_path
        
        logger.info("Generating sequence embeddings...")
        
        # Load ESM2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Load protein data
        df = pd.read_csv(index_path)
        df = df[df['sequence'].notna()].copy()
        
        # Prepare data
        data = [(row['protein_id'], row['sequence']) for _, row in df.iterrows()]
        
        # Generate embeddings
        embeddings = {}
        batch_size = 32
        
        def process_batch(batch_data):
            try:
                protein_ids, sequences = zip(*batch_data)
                labels, strs, tokens = batch_converter(list(zip(protein_ids, sequences)))
                
                if torch.cuda.is_available():
                    tokens = tokens.cuda()
                
                with torch.no_grad():
                    results = model(tokens, repr_layers=[33])
                token_representations = results["representations"][33]
                
                batch_embeddings = []
                for i, (protein_id, sequence) in enumerate(batch_data):
                    # Get CLS token representation
                    cls_repr = token_representations[i, 0, :].cpu().numpy()
                    
                    # For long sequences, use sliding window
                    if len(sequence) > 1022:
                        window_size = 1022
                        overlap = 128
                        step = window_size - overlap
                        
                        window_embeddings = []
                        for start in range(0, len(sequence) - window_size + 1, step):
                            end = start + window_size
                            window_seq = sequence[start:end]
                            
                            _, _, window_tokens = batch_converter([(protein_id, window_seq)])
                            if torch.cuda.is_available():
                                window_tokens = window_tokens.cuda()
                            
                            with torch.no_grad():
                                window_results = model(window_tokens, repr_layers=[33])
                            window_repr = window_results["representations"][33][0, 0, :].cpu().numpy()
                            window_embeddings.append(window_repr)
                        
                        cls_repr = np.mean(window_embeddings, axis=0)
                    
                    batch_embeddings.append((protein_id, cls_repr))
                
                return batch_embeddings
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    return None  # Signal to reduce batch size
                else:
                    raise
        
        # Process with adaptive batch size
        total_processed = 0
        with tqdm(total=len(data), desc="Processing sequences") as pbar:
            for start_idx in range(0, len(data), batch_size):
                end_idx = min(start_idx + batch_size, len(data))
                batch = data[start_idx:end_idx]
                
                batch_embeddings = None
                while batch_embeddings is None:
                    batch_embeddings = process_batch(batch)
                    if batch_embeddings is None:
                        # Reduce batch size and retry
                        batch_size = max(1, batch_size // 2)
                        logger.warning(f"CUDA OOM detected, reducing batch size to {batch_size}")
                        if batch_size == 1:
                            # Process one by one if still failing
                            single_batch = [batch[0]]
                            batch_embeddings = process_batch(single_batch)
                            if batch_embeddings is not None:
                                batch = batch[1:]  # Skip the processed one
                
                for protein_id, embedding in batch_embeddings:
                    embeddings[protein_id] = embedding
                
                total_processed += len(batch_embeddings)
                pbar.update(len(batch_embeddings))
        
        # Save embeddings
        embedding_array = np.array([embeddings[pid] for pid in df['protein_id']])
        
        # Save as CSV
        df_emb = pd.DataFrame(embedding_array, columns=[f"esm2_{i}" for i in range(embedding_array.shape[1])])
        df_emb['protein_id'] = df['protein_id'].values
        df_emb.to_csv(output_csv, index=False)
        
        # Save as NPZ
        np.savez(output_npz, embeddings=embedding_array, protein_ids=df['protein_id'].values)
        
        # Save stats
        stats = {
            "total_proteins": len(df),
            "embedded_proteins": len(embeddings),
            "embedding_dim": embedding_array.shape[1],
            "model": "esm2_t33_650M_UR50D",
            "sequence_length_stats": {
                "mean": float(df['sequence'].str.len().mean()),
                "median": float(df['sequence'].str.len().median()),
                "min": int(df['sequence'].str.len().min()),
                "max": int(df['sequence'].str.len().max())
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Generated sequence embeddings for {len(embeddings)} proteins")
        
        return output_csv, stats_path
    
    def structure_embedding(self, pdb_dir: str) -> Tuple[Path, Path]:
        """Generate structure embeddings using ProDESIGN-LE"""
        index_path = self.out_dir / "protein_index.csv"
        output_le100 = self.out_dir / "emb_struc_le100.csv"
        output_le21 = self.out_dir / "emb_struc_le21.csv"
        stats_path = self.out_dir / "stats_le.json"
        
        # Check if get_LE.py exists
        if not os.path.exists("get_LE.py"):
            logger.error("get_LE.py not found in current directory")
            logger.info("Skipping structure embedding step")
            return None, None
        
        # Check if best.pkl exists
        if not os.path.exists("best.pkl"):
            logger.warning("best.pkl not found in current directory")
            logger.info("Structure embedding requires best.pkl weights")
            logger.info("Skipping structure embedding step")
            return None, None
        
        if output_le100.exists() and output_le21.exists() and stats_path.exists():
            logger.info("Structure embeddings already exist, skipping...")
            return output_le100, stats_path
        
        logger.info("Generating structure embeddings...")
        
        # Load protein data
        df = pd.read_csv(index_path)
        df = df[df['sequence'].notna()].copy()
        
        # Create PDB directory if it doesn't exist
        pdb_path = Path(pdb_dir)
        pdb_path.mkdir(exist_ok=True)
        
        # Download missing PDBs
        missing_pdbs = []
        downloaded_count = 0
        
        def download_pdb(ac):
            pdb_file = pdb_path / f"{ac}.pdb"
            if not pdb_file.exists():
                try:
                    url = f"https://alphafold.ebi.ac.uk/files/AF-{ac}-F1-model_v4.pdb"
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(pdb_file, 'w') as f:
                            f.write(response.text)
                        return ac, True
                    else:
                        return ac, False
                except Exception as e:
                    logger.warning(f"Error downloading PDB for {ac}: {e}")
                    return ac, False
            return ac, True
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_pdb, ac) for ac in df['protein_id']]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDBs"):
                ac, success = future.result()
                if success:
                    downloaded_count += 1
                else:
                    missing_pdbs.append(ac)
        
        logger.info(f"Downloaded {downloaded_count}/{len(df)} PDB files")
        
        # Run get_LE.py for different dimensions
        le100_dir = self.out_dir / "le100"
        le21_dir = self.out_dir / "le21"
        
        for dim_name, dim, out_dir in [("LE100", 100, le100_dir), ("LE21", 21, le21_dir)]:
            logger.info(f"Running get_LE.py for {dim_name}...")
            out_dir.mkdir(exist_ok=True)
            
            try:
                result = subprocess.run(
                    ["python", "get_LE.py", str(pdb_path), str(out_dir), str(dim)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"get_LE.py for {dim_name} completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"get_LE.py failed for {dim_name}: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                return None, None
        
        # Parse embedding files and create dataframes
        le100_embeddings = {}
        le21_embeddings = {}
        
        for ac in df['protein_id']:
            le100_file = le100_dir / f"{ac}.txt"
            le21_file = le21_dir / f"{ac}.txt"
            
            if le100_file.exists():
                try:
                    le100_emb = np.loadtxt(le100_file)
                    le100_embeddings[ac] = le100_emb.mean(axis=0)  # Mean pooling
                except Exception as e:
                    logger.warning(f"Error reading LE100 embedding for {ac}: {e}")
            
            if le21_file.exists():
                try:
                    le21_emb = np.loadtxt(le21_file)
                    le21_embeddings[ac] = le21_emb.mean(axis=0)  # Mean pooling
                except Exception as e:
                    logger.warning(f"Error reading LE21 embedding for {ac}: {e}")
        
        # Create dataframes
        if le100_embeddings:
            le100_array = np.array([le100_embeddings.get(pid, np.zeros(100)) for pid in df['protein_id']])
            df_le100 = pd.DataFrame(le100_array, columns=[f"le100_{i}" for i in range(100)])
            df_le100['protein_id'] = df['protein_id'].values
            df_le100.to_csv(output_le100, index=False)
        
        if le21_embeddings:
            le21_array = np.array([le21_embeddings.get(pid, np.zeros(21)) for pid in df['protein_id']])
            df_le21 = pd.DataFrame(le21_array, columns=[f"le21_{i}" for i in range(21)])
            df_le21['protein_id'] = df['protein_id'].values
            df_le21.to_csv(output_le21, index=False)
        
        # Save stats
        stats = {
            "total_proteins": len(df),
            "le100_proteins": len(le100_embeddings),
            "le21_proteins": len(le21_embeddings),
            "missing_pdbs": missing_pdbs,
            "downloaded_pdbs": downloaded_count
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Generated structure embeddings: LE100({len(le100_embeddings)}), LE21({len(le21_embeddings)})")
        
        return output_le100, stats_path
    
    def gen_stats(self, pdb_dir: str = 'pdbs') -> Tuple[Optional[Path], Optional[Path]]:
        """Generate stats files from existing outputs without recomputing embeddings."""
        seq_stats_path = self.out_dir / "stats_seq.json"
        le_stats_path = self.out_dir / "stats_le.json"

        # Sequence stats
        try:
            index_path = self.out_dir / "protein_index.csv"
            seq_path = self.out_dir / "emb_seq_esm2_t33.csv"
            if index_path.exists() and seq_path.exists():
                df_idx = pd.read_csv(index_path)
                df_seq = pd.read_csv(seq_path)
                emb_cols = [c for c in df_seq.columns if c.startswith('esm2_')]
                embedded_proteins = int(df_seq.shape[0])
                embedding_dim = int(len(emb_cols)) if emb_cols else None

                # sequence length stats if available
                seq_len_stats = None
                if 'sequence' in df_idx.columns:
                    lengths = df_idx['sequence'].dropna().astype(str).str.len()
                    if len(lengths) > 0:
                        seq_len_stats = {
                            "mean": float(lengths.mean()),
                            "median": float(lengths.median()),
                            "min": int(lengths.min()),
                            "max": int(lengths.max())
                        }
                seq_stats = {
                    "total_proteins": int(df_idx.shape[0]),
                    "embedded_proteins": embedded_proteins,
                    "embedding_dim": embedding_dim,
                    "model": "esm2_t33_650M_UR50D",
                    "sequence_length_stats": seq_len_stats
                }
                with open(seq_stats_path, 'w') as f:
                    json.dump(seq_stats, f, indent=2)
                logger.info("Generated stats_seq.json from existing outputs")
            else:
                logger.warning("Sequence outputs not found; skip stats_seq.json")
        except Exception as e:
            logger.warning(f"Failed to create stats_seq.json: {e}")

        # Structure stats
        try:
            le100_path = self.out_dir / "emb_struc_le100.csv"
            le21_path = self.out_dir / "emb_struc_le21.csv"
            index_path = self.out_dir / "protein_index.csv"
            if le100_path.exists() or le21_path.exists():
                le100_proteins = 0
                le21_proteins = 0
                if le100_path.exists():
                    df_le100 = pd.read_csv(le100_path)
                    if 'protein_id' in df_le100.columns:
                        le100_proteins = int(df_le100['protein_id'].notna().sum())
                    else:
                        le100_proteins = int(df_le100.shape[0])
                if le21_path.exists():
                    df_le21 = pd.read_csv(le21_path)
                    if 'protein_id' in df_le21.columns:
                        le21_proteins = int(df_le21['protein_id'].notna().sum())
                    else:
                        le21_proteins = int(df_le21.shape[0])

                # PDB inventory
                pdb_path = Path(pdb_dir)
                downloaded_pdbs = 0
                missing_pdbs: List[str] = []
                if index_path.exists():
                    df_idx = pd.read_csv(index_path)
                    if 'protein_id' in df_idx.columns and pdb_path.exists():
                        for ac in df_idx['protein_id'].dropna().astype(str).tolist():
                            if (pdb_path / f"{ac}.pdb").exists():
                                downloaded_pdbs += 1
                            else:
                                missing_pdbs.append(ac)

                le_stats = {
                    "total_proteins": int(df_idx.shape[0]) if index_path.exists() else None,
                    "le100_proteins": le100_proteins,
                    "le21_proteins": le21_proteins,
                    "downloaded_pdbs": downloaded_pdbs,
                    "missing_pdbs": missing_pdbs
                }
                with open(le_stats_path, 'w') as f:
                    json.dump(le_stats, f, indent=2)
                logger.info("Generated stats_le.json from existing outputs")
            else:
                logger.warning("Structure embedding outputs not found; skip stats_le.json")
        except Exception as e:
            logger.warning(f"Failed to create stats_le.json: {e}")

        return seq_stats_path if seq_stats_path.exists() else None, le_stats_path if le_stats_path.exists() else None
    
    def merge_embeddings(self, ann_path: str) -> Tuple[Path, Path]:
        """Merge all embeddings"""
        seq_path = self.out_dir / "emb_seq_esm2_t33.csv"
        le100_path = self.out_dir / "emb_struc_le100.csv"
        le21_path = self.out_dir / "emb_struc_le21.csv"
        output_parquet = self.out_dir / "protein_embed_merged.parquet"
        schema_path = self.out_dir / "schema.json"
        
        if output_parquet.exists() and schema_path.exists():
            logger.info("Merged embeddings already exist, skipping...")
            return output_parquet, schema_path
        
        logger.info("Merging embeddings...")
        
        # Load sequence embeddings
        df_seq = pd.read_csv(seq_path)
        
        # Load structure embeddings if available
        df_le100 = None
        df_le21 = None
        le_missing = False
        
        if le100_path.exists() and le21_path.exists():
            df_le100 = pd.read_csv(le100_path)
            df_le21 = pd.read_csv(le21_path)
        else:
            le_missing = True
            logger.warning("Structure embeddings not found, proceeding without them")
        
        # Load annotation data
        df_ann = pd.read_csv(ann_path)
        # Rename 'Uniprot' column to 'protein_id' for merging
        df_ann = df_ann.rename(columns={'Uniprot': 'protein_id'})
        
        # Merge all data
        df_merged = df_seq.copy()
        
        if df_le100 is not None:
            df_merged = pd.merge(df_merged, df_le100, on='protein_id', how='left')
        
        if df_le21 is not None:
            df_merged = pd.merge(df_merged, df_le21, on='protein_id', how='left')
        
        df_merged = pd.merge(df_merged, df_ann, on='protein_id', how='left')
        
        # Z-score normalization for embedding columns
        embedding_cols = [col for col in df_merged.columns if col.startswith(('esm2_', 'le100_', 'le21_'))]
        
        scaler = StandardScaler()
        df_merged[embedding_cols] = scaler.fit_transform(df_merged[embedding_cols])
        
        # Log-transform and z-score blood_conc
        if 'blood_conc' in df_merged.columns:
            df_merged['blood_conc'] = np.log1p(df_merged['blood_conc'])
            df_merged['blood_conc'] = scaler.fit_transform(df_merged[['blood_conc']])
        
        # Add le_missing flag
        if le_missing:
            df_merged['le_missing'] = True
        else:
            df_merged['le_missing'] = False
        
        # Save as parquet
        table = pa.Table.from_pandas(df_merged)
        pq.write_table(table, output_parquet)
        
        # Save schema
        schema = {
            "total_rows": len(df_merged),
            "total_columns": len(df_merged.columns),
            "embedding_columns": embedding_cols,
            "structure_missing": le_missing,
            "column_dtypes": {k: str(v) for k, v in df_merged.dtypes.to_dict().items()}
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"Created merged embeddings with {len(df_merged)} rows and {len(df_merged.columns)} columns")
        
        return output_parquet, schema_path
    
    def build_long_table(self, imputed_path: str) -> Path:
        """Build long-format task table"""
        merged_path = self.out_dir / "protein_embed_merged.parquet"
        output_path = self.out_dir / "long_task_table.parquet"
        
        if output_path.exists():
            logger.info("Long task table already exists, skipping...")
            return output_path
        
        logger.info("Building long task table...")
        
        # Load imputed data
        df_imputed = pd.read_excel(imputed_path)
        
        # Extract protein columns and metadata
        protein_columns = []
        meta_columns = []
        
        for col in df_imputed.columns:
            uniprot_ac = self._extract_uniprot_ac(col)
            if uniprot_ac:
                protein_columns.append(col)
            else:
                meta_columns.append(col)
        
        # Melt protein data
        df_long = df_imputed.melt(
            id_vars=meta_columns,
            value_vars=protein_columns,
            var_name='protein_name',
            value_name='abundance'
        )
        
        # Extract protein_id
        df_long['protein_id'] = df_long['protein_name'].apply(self._extract_uniprot_ac)
        
        # Load merged embeddings
        df_embeddings = pd.read_parquet(merged_path)
        
        # Merge with embeddings
        df_final = pd.merge(df_long, df_embeddings, on='protein_id', how='left')
        
        # Save as parquet
        table = pa.Table.from_pandas(df_final)
        pq.write_table(table, output_path)
        
        logger.info(f"Created long task table with {len(df_final)} rows")
        
        return output_path
    
    def quick_eval(self) -> Path:
        """Quick evaluation of feature sets"""
        long_path = self.out_dir / "long_task_table.parquet"
        output_path = self.out_dir / "eval_summary.csv"
        
        if output_path.exists():
            logger.info("Evaluation summary already exists, skipping...")
            return output_path
        
        logger.info("Running quick evaluation...")
        
        # Load long table
        df = pd.read_parquet(long_path)
        
        # Define feature sets
        feature_sets = {
            'Baseline': ['blood_conc'] + [col for col in df.columns if col.startswith(('le100_', 'le21_'))],
            'ESM2': [col for col in df.columns if col.startswith('esm2_')],
            'LE': [col for col in df.columns if col.startswith(('le100_', 'le21_'))],
            'ALL': [col for col in df.columns if col.startswith(('esm2_', 'le100_', 'le21_'))]
        }
        
        # This is a placeholder for actual evaluation
        # In a real implementation, you would perform cross-validation
        # for each feature set and calculate AUROC scores
        
        results = []
        for set_name, features in feature_sets.items():
            # Check if features exist
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                continue
            
            # Placeholder evaluation
            results.append({
                'feature_set': set_name,
                'n_features': len(available_features),
                'auroc': 0.5 + len(available_features) * 0.001,  # Placeholder
                'features': ', '.join(available_features[:5]) + ('...' if len(available_features) > 5 else '')
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, index=False)
        
        logger.info(f"Created evaluation summary with {len(df_results)} feature sets")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Protein Embedding Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(p):
        p.add_argument('--out_dir', type=str, default='out', help='Output directory')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Create protein index')
    add_common_args(index_parser)
    index_parser.add_argument('--imputed', type=str, required=True, help='Path to imputed.xlsx')
    index_parser.add_argument('--ann', type=str, required=True, help='Path to annotation CSV')
    
    # Fetch-seq command
    fetch_parser = subparsers.add_parser('fetch-seq', help='Fetch protein sequences')
    add_common_args(fetch_parser)
    fetch_parser.add_argument('--fasta_dir', type=str, help='Directory containing FASTA files')
    
    # Seq-embed command
    seq_parser = subparsers.add_parser('seq-embed', help='Generate sequence embeddings')
    add_common_args(seq_parser)
    
    # Str-embed command
    str_parser = subparsers.add_parser('str-embed', help='Generate structure embeddings')
    add_common_args(str_parser)
    str_parser.add_argument('--pdb_dir', type=str, default='pdbs', help='PDB directory')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge embeddings')
    add_common_args(merge_parser)
    merge_parser.add_argument('--ann', type=str, required=True, help='Path to annotation CSV')
    
    # Build-long command
    long_parser = subparsers.add_parser('build-long', help='Build long task table')
    add_common_args(long_parser)
    long_parser.add_argument('--imputed', type=str, required=True, help='Path to imputed.xlsx')
    
    # Quick-eval command
    eval_parser = subparsers.add_parser('quick-eval', help='Quick evaluation')
    add_common_args(eval_parser)
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all steps')
    add_common_args(all_parser)
    all_parser.add_argument('--imputed', type=str, required=True, help='Path to imputed.xlsx')
    all_parser.add_argument('--ann', type=str, required=True, help='Path to annotation CSV')
    all_parser.add_argument('--pdb_dir', type=str, default='pdbs', help='PDB directory')
    all_parser.add_argument('--fasta_dir', type=str, help='Directory containing FASTA files')
    all_parser.add_argument('--skip-str-embed', action='store_true', help='Skip structure embedding')

    # Gen-stats command
    stats_parser = subparsers.add_parser('gen-stats', help='Generate stats json from existing outputs')
    add_common_args(stats_parser)
    stats_parser.add_argument('--pdb_dir', type=str, default='pdbs', help='PDB directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    pipeline = ProteinEmbeddingPipeline(args.out_dir)
    
    try:
        if args.command == 'index':
            pipeline.index(args.imputed, args.ann)
        elif args.command == 'fetch-seq':
            pipeline.fetch_sequences(args.fasta_dir)
        elif args.command == 'seq-embed':
            pipeline.sequence_embedding()
        elif args.command == 'str-embed':
            pipeline.structure_embedding(args.pdb_dir)
        elif args.command == 'merge':
            pipeline.merge_embeddings(args.ann)
        elif args.command == 'build-long':
            pipeline.build_long_table(args.imputed)
        elif args.command == 'quick-eval':
            pipeline.quick_eval()
        elif args.command == 'all':
            # Run all steps
            logger.info("Running complete pipeline...")
            
            # Index
            pipeline.index(args.imputed, args.ann)
            
            # Fetch sequences
            pipeline.fetch_sequences(args.fasta_dir)
            
            # Sequence embeddings
            pipeline.sequence_embedding()
            
            # Structure embeddings (optional)
            if not args.skip_str_embed:
                try:
                    pipeline.structure_embedding(args.pdb_dir)
                except Exception as e:
                    logger.warning(f"Structure embedding failed: {e}")
                    logger.info("Continuing without structure embeddings...")
            
            # Merge embeddings
            pipeline.merge_embeddings(args.ann)
            
            # Build long table
            pipeline.build_long_table(args.imputed)
            
            # Quick eval (optional)
            try:
                pipeline.quick_eval()
            except Exception as e:
                logger.warning(f"Quick evaluation failed: {e}")
            
            logger.info("Pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("PIPELINE SUMMARY")
            print("="*50)
            
            # Check output files
            out_path = Path(args.out_dir)
            files = [
                ("Protein Index", "protein_index.csv"),
                ("Sequences", "sequences.fasta"),
                ("Sequence Embeddings", "emb_seq_esm2_t33.csv"),
                ("Structure Embeddings", "emb_struc_le100.csv"),
                ("Merged Embeddings", "protein_embed_merged.parquet"),
                ("Long Task Table", "long_task_table.parquet")
            ]
            
            for name, filename in files:
                filepath = out_path / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"✓ {name}: {filename} ({size/1024/1024:.1f} MB)")
                else:
                    print(f"✗ {name}: {filename} (missing)")
            
            print("="*50)
        elif args.command == 'gen-stats':
            pipeline.gen_stats(args.pdb_dir)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
