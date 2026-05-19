from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from emulator_bench.feature_pipeline import load_molecule, load_protein


class CachedFeatureStore:
    def __init__(self, cache_dir: Path, max_items: int = 8192):
        self.cache_dir = Path(cache_dir)
        self.max_items = int(max_items)

    @lru_cache(maxsize=8192)
    def molecule(self, smiles: str) -> Dict[str, np.ndarray]:
        return load_molecule(self.cache_dir, smiles)

    @lru_cache(maxsize=8192)
    def protein(self, sequence: str) -> Dict[str, np.ndarray]:
        return load_protein(self.cache_dir, sequence)


class DLKcatCachedDataset(Dataset):
    def __init__(self, manifest_path: Path, cache_dir: Path, cache_items: int = 8192):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {self.manifest_path}. Run cache_embeddings.py first.")
        self.frame = pd.read_csv(self.manifest_path)
        self.store = CachedFeatureStore(cache_dir, max_items=cache_items)
        required = {"smiles", "sequence", "target_log2", "target_log10"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {self.manifest_path}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        smiles = str(row["smiles"])
        sequence = str(row["sequence"])
        molecule = self.store.molecule(smiles)
        protein = self.store.protein(sequence)
        return {
            "fingerprints": torch.from_numpy(molecule["fingerprints"].astype(np.int64, copy=False)),
            "adjacency": torch.from_numpy(molecule["adjacency"].astype(np.float32, copy=False)),
            "words": torch.from_numpy(protein["words"].astype(np.int64, copy=False)),
            "target_log2": torch.tensor(float(row["target_log2"]), dtype=torch.float32),
            "target_log10": torch.tensor(float(row["target_log10"]), dtype=torch.float32),
            "smiles": smiles,
            "sequence_key": str(row["sequence_key"]) if "sequence_key" in row else "",
        }


def dlkcat_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_atoms = max(int(item["fingerprints"].numel()) for item in batch)
    max_words = max(int(item["words"].numel()) for item in batch)

    fingerprints = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    adjacency = torch.zeros((batch_size, max_atoms, max_atoms), dtype=torch.float32)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.float32)
    words = torch.zeros((batch_size, max_words), dtype=torch.long)
    word_mask = torch.zeros((batch_size, max_words), dtype=torch.float32)
    target_log2 = torch.zeros((batch_size,), dtype=torch.float32)
    target_log10 = torch.zeros((batch_size,), dtype=torch.float32)

    smiles = []
    sequence_keys = []
    for idx, item in enumerate(batch):
        n_atoms = int(item["fingerprints"].numel())
        n_words = int(item["words"].numel())
        fingerprints[idx, :n_atoms] = item["fingerprints"]
        adjacency[idx, :n_atoms, :n_atoms] = item["adjacency"]
        atom_mask[idx, :n_atoms] = 1.0
        words[idx, :n_words] = item["words"]
        word_mask[idx, :n_words] = 1.0
        target_log2[idx] = item["target_log2"]
        target_log10[idx] = item["target_log10"]
        smiles.append(item["smiles"])
        sequence_keys.append(item["sequence_key"])

    return {
        "fingerprints": fingerprints,
        "adjacency": adjacency,
        "atom_mask": atom_mask,
        "words": words,
        "word_mask": word_mask,
        "target_log2": target_log2,
        "target_log10": target_log10,
        "smiles": smiles,
        "sequence_key": sequence_keys,
    }
