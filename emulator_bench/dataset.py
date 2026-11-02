from pathlib import Path

import numpy as np
import pandas as pd
import torch

from emulator_bench.feature_pipeline import load_molecule, load_protein


class PreloadedSplit:
    """One split on the device. Rows with the same SMILES or sequence share one tensor.

    `samples[i]` is `(fingerprints, adjacency, words, target_log2)` for row i, `target_log10` holds the log10 targets.
    """

    def __init__(self, manifest_path: Path, cache_dir: Path, device: torch.device, store: dict):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}. Run cache_embeddings.py first.")
        frame = pd.read_csv(manifest_path, usecols=["smiles", "sequence", "target_log2", "target_log10"])
        molecules, proteins = store.setdefault("molecules", {}), store.setdefault("proteins", {})
        for smiles in frame["smiles"].unique():
            if smiles not in molecules:
                data = load_molecule(cache_dir, smiles)
                molecules[smiles] = (
                    torch.from_numpy(data["fingerprints"].astype(np.int64, copy=False)).to(device),
                    torch.from_numpy(data["adjacency"].astype(np.float32, copy=False)).to(device),
                )
        for sequence in frame["sequence"].unique():
            if sequence not in proteins:
                proteins[sequence] = torch.from_numpy(load_protein(cache_dir, sequence)["words"].astype(np.int64, copy=False)).to(device)
        targets = torch.tensor(frame["target_log2"].to_numpy(np.float32), device=device).view(-1, 1, 1)
        self.samples = [(*molecules[s], proteins[q], t) for s, q, t in zip(frame["smiles"], frame["sequence"], targets.unbind())]
        self.target_log10 = frame["target_log10"].to_numpy(np.float32)

    def __len__(self):
        return len(self.samples)
