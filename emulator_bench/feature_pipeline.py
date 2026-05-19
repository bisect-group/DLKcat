import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from emulator_bench.common import ensure_parent, normalize_sequence, stable_hash


def smiles_key(smiles: str) -> str:
    return stable_hash(str(smiles).strip())


def sequence_key(sequence: str) -> str:
    return stable_hash(normalize_sequence(sequence))


def molecule_cache_path(cache_dir: Path, smiles: str) -> Path:
    key = smiles_key(smiles)
    return Path(cache_dir) / "molecules" / key[:2] / f"{key}.npz"


def protein_cache_path(cache_dir: Path, sequence: str) -> Path:
    key = sequence_key(sequence)
    return Path(cache_dir) / "proteins" / key[:2] / f"{key}.npz"


def dictionaries_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "dictionaries.pkl"


def manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "manifest.json"


def split_manifest_path(cache_dir: Path, split_group: str, split_name: str, split: str) -> Path:
    safe_group = str(split_group).replace("/", "_")
    safe_name = str(split_name).replace("/", "_")
    return Path(cache_dir) / "split_manifests" / safe_group / safe_name / f"{split}.csv"


class IndexedDefaultDict(defaultdict):
    def __init__(self):
        super().__init__(self._new_index)

    def _new_index(self):
        return len(self)


def new_dictionaries() -> Dict[str, IndexedDefaultDict]:
    return {
        "word_dict": IndexedDefaultDict(),
        "atom_dict": IndexedDefaultDict(),
        "bond_dict": IndexedDefaultDict(),
        "fingerprint_dict": IndexedDefaultDict(),
        "edge_dict": IndexedDefaultDict(),
    }


def _freeze_dictionary(dictionary):
    return dict(dictionary)


def save_dictionaries(cache_dir: Path, dictionaries: Dict[str, Dict]) -> None:
    path = dictionaries_path(cache_dir)
    ensure_parent(path)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump({key: _freeze_dictionary(value) for key, value in dictionaries.items()}, handle)
    tmp_path.replace(path)


def load_dictionaries(cache_dir: Path) -> Dict[str, Dict]:
    path = dictionaries_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing DLKcat dictionaries: {path}. Run cache_embeddings.py first.")
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    dictionaries = new_dictionaries()
    for key, values in loaded.items():
        dictionaries[key].update(values)
    return dictionaries


def split_sequence(sequence: str, ngram: int, word_dict: Dict) -> np.ndarray:
    sequence = "-" + normalize_sequence(sequence) + "="
    return np.asarray([word_dict[sequence[i : i + ngram]] for i in range(len(sequence) - ngram + 1)], dtype=np.int64)


def create_atoms(mol, atom_dict: Dict) -> np.ndarray:
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for atom in mol.GetAromaticAtoms():
        idx = atom.GetIdx()
        atoms[idx] = (atoms[idx], "aromatic")
    return np.asarray([atom_dict[atom] for atom in atoms], dtype=np.int64)


def create_ijbonddict(mol, bond_dict: Dict):
    i_jbond_dict = defaultdict(lambda: [])
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_id = bond_dict[str(bond.GetBondType())]
        i_jbond_dict[i].append((j, bond_id))
        i_jbond_dict[j].append((i, bond_id))
    return i_jbond_dict


def extract_fingerprints(atoms: np.ndarray, i_jbond_dict, radius: int, fingerprint_dict: Dict, edge_dict: Dict) -> np.ndarray:
    if len(atoms) == 1 or radius == 0:
        fingerprints = [fingerprint_dict[int(atom)] for atom in atoms]
        return np.asarray(fingerprints, dtype=np.int64)

    nodes = atoms.tolist()
    i_jedge_dict = i_jbond_dict
    fingerprints = []
    for _ in range(radius):
        fingerprints = []
        for i, j_edge in i_jedge_dict.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            fingerprints.append(fingerprint_dict[fingerprint])
        nodes = fingerprints

        next_i_jedge_dict = defaultdict(lambda: [])
        for i, j_edge in i_jedge_dict.items():
            for j, edge in j_edge:
                both_side = tuple(sorted((nodes[i], nodes[j])))
                edge_id = edge_dict[(both_side, edge)]
                next_i_jedge_dict[i].append((j, edge_id))
        i_jedge_dict = next_i_jedge_dict
    return np.asarray(fingerprints, dtype=np.int64)


def create_adjacency(mol) -> np.ndarray:
    return np.asarray(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)


def featurize_smiles(smiles: str, radius: int, dictionaries: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    raw_smiles = str(smiles).strip()
    if "." in raw_smiles:
        raise ValueError(f"DLKcat preprocessing excludes multi-component SMILES containing '.': {raw_smiles}")
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {raw_smiles}")
    mol = Chem.AddHs(mol)
    atoms = create_atoms(mol, dictionaries["atom_dict"])
    i_jbond_dict = create_ijbonddict(mol, dictionaries["bond_dict"])
    fingerprints = extract_fingerprints(
        atoms,
        i_jbond_dict,
        radius,
        dictionaries["fingerprint_dict"],
        dictionaries["edge_dict"],
    )
    adjacency = create_adjacency(mol)
    return {"fingerprints": fingerprints, "adjacency": adjacency, "num_atoms": np.asarray([len(fingerprints)], dtype=np.int64)}


def featurize_sequence(sequence: str, ngram: int, dictionaries: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    words = split_sequence(sequence, ngram, dictionaries["word_dict"])
    return {"words": words, "length": np.asarray([len(words)], dtype=np.int64)}


def _save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


def save_molecule(cache_dir: Path, smiles: str, payload: Dict[str, np.ndarray]) -> None:
    _save_npz(molecule_cache_path(cache_dir, smiles), payload)


def save_protein(cache_dir: Path, sequence: str, payload: Dict[str, np.ndarray]) -> None:
    _save_npz(protein_cache_path(cache_dir, sequence), payload)


def load_molecule(cache_dir: Path, smiles: str) -> Dict[str, np.ndarray]:
    path = molecule_cache_path(cache_dir, smiles)
    if not path.exists():
        raise FileNotFoundError(f"Missing molecule cache for {smiles}: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_protein(cache_dir: Path, sequence: str) -> Dict[str, np.ndarray]:
    path = protein_cache_path(cache_dir, sequence)
    if not path.exists():
        raise FileNotFoundError(f"Missing protein cache for sequence hash {sequence_key(sequence)}: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def collect_unique_inputs(jobs: Iterable[Dict[str, str]], sequence_col: str, smiles_col: str) -> Tuple[List[str], List[str]]:
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = pd.read_parquet(job[split_key]) if str(job[split_key]).endswith(".parquet") else pd.read_csv(job[split_key])
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected `{sequence_col}` and `{smiles_col}` in {job[split_key]}")
            for sequence in frame[sequence_col].dropna().astype(str):
                normalized = normalize_sequence(sequence)
                if normalized:
                    sequences.add(normalized)
            for smiles in frame[smiles_col].dropna().astype(str):
                value = str(smiles).strip()
                if value and "." not in value:
                    smiles_values.add(value)
    return sorted(sequences), sorted(smiles_values)


def build_split_manifest(
    frame: pd.DataFrame,
    sequence_col: str,
    smiles_col: str,
    target_col: str,
    log10_col: str,
    source_path: Path,
    excluded_smiles=None,
) -> pd.DataFrame:
    excluded_smiles = set(excluded_smiles or [])
    rows = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        value = float(row[target_col])
        sequence = normalize_sequence(str(row[sequence_col]))
        smiles = str(row[smiles_col]).strip()
        if not sequence or not smiles or "." in smiles or smiles in excluded_smiles or not math.isfinite(value) or value <= 0:
            continue
        target_log10 = float(row[log10_col]) if log10_col in frame.columns and pd.notna(row[log10_col]) else math.log10(value)
        rows.append(
            {
                "row_index": int(idx),
                "smiles": smiles,
                "sequence": sequence,
                "smiles_key": smiles_key(smiles),
                "sequence_key": sequence_key(sequence),
                "target_value": value,
                "target_log2": math.log2(value),
                "target_log10": target_log10,
            }
        )
    return pd.DataFrame(rows)
