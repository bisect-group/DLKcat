"""Unbonded fragments (lone ions or atoms in dotted SMILES) get one fingerprint per atom and never grow frozen dictionaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

from emulator_bench.feature_pipeline import featurize_smiles, new_dictionaries  # noqa: E402
from rdkit import Chem  # noqa: E402

FIXTURE = pd.read_csv(Path(__file__).parent / "fixtures" / "train.csv")
NORMAL = sorted({s for s in FIXTURE["smiles"] if "." not in s})[:20]
FRAGMENTS = ["C1=CCC=C1.C1=CCC=C1.[Fe+3]", "[H].[H]", "CC(=O)O.[K+]"]


def frozen(dictionaries):
    return {key: dict(value) for key, value in dictionaries.items()}


def test_normal_smiles_unchanged_by_frozen_lookup():
    dictionaries = new_dictionaries()
    grown = [featurize_smiles(s, 2, dictionaries) for s in NORMAL]
    sizes = {key: len(value) for key, value in dictionaries.items()}
    lookup = frozen(dictionaries)
    for smiles, expected in zip(NORMAL, grown):
        features = featurize_smiles(smiles, 2, lookup)
        assert all(np.array_equal(features[key], expected[key]) for key in expected)
    assert {key: len(value) for key, value in lookup.items()} == sizes


def test_unbonded_fragments_have_one_fingerprint_per_atom_and_frozen_dictionaries_stay_fixed():
    dictionaries = new_dictionaries()
    for smiles in NORMAL:
        featurize_smiles(smiles, 2, dictionaries)
    lookup = frozen(dictionaries)
    sizes = {key: len(value) for key, value in lookup.items()}
    for smiles in FRAGMENTS:
        n_atoms = Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()
        features = featurize_smiles(smiles, 2, lookup)
        assert len(features["fingerprints"]) == n_atoms == features["adjacency"].shape[0]
        assert features["num_atoms"][0] == n_atoms
    assert {key: len(value) for key, value in lookup.items()} == sizes
