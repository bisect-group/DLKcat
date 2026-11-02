"""Equivalence of the emulator_bench wrapper with the original DLKcat code.

Run A drives the original `Code/model/preprocess_all.py` and `Code/model/run_model.py` unchanged. Run B is the wrapper
(`cache_embeddings.cache_features`, then `train_single_target_tvt.main`). Both use 64 train, 16 val and 16 test real
rows (tests/fixtures) of one kcat split, seed 666 and 3 epochs. The CPU tests compare features, initial weights,
per-epoch train loss and val metrics, final weights and test predictions. The GPU test compares the wrapper with the
original on one fp32 CUDA epoch against a noise floor and needs a visible CUDA device.
"""

import importlib.util
import json
import math
import sys
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

from emulator_bench import cache_embeddings, train_single_target_tvt  # noqa: E402
from emulator_bench.common import hparams_cache_dir, load_hparams  # noqa: E402
from emulator_bench.dataset import PreloadedSplit  # noqa: E402
from emulator_bench.feature_pipeline import load_dictionaries, load_molecule, load_protein  # noqa: E402
from emulator_bench.modeling import build_model  # noqa: E402

SEED, EPOCHS = 666, 3
SPLIT = "random_splits_grouped_sequence"
HPARAMS = load_hparams()


def load_original(name):
    """Import an original script under a fresh module, so its global dictionaries start empty."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "Code" / "model" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_model():
    """The original `run_model.py` with the globals its classes read. Newer numpy no longer converts the 1-element arrays
    that `Trainer.train` and `Tester.test` pass to `math.pow`, so the module gets a `math` that does the conversion."""
    run = load_original("run_model")
    run.math = SimpleNamespace(pow=lambda base, exponent: math.pow(base, np.asarray(exponent).item()), log10=lambda value: math.log10(np.asarray(value).item()))
    for key in ("dim", "layer_gnn", "window", "layer_cnn", "layer_output"):
        setattr(run, key, int(HPARAMS[key]))
    run.lr, run.weight_decay = HPARAMS["lr"], HPARAMS["weight_decay"]
    return run


def fixture_frames():
    return {name: pd.read_csv(FIXTURES / f"{name}.csv") for name in ("train", "val", "test")}


def original_features(frames):
    """Original preprocessing loop of `preprocess_all.main`, over train, val and test rows in order."""
    pre = load_original("preprocess_all")
    compounds, adjacencies, proteins, targets = {}, {}, {}, {}
    for name, frame in frames.items():
        compounds[name], adjacencies[name], proteins[name], targets[name] = [], [], [], []
        for smiles, sequence, value in zip(frame["smiles"], frame["sequence"], frame["value"]):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = pre.create_atoms(mol)
            fingerprints = pre.extract_fingerprints(atoms, pre.create_ijbonddict(mol), HPARAMS["radius"])
            compounds[name].append(fingerprints)
            adjacencies[name].append(pre.create_adjacency(mol))
            proteins[name].append(pre.split_sequence(sequence, HPARAMS["ngram"]))
            targets[name].append(np.array([math.log2(float(value))]))
    return pre, compounds, adjacencies, proteins, targets


@pytest.fixture(scope="module")
def wrapper_cache(tmp_path_factory):
    """Cache of the fixture rows. The task table holds them in train, val, test order, as the original dataset would."""
    base = tmp_path_factory.mktemp("kcat") / "kcat"
    frames = fixture_frames()
    (base / SPLIT).mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(base / SPLIT / f"{name}.csv", index=False)
    pd.concat(frames.values()).to_parquet(base / "kcat_kinetic_params_1d.parquet", index=False)
    args = Namespace(
        base_dir=base, embeddings_dir=base.parent / "embeddings", hparams_json=str(ROOT / "emulator_bench" / "default_hparams_paper.json"),
        split_groups=[SPLIT], thresholds=None, max_jobs=0, sample_rows=0, sequence_col="sequence", smiles_col="smiles",
        target_col="value", log10_col="log10_value", overwrite=False, strict_smiles=False,
    )
    cache_embeddings.cache_features(args)
    return base, hparams_cache_dir(HPARAMS, args.embeddings_dir, base)


@pytest.fixture(scope="module")
def original_run(wrapper_cache):
    """Run A: the original Trainer and Tester on the fixture rows for EPOCHS epochs. Seeds as in the wrapper."""
    frames = fixture_frames()
    pre, compounds, adjacencies, proteins, targets = original_features(frames)
    run = load_run_model()
    run.n_fingerprint, run.n_word, run.device = len(pre.fingerprint_dict), len(pre.word_dict), torch.device("cpu")

    def dataset(name):
        return list(zip(
            [torch.LongTensor(c) for c in compounds[name]], [torch.FloatTensor(a) for a in adjacencies[name]],
            [torch.LongTensor(p) for p in proteins[name]], [torch.FloatTensor(t) for t in targets[name]],
        ))

    train, val, test = dataset("train"), dataset("val"), dataset("test")
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = run.KcatPrediction()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    trainer, tester = run.Trainer(model), run.Tester(model)
    log = []
    for epoch in range(1, EPOCHS + 1):
        if epoch % HPARAMS["decay_interval"] == 0:
            trainer.optimizer.param_groups[0]["lr"] *= HPARAMS["lr_decay"]
        loss_train, _rmse, _r2 = trainer.train(train)
        mae_dev, rmse_dev, r2_dev = tester.test(val)
        log.append({"loss_sum": float(loss_train), "val_mae": mae_dev, "val_rmse": rmse_dev, "val_r2_score": r2_dev})
    with torch.no_grad():
        test_pred = [math.log10(math.pow(2, float(model.forward(sample[:-1])))) for sample in test]
    return {"pre": pre, "compounds": compounds, "adjacencies": adjacencies, "proteins": proteins, "initial": initial,
            "log": log, "model": model, "test_pred": np.array(test_pred), "n_train": len(train)}


def run_wrapper(base, out, epochs, patch):
    argv = ["train", "--base_dir", str(base), "--embeddings_dir", str(base.parent / "embeddings"), "--split_group", SPLIT,
            "--out_dir", str(out), "--seed", str(SEED), "--epochs", str(epochs), "--device", "cpu"]
    patch.setattr(sys, "argv", argv)
    train_single_target_tvt.main()
    return out


@pytest.fixture(scope="module")
def wrapper_run(wrapper_cache, tmp_path_factory, monkeypatch_module):
    return run_wrapper(wrapper_cache[0], tmp_path_factory.mktemp("run"), EPOCHS, monkeypatch_module)


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_features_match_original(wrapper_cache, original_run):
    _base, cache_dir = wrapper_cache
    pre = original_run["pre"]
    dictionaries = load_dictionaries(cache_dir)
    for key, original in (("word_dict", pre.word_dict), ("atom_dict", pre.atom_dict), ("bond_dict", pre.bond_dict),
                          ("fingerprint_dict", pre.fingerprint_dict), ("edge_dict", pre.edge_dict)):
        assert list(dictionaries[key].items()) == list(original.items()), key
    for name, frame in fixture_frames().items():
        for i, (smiles, sequence) in enumerate(zip(frame["smiles"], frame["sequence"])):
            molecule, protein = load_molecule(cache_dir, smiles), load_protein(cache_dir, sequence)
            assert np.array_equal(molecule["fingerprints"], original_run["compounds"][name][i])
            assert np.array_equal(molecule["adjacency"], original_run["adjacencies"][name][i])
            assert np.array_equal(protein["words"], original_run["proteins"][name][i])


def test_initial_weights_match_original(wrapper_cache, original_run):
    _base, cache_dir = wrapper_cache
    dictionaries = load_dictionaries(cache_dir)
    torch.manual_seed(SEED)
    model = build_model(HPARAMS, len(dictionaries["fingerprint_dict"]), len(dictionaries["word_dict"]))
    assert list(model.state_dict()) == list(original_run["initial"])
    assert all(torch.equal(value, original_run["initial"][key]) for key, value in model.state_dict().items())


def test_training_matches_original(wrapper_run, original_run):
    log = pd.read_csv(wrapper_run / "logfile.csv")
    for epoch, original in enumerate(original_run["log"]):
        row = log.iloc[epoch]
        assert row["train_loss"] * original_run["n_train"] == pytest.approx(original["loss_sum"], rel=1e-5)
        for key in ("mae", "rmse", "r2_score"):
            assert row[f"val_{key}"] == pytest.approx(original[f"val_{key}"], rel=1e-4, abs=1e-6), (epoch, key)
    weights = torch.load(wrapper_run / "lastmodel.pth")
    for key, value in original_run["model"].state_dict().items():
        assert torch.equal(weights[key], value), key


def test_test_predictions_match_original(wrapper_run, original_run):
    predictions = pd.read_csv(wrapper_run / "pred_label_test.csv")["y_pred"].to_numpy()
    assert np.allclose(predictions, original_run["test_pred"], rtol=1e-5, atol=1e-6)


def test_resume_matches_uninterrupted_run(wrapper_cache, wrapper_run, tmp_path, monkeypatch):
    """A run stopped after 2 epochs and restarted for 3 epochs ends with the weights of the uninterrupted 3-epoch run."""
    run_wrapper(wrapper_cache[0], tmp_path, EPOCHS - 1, monkeypatch)
    run_wrapper(wrapper_cache[0], tmp_path, EPOCHS, monkeypatch)
    resumed, straight = torch.load(tmp_path / "lastmodel.pth"), torch.load(wrapper_run / "lastmodel.pth")
    assert all(torch.equal(resumed[key], value) for key, value in straight.items())
    assert len(pd.read_csv(tmp_path / "logfile.csv")) == EPOCHS


def test_drop_counts_are_zero(wrapper_run):
    assert json.loads((wrapper_run / "run_summary.json").read_text())["rows_dropped"] == {"train": 0, "val": 0, "test": 0}


@pytest.mark.gpu
def test_gpu_epoch_matches_original_within_noise(wrapper_cache):
    """One fp32 CUDA epoch on the first 24 train rows. gap(wrapper, original) must stay within 10x the noise floor.

    floor = gap(wrapper, wrapper with weights perturbed at 1e-6, the fp32 unit roundoff). The negative control
    (EQUIV_NEGATIVE_CONTROL=1) scales the wrapper weights by 1.05 and must fail.
    """
    import os

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device is visible")
    _base, cache_dir = wrapper_cache
    device = torch.device("cuda")
    store = {}
    manifest = cache_dir / "split_manifests" / SPLIT / "all" / "train.csv"
    split = PreloadedSplit(manifest, cache_dir, device, store)
    split.samples = split.samples[:24]
    dictionaries = load_dictionaries(cache_dir)

    def wrapper_losses(scale=1.0, noise=0.0):
        torch.manual_seed(SEED)
        model = build_model(HPARAMS, len(dictionaries["fingerprint_dict"]), len(dictionaries["word_dict"])).to(device)
        generator = torch.Generator(device=device).manual_seed(1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale * (1 + noise * torch.randn(parameter.shape, device=device, generator=generator)))
        optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
        step0 = float(torch.nn.functional.mse_loss(model(*split.samples[0][:3]), split.samples[0][3]))
        mean = train_single_target_tvt.train_one_epoch(model, split, order, optimizer, device)["loss"]
        return np.array([step0, mean])

    order = np.arange(24)
    np.random.seed(SEED)
    np.random.shuffle(order)
    original = original_gpu_losses(wrapper_cache, device, order)
    scale = 1.05 if os.environ.get("EQUIV_NEGATIVE_CONTROL") else 1.0
    prod = wrapper_losses(scale=scale)
    floor = np.abs(prod - wrapper_losses(scale=scale, noise=1e-6)) / np.abs(prod)
    gap = np.abs(prod - original) / np.abs(prod)
    print(f"gap {gap}, floor {floor}")
    assert (gap <= 10 * np.maximum(floor, 1e-7)).all()


def original_gpu_losses(wrapper_cache, device, order):
    """Original `Trainer.train` on CUDA over the first 24 fixture train rows: loss of the first sample, mean loss over the epoch."""
    _pre, compounds, adjacencies, proteins, targets = original_features({"train": fixture_frames()["train"].head(24)})
    run = load_run_model()
    dictionaries = load_dictionaries(wrapper_cache[1])
    run.n_fingerprint, run.n_word, run.device = len(dictionaries["fingerprint_dict"]), len(dictionaries["word_dict"]), device
    data = list(zip(
        [torch.LongTensor(c).to(device) for c in compounds["train"]], [torch.FloatTensor(a).to(device) for a in adjacencies["train"]],
        [torch.LongTensor(p).to(device) for p in proteins["train"]], [torch.FloatTensor(t).to(device) for t in targets["train"]],
    ))
    torch.manual_seed(SEED)
    model = run.KcatPrediction().to(device)
    step0 = float(torch.nn.functional.mse_loss(model.forward(data[0][:-1]), data[0][-1]))
    np.random.seed(SEED)
    loss_sum, _rmse, _r2 = run.Trainer(model).train(data)
    return np.array([step0, float(loss_sum) / len(data)])
