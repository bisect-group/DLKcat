"""Rescore finished DLKcat runs on the full test split, with the saved last-epoch models.

Rows that were dropped only because their SMILES had an unbonded fragment are featurized with the frozen dictionaries
of the run and predicted on the device. Rows that were kept keep their stored predictions. The old files are renamed
to `*_pre_repair.csv`. Run with CUDA_VISIBLE_DEVICES="" to predict on the CPU.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_BASE_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    discover_split_jobs,
    load_hparams,
    read_table,
    regression_metrics,
    save_json,
)
from emulator_bench.feature_pipeline import (  # noqa: E402
    build_split_manifest,
    featurize_sequence,
    featurize_smiles,
    load_dictionaries,
    split_manifest_path,
)
from emulator_bench.modeling import build_model  # noqa: E402
from emulator_bench.train_single_target_tvt import LOG10_OF_2, evaluate  # noqa: E402


def backup(path: Path) -> Path:
    """Move the as-run file to `*_pre_repair` once and return the as-run file."""
    old = path.with_name(path.stem + "_pre_repair.csv")
    if not old.exists():
        path.rename(old)
    return old


class Rows(SimpleNamespace):
    def __len__(self):
        return len(self.samples)


def build_rows(manifest: pd.DataFrame, hparams, dictionaries):
    radius, ngram = int(hparams["radius"]), int(hparams["ngram"])
    molecules = {s: featurize_smiles(s, radius, dictionaries) for s in manifest["smiles"].unique()}
    proteins = {q: featurize_sequence(q, ngram, dictionaries)["words"] for q in manifest["sequence"].unique()}
    targets = torch.tensor(manifest["target_log2"].to_numpy(np.float32)).view(-1, 1, 1)
    samples = [
        (torch.from_numpy(molecules[s]["fingerprints"]), torch.from_numpy(molecules[s]["adjacency"]), torch.from_numpy(proteins[q]), t)
        for s, q, t in zip(manifest["smiles"], manifest["sequence"], targets.unbind())
    ]
    return Rows(samples=samples, target_log10=manifest["target_log10"].to_numpy(np.float32))


def rescore_split(job, args, hparams, dictionaries, cache_dir):
    group, name = job["split_group"], job["split_name"]
    run_dirs = [args.out_root / group / name / f"seed_{seed}" for seed in args.seeds]
    run_dirs = [d for d in run_dirs if (d / "lastmodel.pth").exists()]
    if not run_dirs:
        return
    frame = read_table(Path(job["test_path"]))
    new = build_split_manifest(frame, "sequence", "smiles", "value", "log10_value", Path(job["test_path"]), is_train=False)
    old = pd.read_csv(split_manifest_path(cache_dir, group, name, "test"), usecols=["row_index"])
    kept = new["row_index"].isin(old["row_index"]).to_numpy()
    assert kept.sum() == len(old), "as-run test rows are not a subset of the full test rows"
    added = new[~kept]
    reasons = {"unbonded fragment in dotted SMILES": int(added["smiles"].str.contains(".", regex=False).sum())}
    reasons["other"] = len(added) - reasons["unbonded fragment in dotted SMILES"]
    rows = build_rows(new, hparams, dictionaries)
    for run_dir in run_dirs:
        model = build_model(hparams, n_fingerprint=len(dictionaries["fingerprint_dict"]), n_word=len(dictionaries["word_dict"]))
        model.load_state_dict(torch.load(run_dir / "lastmodel.pth", map_location="cpu"))
        y_true, y_cpu, _ = evaluate(model, rows, torch.device("cpu"))

        pred_old = backup(run_dir / "pred_label_test.csv") if (run_dir / "pred_label_test.csv").exists() else run_dir / "pred_label_test_pre_repair.csv"
        results_old = run_dir / "final_results_test.csv"
        if results_old.exists():
            backup(results_old)
        stored = pd.read_csv(pred_old)
        assert len(stored) == len(old) and np.allclose(stored["y_true"], y_true[kept], atol=1e-6), "stored labels differ from the test rows"
        max_diff = float(np.abs(stored["y_pred"].to_numpy() - y_cpu[kept]).max())
        if max_diff > args.max_diff:
            raise ValueError(f"{run_dir}: CPU and stored predictions differ by {max_diff}")

        y_pred = y_cpu.copy()
        y_pred[kept] = stored["y_pred"].to_numpy()
        metrics = regression_metrics(y_true, y_pred)
        metrics["loss"] = float(np.sum((y_pred / LOG10_OF_2 - new["target_log2"].to_numpy(np.float64)) ** 2)) / len(new)
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(run_dir / "pred_label_test.csv", index=False)
        pd.DataFrame([metrics]).to_csv(run_dir / "final_results_test.csv", index=False)
        save_json(
            run_dir / "test_rescore_report.json",
            {"n_before": len(old), "n_after": len(new), "n_added": len(added), "reasons": reasons, "max_kept_row_diff": max_diff, "metrics": metrics},
        )
        print(f"{run_dir}: {len(old)} -> {len(new)} rows, max kept-row diff {max_diff:.2e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--cache_dir", type=Path, required=True, help="Embedding cache of the runs (dictionaries.pkl and the as-run split manifests).")
    parser.add_argument("--hparams_json", type=Path, default=DEFAULT_HPARAMS_PATH)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max_diff", type=float, default=1e-4, help="Largest allowed difference between stored and recomputed predictions.")
    args = parser.parse_args()

    hparams = load_hparams(args.hparams_json)
    cache_dir = args.cache_dir
    dictionaries = {key: dict(value) for key, value in load_dictionaries(cache_dir).items()}
    for job in discover_split_jobs(args.base_dir):
        rescore_split(job, args, hparams, dictionaries, cache_dir)


if __name__ == "__main__":
    main()
