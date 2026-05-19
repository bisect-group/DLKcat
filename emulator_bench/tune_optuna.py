import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_HPARAMS_PATH, DEFAULT_OPTUNA_RESULTS_DIRNAME, DEFAULT_SPLIT_GROUPS, discover_split_jobs, hparams_cache_dir, load_hparams, metric_direction, normalize_threshold_args
from emulator_bench.feature_pipeline import split_manifest_path
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def sqlite_path_from_storage(storage):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    raw_path = unquote(parsed.path or "")
    return Path(raw_path) if raw_path else None


def sqlite_has_optuna_schema(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return "version_info" in tables


def prepare_storage(args):
    db_path = sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and args.reset_storage:
        db_path.unlink()
    elif db_path.exists() and not sqlite_has_optuna_schema(db_path):
        raise RuntimeError(f"Optuna storage exists but is not an Optuna DB: {db_path}")


def suggest_hparams(trial, args):
    return {
        "batch_size": int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [32, 64, 96, 128, 160, 192]),
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 5e-5, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.35, 0.85),
        "decay_interval": trial.suggest_categorical("decay_interval", [5, 10, 15, 20]),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.0, 0.5, 1.0, 2.0, 5.0]),
        "patience": trial.suggest_categorical("patience", [0, 10, 20]),
    }


def run_trial_job(args, job, cache_dir, seed, hparams, trial_number):
    trial_root = Path(job["root_dir"]) / DEFAULT_OPTUNA_RESULTS_DIRNAME / f"trial_{trial_number}" / f"seed_{seed}"
    metric_path = trial_root / f"final_results_{args.eval_split}.csv"
    if not metric_path.exists() or args.overwrite_runs:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--base_dir",
            args.base_dir,
            "--hparams_json",
            args.hparams_json,
            "--train_manifest",
            str(split_manifest_path(cache_dir, job["split_group"], job["split_name"], "train")),
            "--val_manifest",
            str(split_manifest_path(cache_dir, job["split_group"], job["split_name"], "val")),
            "--test_manifest",
            str(split_manifest_path(cache_dir, job["split_group"], job["split_name"], "test")),
            "--out_dir",
            str(trial_root),
            "--task_name",
            f"optuna_trial_{trial_number}_{job['split_group']}_{job['split_name']}_seed{seed}",
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(hparams["batch_size"]),
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--lr_decay",
            str(hparams["lr_decay"]),
            "--decay_interval",
            str(hparams["decay_interval"]),
            "--clip_grad",
            str(hparams["clip_grad"]),
            "--patience",
            str(hparams["patience"]),
            "--monitor_metric",
            args.metric,
            "--val_every",
            str(args.val_every),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.embeddings_dir:
            cmd.extend(["--embeddings_dir", args.embeddings_dir])
        if args.pin_memory:
            cmd.append("--pin_memory")
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(metric_path).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError(f"Metric `{args.metric}` not found in {metric_path}")
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune retraining-safe DLKcat optimization parameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--metric", choices=["rmse", "mae", "mse", "r2_score", "pearson", "spearman", "loss"], default="rmse")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="dlkcat_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = f"sqlite:///{Path(args.base_dir) / 'optuna_studies' / (args.study_name + '.db')}"
    maybe_cache_embeddings(args)
    prepare_storage(args)
    hparams_base = load_hparams(Path(args.hparams_json))
    cache_dir = hparams_cache_dir(Path(args.base_dir), hparams_base, Path(args.embeddings_dir) if args.embeddings_dir else None)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=metric_direction(args.metric),
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )

    def objective(trial):
        hparams = suggest_hparams(trial, args)
        values = []
        for job in jobs:
            for seed in args.seeds:
                values.append(run_trial_job(args, job, cache_dir, seed, hparams, trial.number))
        mean_value = float(sum(values) / len(values))
        trial.set_user_attr("hparams", hparams)
        trial.set_user_attr("values", values)
        return mean_value

    study.optimize(objective, n_trials=args.n_trials)
    out = Path(args.base_dir) / "optuna_studies" / f"{args.study_name}_best.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"best_value": study.best_value, "best_hparams": study.best_params, "study_name": args.study_name, "storage": args.storage}, handle, indent=2)
    print(f"Saved best Optuna result to {out}")


if __name__ == "__main__":
    main()
