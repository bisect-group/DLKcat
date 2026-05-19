import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_RETRAIN_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    hparams_cache_dir,
    load_hparams,
    normalize_threshold_args,
    summarize_seed_runs,
)
from emulator_bench.feature_pipeline import split_manifest_path


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--hparams_json",
        args.hparams_json,
        "--split_groups",
        *args.split_groups,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--log10_col",
        args.log10_col,
    ]
    if args.embeddings_dir:
        cmd.extend(["--embeddings_dir", args.embeddings_dir])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_experiments(args, jobs, cache_dir):
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / DEFAULT_RETRAIN_DIRNAME
    experiments = []
    for job in jobs:
        for seed in args.seeds:
            run_dir = output_root / job["split_group"] / job["split_name"] / f"seed_{seed}"
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "train_manifest": split_manifest_path(cache_dir, job["split_group"], job["split_name"], "train"),
                    "val_manifest": split_manifest_path(cache_dir, job["split_group"], job["split_name"], "val"),
                    "test_manifest": split_manifest_path(cache_dir, job["split_group"], job["split_name"], "test"),
                }
            )
    return experiments


def train_command(args, exp, device):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--hparams_json",
        args.hparams_json,
        "--train_manifest",
        str(exp["train_manifest"]),
        "--val_manifest",
        str(exp["val_manifest"]),
        "--test_manifest",
        str(exp["test_manifest"]),
        "--out_dir",
        str(exp["run_dir"]),
        "--task_name",
        f"{exp['split_group']}_{exp['split_name']}_seed{exp['seed']}",
        "--seed",
        str(exp["seed"]),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
    ]
    if args.embeddings_dir:
        cmd.extend(["--embeddings_dir", args.embeddings_dir])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    return cmd


def run_experiment(args, exp, gpu_id):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    final_path = exp["run_dir"] / "final_results_test.csv"
    if final_path.exists() and not args.overwrite:
        return {"status": "skipped_exists", "gpu_id": str(gpu_id), **_result_identity(exp)}
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        if not Path(exp[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {exp[key]}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if args.device.startswith("cuda") else args.device
    subprocess.run(train_command(args, exp, device), check=True, cwd=str(REPO_ROOT), env=env)
    return {"status": "completed", "gpu_id": str(gpu_id), **_result_identity(exp)}


def _result_identity(exp):
    return {
        "run_dir": str(exp["run_dir"]),
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def run_parallel(args, experiments):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)
    results = []
    result_lock = threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_experiment(args, exp, gpu_id)
                result["slot_index"] = slot_index
            except Exception as exc:
                result = {"status": "failed", "gpu_id": str(gpu_id), "slot_index": slot_index, "error": str(exc), **_result_identity(exp)}
            with result_lock:
                results.append(result)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot in range(args.trials_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu_id, slot), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def summarize(output_root: Path, results):
    rows = []
    for result in results:
        row = dict(result)
        run_dir = Path(result["run_dir"])
        for split in ("train", "val", "test"):
            path = run_dir / f"final_results_{split}.csv"
            if path.exists():
                metrics = pd.read_csv(path).iloc[0].to_dict()
                row.update({f"{split}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(output_root / "retrain_summary_runs.csv", index=False)
    metric_cols = [col for col in runs.columns if col.startswith("test_")]
    summarize_seed_runs(rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(output_root / "retrain_summary_thresholds.csv", index=False)
    summarize_seed_runs(rows, ["split_group"], metric_cols).to_csv(output_root / "retrain_summary_by_split_group.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Retrain DLKcat split jobs in parallel using paper/default settings.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="value")
    parser.add_argument("--log10_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be positive")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache(args)
    hparams = load_hparams(Path(args.hparams_json))
    cache_dir = hparams_cache_dir(Path(args.base_dir), hparams, Path(args.embeddings_dir) if args.embeddings_dir else None)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    experiments = build_experiments(args, jobs, cache_dir)
    results = run_parallel(args, experiments)
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / DEFAULT_RETRAIN_DIRNAME
    summarize(output_root, results)


if __name__ == "__main__":
    main()
