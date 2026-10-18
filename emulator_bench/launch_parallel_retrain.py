import argparse
import json
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
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    hparams_cache_dir,
    load_hparams,
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
        "--embeddings_dir",
        args.embeddings_dir,
        "--split_groups",
        *args.split_groups,
    ]
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_experiments(args, jobs, cache_dir):
    experiments = []
    for job in jobs:
        for seed in args.seeds:
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "run_dir": Path(args.output_root) / job["split_group"] / job["split_name"] / f"seed_{seed}",
                    **{f"{split}_manifest": split_manifest_path(cache_dir, job["split_group"], job["split_name"], split) for split in ("train", "val", "test")},
                }
            )
    return experiments


def train_command(args, exp, device):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--hparams_json",
        args.hparams_json,
        "--embeddings_dir",
        args.embeddings_dir,
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
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.amp:
        cmd.append("--amp")
    return cmd


def _result_identity(exp):
    return {key: exp[key] for key in ("run_dir", "split_group", "split_name", "difficulty", "seed")}


def _finished_with_epochs(run_dir: Path, epochs: int) -> bool:
    summary_path = run_dir / "run_summary.json"
    return (run_dir / "final_results_test.csv").exists() and summary_path.exists() and json.loads(summary_path.read_text())["epochs"] == epochs


def run_experiment(args, exp, gpu_id):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    if _finished_with_epochs(exp["run_dir"], args.effective_epochs) and not args.overwrite:
        return {"status": "skipped_exists", "gpu_id": str(gpu_id), **_result_identity(exp)}
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        if not Path(exp[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {exp[key]}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if args.device.startswith("cuda") else args.device
    subprocess.run(train_command(args, exp, device), check=True, cwd=str(REPO_ROOT), env=env)
    return {"status": "completed", "gpu_id": str(gpu_id), **_result_identity(exp)}


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

    threads = []
    for gpu_id in args.gpus:
        for slot in range(args.runs_per_gpu):
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
    parser = argparse.ArgumentParser(description="Train DLKcat with its original settings on all discovered EMULaToR splits and seeds.")
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids; use any id with --device cpu.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent training processes per GPU.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR), help="Task directory, e.g. data/processed/datasets/kinetic_dataset/kcat.")
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision (bf16 on Ampere or newer, else fp16). The original code trains in fp32.")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test override of the original epoch count.")
    args = parser.parse_args()

    maybe_cache(args)
    hparams = load_hparams(Path(args.hparams_json))
    args.effective_epochs = args.epochs if args.epochs is not None else int(hparams["epochs"])
    cache_dir = hparams_cache_dir(hparams, args.embeddings_dir, args.base_dir)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    results = run_parallel(args, build_experiments(args, jobs, cache_dir))
    summarize(Path(args.output_root), results)
    failed = [result for result in results if result["status"] == "failed"]
    if failed:
        sys.exit("\n".join(f"FAILED {result['run_dir']}: {result['error']}" for result in failed))


if __name__ == "__main__":
    main()
