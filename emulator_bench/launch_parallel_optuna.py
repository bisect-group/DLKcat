import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_HPARAMS_PATH, DEFAULT_SPLIT_GROUPS


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def worker(args, gpu_id, worker_index):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--hparams_json",
        args.hparams_json,
        "--split_groups",
        *args.split_groups,
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--n_trials",
        str(args.n_trials),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
        "--epochs",
        str(args.epochs),
        "--num_workers",
        str(args.num_workers),
        "--sampler_seed",
        str(args.sampler_seed + worker_index),
    ]
    if args.embeddings_dir:
        cmd.extend(["--embeddings_dir", args.embeddings_dir])
    if args.skip_cache:
        cmd.append("--skip_cache")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main():
    parser = argparse.ArgumentParser(description="Launch parallel single-GPU Optuna workers for DLKcat.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--study_name", type=str, default="dlkcat_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    args = parser.parse_args()
    if args.storage is None:
        args.storage = f"sqlite:///{Path(args.base_dir) / 'optuna_studies' / (args.study_name + '.db')}"
    threads = []
    worker_index = 0
    for gpu_id in args.gpus:
        for _slot in range(args.trials_per_gpu):
            thread = threading.Thread(target=worker, args=(args, gpu_id, worker_index), daemon=True)
            thread.start()
            threads.append(thread)
            worker_index += 1
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
