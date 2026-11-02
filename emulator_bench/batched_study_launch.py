"""Run the batch-size study: batch sizes x splits x seeds, `--runs_at_once` runs on the GPU at a time."""

import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_SEEDS,
    discover_split_jobs,
    hparams_cache_dir,
    load_hparams,
)
from emulator_bench.feature_pipeline import split_manifest_path  # noqa: E402

STUDY_SPLITS = ["random_splits_grouped_sequence", "enzyme_sequence_splits", "substrate_splits"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--splits", nargs="+", default=STUDY_SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--runs_at_once", type=int, default=9)
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test override.")
    args = parser.parse_args()

    cache_dir = hparams_cache_dir(load_hparams(DEFAULT_HPARAMS_PATH), args.embeddings_dir, args.base_dir)
    jobs = discover_split_jobs(Path(args.base_dir), args.splits)
    work = queue.Queue()
    for batch_size in args.batch_sizes:
        for job in jobs:
            for seed in args.seeds:
                out = Path(args.output_root) / f"B{batch_size}" / job["split_group"] / job["split_name"] / f"seed_{seed}"
                manifests = [x for split in ("train", "val", "test") for x in (f"--{split}_manifest", str(split_manifest_path(cache_dir, job["split_group"], job["split_name"], split)))]
                cmd = [sys.executable, str(REPO_ROOT / "emulator_bench" / "batched_study.py"), "--batch_size", str(batch_size), "--out_dir", str(out),
                       "--seed", str(seed), "--base_dir", args.base_dir, "--embeddings_dir", args.embeddings_dir, *manifests]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                if not (out / "run_summary.json").exists():
                    work.put(cmd)
    failed = []

    def worker():
        while True:
            try:
                cmd = work.get_nowait()
            except queue.Empty:
                return
            if subprocess.run(cmd).returncode:
                failed.append(cmd[cmd.index("--out_dir") + 1])

    threads = [threading.Thread(target=worker) for _ in range(args.runs_at_once)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        sys.exit("FAILED " + "\n".join(failed))


if __name__ == "__main__":
    main()
