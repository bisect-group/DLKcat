import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_HPARAMS_PATH, DEFAULT_SPLIT_GROUPS, discover_split_jobs, hparams_cache_dir, load_hparams, normalize_threshold_args
from emulator_bench.feature_pipeline import split_manifest_path


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [sys.executable, str(CACHE_SCRIPT), "--base_dir", args.base_dir, "--hparams_json", args.hparams_json, "--split_groups", *args.split_groups]
    if getattr(args, "embeddings_dir", None):
        cmd.extend(["--embeddings_dir", args.embeddings_dir])
    if getattr(args, "thresholds", None):
        cmd.extend(["--thresholds", *args.thresholds])
    if getattr(args, "cache_overwrite", False):
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run DLKcat paper-setting benchmarks sequentially across split jobs.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)

    hparams = load_hparams(Path(args.hparams_json))
    cache_dir = hparams_cache_dir(Path(args.base_dir), hparams, Path(args.embeddings_dir) if args.embeddings_dir else None)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    for job in jobs:
        for seed in args.seeds:
            out_dir = Path(job["root_dir"]) / "dlkcat_results" / f"seed_{seed}"
            if (out_dir / "final_results_test.csv").exists() and not args.overwrite:
                continue
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
                str(out_dir),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
            ]
            if args.embeddings_dir:
                cmd.extend(["--embeddings_dir", args.embeddings_dir])
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
