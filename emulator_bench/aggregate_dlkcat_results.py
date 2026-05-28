import argparse
import sys
from pathlib import Path

import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR


def _parse_run_identity(root: Path, run_dir: Path):
    parts = run_dir.relative_to(root).parts
    if len(parts) < 3:
        return {}
    seed_name = parts[-1]
    seed = seed_name.replace("seed_", "") if seed_name.startswith("seed_") else seed_name
    return {
        "split_group": parts[-3],
        "split_name": parts[-2],
        "seed": int(seed) if str(seed).isdigit() else seed,
    }


def _metric_run_dirs(root: Path):
    run_dirs = {path.parent for path in root.glob("**/final_results_*.csv")}
    run_dirs.update(path.parent for path in root.glob("**/run_summary.json"))
    return sorted(path for path in run_dirs if path.name.startswith("seed_"))


def collect_runs(root: Path):
    rows = []
    for run_dir in _metric_run_dirs(root):
        row = {"run_dir": str(run_dir)}
        row.update(_parse_run_identity(root, run_dir))

        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as handle:
                summary = json.load(handle)
            for key in ("task_name", "precision_mode", "best_epoch", "best_val_metric", "monitor_metric", "elapsed_seconds"):
                if key in summary:
                    row[key] = summary[key]

        for split in ("train", "val", "test"):
            path = run_dir / f"final_results_{split}.csv"
            if path.exists():
                metrics = pd.read_csv(path).iloc[0].to_dict()
                row.update({f"{split}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return rows


def summarize_mean_std(runs: pd.DataFrame, group_cols):
    if runs.empty or not set(group_cols).issubset(runs.columns):
        return pd.DataFrame()
    excluded = {"seed"}
    numeric_cols = [
        col
        for col in runs.select_dtypes(include="number").columns
        if col not in excluded and col not in group_cols
    ]
    if not numeric_cols:
        return runs[group_cols].drop_duplicates().reset_index(drop=True)
    summary = runs.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Aggregate DLKcat emulator_bench result directories.")
    parser.add_argument("--root", type=str, default=str(Path(DEFAULT_BASE_DIR) / "retrain_paper_settings"))
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(root)
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "aggregated_run_summaries.csv", index=False)
    if rows and {"split_group", "split_name"}.issubset(runs.columns):
        summarize_mean_std(runs, ["split_group", "split_name"]).to_csv(out_dir / "summary_by_split.csv", index=False)
        summarize_mean_std(runs, ["split_group"]).to_csv(out_dir / "summary_by_split_group.csv", index=False)
    print(f"Aggregated {len(runs)} seed runs. Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
