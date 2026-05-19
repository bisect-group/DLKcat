import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, summarize_seed_runs


def collect_runs(root: Path):
    rows = []
    for summary_path in sorted(root.glob("**/run_summary.json")):
        run_dir = summary_path.parent
        row = {"run_dir": str(run_dir)}
        parts = run_dir.relative_to(root).parts
        if len(parts) >= 3:
            row.update({"split_group": parts[-3], "split_name": parts[-2], "seed": parts[-1].replace("seed_", "")})
        for split in ("train", "val", "test"):
            path = run_dir / f"final_results_{split}.csv"
            if path.exists():
                metrics = pd.read_csv(path).iloc[0].to_dict()
                row.update({f"{split}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return rows


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
    metric_cols = [col for col in runs.columns if col.startswith("test_")]
    if rows and {"split_group", "split_name"}.issubset(runs.columns):
        summarize_seed_runs(rows, ["split_group", "split_name"], metric_cols).to_csv(out_dir / "summary_by_split.csv", index=False)
        summarize_seed_runs(rows, ["split_group"], metric_cols).to_csv(out_dir / "summary_by_split_group.csv", index=False)
    print(f"Saved aggregation outputs to {out_dir}")


if __name__ == "__main__":
    main()
