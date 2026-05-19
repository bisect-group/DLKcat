import argparse
import json
import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.launch_parallel_retrain import main as paper_launcher_main


def main():
    parser = argparse.ArgumentParser(description="Convert an Optuna best result into a DLKcat retrain hparams JSON, then use launch_parallel_retrain.")
    parser.add_argument("--hparams_json_out", type=str, default=None)
    parser.add_argument("--study_name", type=str, default="dlkcat_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--best_json", type=str, default=None)
    args, remaining = parser.parse_known_args()

    if args.best_json:
        with open(args.best_json, "r") as handle:
            payload = json.load(handle)
        best = payload.get("best_hparams", payload)
    else:
        if not args.storage:
            raise ValueError("Provide --best_json or --storage.")
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        best = dict(study.best_params)

    base_hparams_path = None
    for idx, token in enumerate(remaining):
        if token == "--hparams_json" and idx + 1 < len(remaining):
            base_hparams_path = remaining[idx + 1]
            break
    if base_hparams_path is None:
        base_hparams_path = str(Path(__file__).resolve().parent / "default_hparams_paper.json")
    with open(base_hparams_path, "r") as handle:
        hparams = json.load(handle)
    hparams.update(best)
    out_path = Path(args.hparams_json_out) if args.hparams_json_out else Path(base_hparams_path).with_name("default_hparams_from_optuna.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(hparams, handle, indent=2, sort_keys=True)

    cleaned = []
    skip_next = False
    for token in remaining:
        if skip_next:
            skip_next = False
            continue
        if token == "--hparams_json":
            skip_next = True
            continue
        cleaned.append(token)
    sys.argv = [sys.argv[0], *cleaned, "--hparams_json", str(out_path)]
    paper_launcher_main()


if __name__ == "__main__":
    main()
