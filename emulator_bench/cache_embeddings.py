import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    hparams_cache_dir,
    load_hparams,
    normalize_threshold_args,
    read_table,
    require_columns,
    save_json,
)
from emulator_bench.feature_pipeline import (
    build_split_manifest,
    collect_unique_inputs,
    featurize_sequence,
    featurize_smiles,
    load_dictionaries,
    manifest_path,
    molecule_cache_path,
    new_dictionaries,
    protein_cache_path,
    save_dictionaries,
    save_molecule,
    save_protein,
    split_manifest_path,
)


def _limit_jobs_for_smoke(jobs, max_jobs):
    if max_jobs is None or max_jobs <= 0:
        return jobs
    return jobs[: int(max_jobs)]


def _read_split(path: Path, sample_rows: int = 0):
    frame = read_table(path)
    if sample_rows and sample_rows > 0:
        return frame.head(sample_rows).copy()
    return frame


def cache_features(args):
    hparams = load_hparams(Path(args.hparams_json))
    cache_dir = hparams_cache_dir(Path(args.base_dir), hparams, Path(args.embeddings_dir) if args.embeddings_dir else None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    jobs = _limit_jobs_for_smoke(jobs, args.max_jobs)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    started = time.time()
    dictionaries_file = cache_dir / "dictionaries.pkl"
    if dictionaries_file.exists() and not args.overwrite:
        dictionaries = load_dictionaries(cache_dir)
        print(f"Loaded existing dictionaries from {dictionaries_file}")
    else:
        dictionaries = new_dictionaries()

    sequences = set()
    smiles_values = set()
    split_frames = {}
    for job in jobs:
        for split_name, key in (("train", "train_path"), ("val", "val_path"), ("test", "test_path")):
            path = Path(job[key])
            frame = _read_split(path, sample_rows=args.sample_rows)
            require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], path)
            split_frames[(job["split_group"], job["split_name"], split_name)] = (job, split_name, path, frame)
            sequences.update(str(value) for value in frame[args.sequence_col].dropna().astype(str))
            for smiles in frame[args.smiles_col].dropna().astype(str):
                value = str(smiles).strip()
                if value and "." not in value:
                    smiles_values.add(value)

    sequences = sorted({sequence for sequence in sequences if str(sequence).strip()})
    smiles_values = sorted(smiles_values)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique sequences: {len(sequences)}")
    print(f"Unique single-component SMILES: {len(smiles_values)}")

    excluded_smiles = set()
    molecules_written = 0
    molecules_reused = 0
    for smiles in tqdm(smiles_values, desc="Caching DLKcat molecule features", unit="smiles"):
        path = molecule_cache_path(cache_dir, smiles)
        if path.exists() and not args.overwrite:
            molecules_reused += 1
            continue
        try:
            save_molecule(cache_dir, smiles, featurize_smiles(smiles, int(hparams["radius"]), dictionaries))
            molecules_written += 1
        except Exception as exc:
            excluded_smiles.add(smiles)
            if args.strict_smiles:
                raise
            print(f"Skipping SMILES `{smiles}`: {exc}")

    proteins_written = 0
    proteins_reused = 0
    for sequence in tqdm(sequences, desc="Caching DLKcat protein n-grams", unit="seq"):
        path = protein_cache_path(cache_dir, sequence)
        if path.exists() and not args.overwrite:
            proteins_reused += 1
            continue
        save_protein(cache_dir, sequence, featurize_sequence(sequence, int(hparams["ngram"]), dictionaries))
        proteins_written += 1

    save_dictionaries(cache_dir, dictionaries)

    manifest_rows = []
    split_rows = []
    for (split_group, split_job_name, split_name), (_job, _split_name, path, frame) in split_frames.items():
        split_manifest = build_split_manifest(
            frame,
            sequence_col=args.sequence_col,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            log10_col=args.log10_col,
            source_path=path,
            excluded_smiles=excluded_smiles,
        )
        out_path = split_manifest_path(cache_dir, split_group, split_job_name, split_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        split_manifest.to_csv(out_path, index=False)
        split_rows.append(
            {
                "split_group": split_group,
                "split_name": split_job_name,
                "split": split_name,
                "source_path": str(path),
                "manifest_path": str(out_path),
                "source_rows": int(len(frame)),
                "cached_rows": int(len(split_manifest)),
                "skipped_rows": int(len(frame) - len(split_manifest)),
            }
        )

    pd.DataFrame(split_rows).to_csv(cache_dir / "split_manifest_index.csv", index=False)
    manifest = {
        "cache_version": 1,
        "base_dir": str(args.base_dir),
        "cache_dir": str(cache_dir),
        "hparams": hparams,
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "target_col": args.target_col,
        "log10_col": args.log10_col,
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "jobs": jobs,
        "n_fingerprint": len(dictionaries["fingerprint_dict"]),
        "n_word": len(dictionaries["word_dict"]),
        "molecules_total": len(smiles_values),
        "molecules_written": molecules_written,
        "molecules_reused": molecules_reused,
        "proteins_total": len(sequences),
        "proteins_written": proteins_written,
        "proteins_reused": proteins_reused,
        "excluded_smiles_count": len(excluded_smiles),
        "elapsed_seconds": time.time() - started,
    }
    save_json(manifest_path(cache_dir), manifest)
    if excluded_smiles:
        pd.DataFrame({"smiles": sorted(excluded_smiles)}).to_csv(cache_dir / "excluded_smiles.csv", index=False)
    print(f"Saved DLKcat cache manifest to {manifest_path(cache_dir)}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Cache DLKcat WL molecular features and protein n-gram features once for EMULaToR splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="value")
    parser.add_argument("--log10_col", type=str, default="log10_value")
    parser.add_argument("--overwrite", "--overwrite_cache", action="store_true")
    parser.add_argument("--strict_smiles", action="store_true")
    parser.add_argument("--sample_rows", type=int, default=0, help="Smoke-test helper: use only this many rows per split.")
    parser.add_argument("--max_jobs", type=int, default=0, help="Smoke-test helper: use only this many discovered jobs.")
    args = parser.parse_args()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    cache_features(args)


if __name__ == "__main__":
    main()
