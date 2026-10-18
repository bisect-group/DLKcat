import argparse
import sys
import time
from pathlib import Path

import pandas as pd
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    CACHE_VERSION,
    hparams_cache_dir,
    normalize_sequence,
    task_table_inputs,
    load_hparams,
    read_table,
    require_columns,
    save_json,
)
from emulator_bench.feature_pipeline import (
    build_split_manifest,
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


CACHE_BATCH = 2000


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
    cache_dir = hparams_cache_dir(hparams, args.embeddings_dir, args.base_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    jobs = _limit_jobs_for_smoke(jobs, args.max_jobs)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    started = time.time()
    if args.overwrite:
        manifest_path(cache_dir).unlink(missing_ok=True)
    dictionaries_file = cache_dir / "dictionaries.pkl"
    if dictionaries_file.exists() and not args.overwrite:
        dictionaries = load_dictionaries(cache_dir)
        print(f"Loaded existing dictionaries from {dictionaries_file}")
    else:
        dictionaries = new_dictionaries()

    table_smiles, table_sequences, table_fingerprint = task_table_inputs(args.base_dir)
    sequences = set()
    smiles_values = set()
    split_frames = {}
    for job in jobs:
        for split_name, key in (("train", "train_path"), ("val", "val_path"), ("test", "test_path")):
            path = Path(job[key])
            frame = _read_split(path, sample_rows=args.sample_rows)
            require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], path)
            split_frames[(job["split_group"], job["split_name"], split_name)] = (job, split_name, path, frame)
            sequences.update(normalize_sequence(value) for value in frame[args.sequence_col].dropna().astype(str))
            for smiles in frame[args.smiles_col].dropna().astype(str):
                value = smiles.strip()
                if value and not (split_name == "train" and "." in value):
                    smiles_values.add(value)

    # Ids follow the original: first occurrence in the full task table. Split-only items come after, sorted.
    table_smiles = list(dict.fromkeys(table_smiles))
    table_sequences = list(dict.fromkeys(table_sequences))
    smiles_values = table_smiles + sorted(smiles_values.difference(table_smiles))
    sequences = table_sequences + sorted(sequences.difference(table_sequences) - {""})
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique sequences: {len(sequences)}")
    print(f"Unique SMILES: {len(smiles_values)}")

    excluded_smiles = set()
    molecules_reused = 0
    pending_molecules = []
    for smiles in smiles_values:
        if molecule_cache_path(cache_dir, smiles).exists() and not args.overwrite:
            molecules_reused += 1
        else:
            pending_molecules.append(smiles)
    proteins_reused = sum(1 for sequence in sequences if protein_cache_path(cache_dir, sequence).exists() and not args.overwrite)
    pending_sequences = [sequence for sequence in sequences if args.overwrite or not protein_cache_path(cache_dir, sequence).exists()]

    def write_in_batches(items, desc, featurize, save, skip_failures=False):
        """Featurize a batch, save the dictionaries, then write the batch files, so every file on disk has ids that the saved dictionaries contain."""
        written = 0
        for start in progress(range(0, len(items), CACHE_BATCH), desc=desc, unit="batch"):
            payloads = []
            for item in items[start : start + CACHE_BATCH]:
                try:
                    payloads.append((item, featurize(item)))
                except Exception as exc:
                    if not skip_failures or args.strict_smiles:
                        raise
                    excluded_smiles.add(item)
                    print(f"Skipping SMILES `{item}`: {exc}")
            save_dictionaries(cache_dir, dictionaries)
            for item, payload in payloads:
                save(cache_dir, item, payload)
            written += len(payloads)
        return written

    radius, ngram = int(hparams["radius"]), int(hparams["ngram"])
    molecules_written = write_in_batches(pending_molecules, "Caching DLKcat molecule features", lambda smiles: featurize_smiles(smiles, radius, dictionaries), save_molecule, skip_failures=True)
    proteins_written = write_in_batches(pending_sequences, "Caching DLKcat protein n-grams", lambda sequence: featurize_sequence(sequence, ngram, dictionaries), save_protein)
    save_dictionaries(cache_dir, dictionaries)

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
            is_train=split_name == "train",
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

    index_path = cache_dir / "split_manifest_index.csv"
    index = pd.DataFrame(split_rows)
    if index_path.exists():
        index = pd.concat([pd.read_csv(index_path), index]).drop_duplicates("manifest_path", keep="last")
    index.to_csv(index_path, index=False)
    manifest = {
        "cache_version": CACHE_VERSION,
        "table_fingerprint": table_fingerprint,
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
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
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
    cache_features(args)


if __name__ == "__main__":
    main()
