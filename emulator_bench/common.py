import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "DLKcat"
DEFAULT_BASE_DIR = Path(f"/home/adhil/github/EMULaToR/data/processed/baselines/{DEFAULT_MODEL_NAME}")
DEFAULT_HPARAMS_PATH = REPO_ROOT / "emulator_bench" / "default_hparams_paper.json"
DEFAULT_RESULTS_DIRNAME = "dlkcat_results"
DEFAULT_OPTUNA_RESULTS_DIRNAME = "dlkcat_optuna_runs"
DEFAULT_RETRAIN_DIRNAME = "retrain_paper_settings"
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "substrate_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "conformer_cosine_splits",
]

MINIMIZE_METRICS = {"rmse", "mse", "mae", "loss"}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def load_hparams(path: Optional[Path] = None) -> Dict:
    return load_json(Path(path or DEFAULT_HPARAMS_PATH))


def hparams_cache_dir(base_dir: Path, hparams: Dict, embeddings_dir: Optional[Path] = None) -> Path:
    root = Path(embeddings_dir) if embeddings_dir is not None else Path(base_dir) / "embeddings"
    return root / f"dlkcat_radius{int(hparams['radius'])}_ngram{int(hparams['ngram'])}"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _is_grouped_random_split(split_group: str) -> bool:
    return split_group.startswith("random_splits_grouped_")


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels_for_thresholds(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {ordered[0]: "hard", ordered[1]: "easy"}
    if len(ordered) == 3:
        return {ordered[0]: "hard", ordered[1]: "medium", ordered[2]: "easy"}
    return {name: f"rank_{idx}" for idx, name in enumerate(ordered, start=1)}


def normalize_threshold_args(thresholds: Optional[Iterable[str]] = None, threshold: Optional[str] = None) -> Optional[List[str]]:
    values: List[str] = []
    if thresholds is not None:
        values.extend(str(value) for value in thresholds if str(value).strip())
    if threshold is not None and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    split_groups = list(split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = set(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []
    for split_group in split_groups:
        group_dir = Path(base_dir) / split_group
        if not group_dir.exists():
            continue

        train_path = _find_split_file(group_dir, "train")
        val_path = _find_split_file(group_dir, "val")
        test_path = _find_split_file(group_dir, "test")
        if train_path and val_path and test_path:
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": split_group,
                    "difficulty": split_group,
                    "root_dir": str(group_dir),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
            continue

        candidate_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            if threshold_filter is not None and child.name not in threshold_filter:
                continue
            if child.name.startswith("threshold_") or child.name in {"easy", "medium", "hard"}:
                candidate_dirs.append(child)
        threshold_names = [child.name for child in candidate_dirs if child.name.startswith("threshold_")]
        threshold_difficulties = _difficulty_labels_for_thresholds(threshold_names)
        for child in candidate_dirs:
            train_path = _find_split_file(child, "train")
            val_path = _find_split_file(child, "val")
            test_path = _find_split_file(child, "test")
            if not (train_path and val_path and test_path):
                continue
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": threshold_difficulties.get(child.name, child.name),
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    no_threshold = _is_grouped_random_split(split_group) or split_group == "uniprot_time_splits"
    threshold_filter = None if no_threshold else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = f"{split_group}/{threshold}" if threshold else split_group
        raise FileNotFoundError(f"No split job discovered for {detail} in {base_dir}")
    if no_threshold:
        return jobs[0]
    if threshold is None and len(jobs) > 1:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError(f"Multiple threshold jobs found for {split_group}. Specify --threshold. Available: {available}")
    return jobs[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None, "fp32"
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def metric_direction(metric: str) -> str:
    return "minimize" if metric in MINIMIZE_METRICS else "maximize"


def regression_metrics(y_true_log10, y_pred_log10) -> Dict[str, float]:
    y_true = np.asarray(y_true_log10, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred_log10, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "r2_score": math.nan, "pearson": math.nan, "spearman": math.nan}
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    pearson = 0.0 if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0 else float(np.corrcoef(y_true, y_pred)[0, 1])
    try:
        from scipy import stats

        spearman_value = float(stats.spearmanr(y_true, y_pred).statistic)
        spearman = 0.0 if math.isnan(spearman_value) else spearman_value
    except Exception:
        true_ranks = np.argsort(np.argsort(y_true))
        pred_ranks = np.argsort(np.argsort(y_pred))
        spearman = 0.0 if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0 else float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2, "pearson": pearson, "spearman": spearman}


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_seed_runs(rows: List[Dict], group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [col for col in metric_cols if col in df.columns]
    if not metric_cols:
        return df[group_cols].drop_duplicates() if all(col in df.columns for col in group_cols) else df
    return df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
