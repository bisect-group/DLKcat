import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_OUTPUT_ROOT,
    append_csv_row,
    hparams_cache_dir,
    load_hparams,
    regression_metrics,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import DLKcatCachedDataset, dlkcat_collate
from emulator_bench.feature_pipeline import load_dictionaries, manifest_path, split_manifest_path
from emulator_bench.modeling import build_model

LOG10_OF_2 = math.log10(2.0)


def _move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model, loader, device, desc="eval", show_progress=False):
    model.eval()
    total_loss = 0.0
    pred_log10, true_log10 = [], []
    iterator = progress(loader, desc=desc, leave=False, unit="batch") if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = _move_batch(batch, device)
            pred_log2 = model(batch)
            total_loss += float(torch.nn.functional.mse_loss(pred_log2, batch["target_log2"], reduction="sum"))
            pred_log10.append(pred_log2.cpu().numpy() * LOG10_OF_2)
            true_log10.append(batch["target_log10"].cpu().numpy())
    y_pred, y_true = np.concatenate(pred_log10), np.concatenate(true_log10)
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return y_true, y_pred, metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in progress(loader, desc="train", leave=False, unit="batch"):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), batch["target_log2"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch["target_log2"])
        total_count += len(batch["target_log2"])
    return {"loss": total_loss / total_count}


def _resolve_manifest_paths(args, cache_dir):
    if args.train_manifest and args.val_manifest and args.test_manifest:
        if not args.out_dir:
            raise ValueError("--out_dir is required when explicit manifest paths are used.")
        return Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest), Path(args.out_dir)
    job = resolve_single_split_job(Path(args.base_dir), args.split_group, threshold=args.threshold)
    manifests = [split_manifest_path(cache_dir, job["split_group"], job["split_name"], split) for split in ("train", "val", "test")]
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / job["split_group"] / job["split_name"] / f"seed_{args.seed}"
    return *manifests, out_dir


def _save_predictions(path: Path, y_true, y_pred):
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def _save_metrics(path: Path, metrics: dict):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train DLKcat with its original settings on one explicit EMULaToR train/val/test split.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_group", type=str, default="random_splits_grouped_sequence")
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--test_manifest", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="dlkcat")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test override of the original epoch count.")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    hparams = load_hparams(Path(args.hparams_json))
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device(args.device)

    cache_dir = hparams_cache_dir(hparams, args.embeddings_dir, args.base_dir)
    if not manifest_path(cache_dir).exists():
        raise FileNotFoundError(f"Missing cache manifest {manifest_path(cache_dir)}. Run cache_embeddings.py first.")
    train_manifest, val_manifest, test_manifest, out_dir = _resolve_manifest_paths(args, cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logfile.csv").unlink(missing_ok=True)

    index = pd.read_csv(cache_dir / "split_manifest_index.csv")
    index = index.set_index(index["manifest_path"].map(lambda value: str(Path(value).resolve())))
    manifests = {"train": train_manifest, "val": val_manifest, "test": test_manifest}
    missing = [str(path) for path in manifests.values() if str(path.resolve()) not in index.index]
    if missing:
        raise KeyError(f"Splits missing from {cache_dir / 'split_manifest_index.csv'}; run cache_embeddings.py for them: {missing}")
    rows_dropped = {name: int(index.loc[str(path.resolve()), "skipped_rows"]) for name, path in manifests.items()}

    dictionaries = load_dictionaries(cache_dir)
    n_fingerprint = len(dictionaries["fingerprint_dict"])
    n_word = len(dictionaries["word_dict"])
    datasets = {name: DLKcatCachedDataset(path, cache_dir) for name, path in (("train", train_manifest), ("val", val_manifest), ("test", test_manifest))}
    if min(len(dataset) for dataset in datasets.values()) == 0:
        raise ValueError("Train/val/test manifests must all contain at least one row after DLKcat filtering.")

    loader_kwargs = {"batch_size": int(hparams["batch_size"]), "num_workers": args.num_workers, "collate_fn": dlkcat_collate}
    # The original shuffles the train list in place with np.random each epoch; the loader reads this order.
    train_order = np.arange(len(datasets["train"]))
    train_loader = DataLoader(datasets["train"], sampler=train_order, **loader_kwargs)
    loaders = {name: DataLoader(dataset, shuffle=False, **loader_kwargs) for name, dataset in datasets.items()}

    model = build_model(hparams, n_fingerprint=n_fingerprint, n_word=n_word).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))

    started = time.time()
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Device: {device} | precision: fp32", flush=True)

    for epoch in progress(range(1, int(hparams["epochs"]) + 1), desc="epochs", unit="epoch"):
        if epoch % int(hparams["decay_interval"]) == 0:
            optimizer.param_groups[0]["lr"] *= float(hparams["lr_decay"])
        np.random.shuffle(train_order)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        _, _, val_metrics = evaluate(model, loaders["val"], device)
        append_csv_row(
            out_dir / "logfile.csv",
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "elapsed_seconds": time.time() - started,
            },
        )
        torch.save(model.state_dict(), out_dir / "model_state_dict.pth")

    final = {}
    for name in ("train", "val", "test"):
        y_true, y_pred, metrics = evaluate(model, loaders[name], device, desc=f"final {name}", show_progress=True)
        final[name] = metrics
        _save_predictions(out_dir / f"pred_label_{name}.csv", y_true, y_pred)
        _save_metrics(out_dir / f"final_results_{name}.csv", metrics)

    save_json(
        out_dir / "run_summary.json",
        {
            "task_name": args.task_name,
            "started_at": started_at,
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started,
            "epochs": int(hparams["epochs"]),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_manifest": str(test_manifest),
            "cache_dir": str(cache_dir),
            "rows_dropped": rows_dropped,
            "final_train_metrics": final["train"],
            "final_val_metrics": final["val"],
            "final_test_metrics": final["test"],
            "hparams": hparams,
            "args": vars(args),
        },
    )
    print(f"Saved run outputs to {out_dir}")


if __name__ == "__main__":
    main()
