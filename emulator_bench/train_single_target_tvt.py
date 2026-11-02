import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    resolve_amp_dtype,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import PreloadedSplit
from emulator_bench.feature_pipeline import load_dictionaries, manifest_path, split_manifest_path
from emulator_bench.modeling import build_model

LOG10_OF_2 = math.log10(2.0)


def _autocast(device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def evaluate(model, split, device, amp_dtype=None):
    model.eval()
    with torch.no_grad():
        preds = []
        for fingerprints, adjacency, words, _target in split.samples:
            with _autocast(device, amp_dtype):
                preds.append(model(fingerprints, adjacency, words))
    pred_log2 = torch.cat(preds).view(-1).float()
    target_log2 = torch.cat([sample[3] for sample in split.samples]).view(-1)
    loss = float(torch.nn.functional.mse_loss(pred_log2, target_log2, reduction="sum")) / len(split)
    y_pred, y_true = pred_log2.cpu().numpy() * LOG10_OF_2, split.target_log10
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = loss
    return y_true, y_pred, metrics


def train_one_epoch(model, split, order, optimizer, device, amp_dtype=None, scaler=None):
    model.train()
    total_loss = torch.zeros((), device=device)
    for index in order:
        fingerprints, adjacency, words, target = split.samples[index]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            pred_log2 = model(fingerprints, adjacency, words)
        loss = torch.nn.functional.mse_loss(pred_log2.float(), target)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.detach()
    return {"loss": float(total_loss) / len(order)}


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
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision (bf16 on Ampere or newer, else fp16). The original code trains in fp32.")
    args = parser.parse_args()

    hparams = load_hparams(Path(args.hparams_json))
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype, precision = resolve_amp_dtype(device) if args.amp else (None, "fp32")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    cache_dir = hparams_cache_dir(hparams, args.embeddings_dir, args.base_dir)
    if not manifest_path(cache_dir).exists():
        raise FileNotFoundError(f"Missing cache manifest {manifest_path(cache_dir)}. Run cache_embeddings.py first.")
    train_manifest, val_manifest, test_manifest, out_dir = _resolve_manifest_paths(args, cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint_last.pt"
    if not checkpoint.exists():
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
    store = {}
    splits = {name: PreloadedSplit(path, cache_dir, device, store) for name, path in manifests.items()}
    if min(len(split) for split in splits.values()) == 0:
        raise ValueError("Train/val/test manifests must all contain at least one row after DLKcat filtering.")
    if int(hparams["batch_size"]) != 1:
        raise ValueError("DLKcat trains with batch size 1.")

    # The original shuffles the train list in place with np.random each epoch.
    train_order = np.arange(len(splits["train"]))
    model = build_model(hparams, n_fingerprint=n_fingerprint, n_word=n_word).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))

    start_epoch, elapsed_before = 1, 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        train_order[:] = state["train_order"]
        np.random.set_state(state["np_random_state"])
        start_epoch, elapsed_before = state["epoch"] + 1, state["elapsed"]
        print(f"Resuming from epoch {start_epoch}", flush=True)

    started = time.time() - elapsed_before
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Device: {device} | precision: {precision}", flush=True)

    for epoch in progress(range(start_epoch, int(hparams["epochs"]) + 1), desc="epochs", unit="epoch"):
        if epoch % int(hparams["decay_interval"]) == 0:
            optimizer.param_groups[0]["lr"] *= float(hparams["lr_decay"])
        np.random.shuffle(train_order)
        train_metrics = train_one_epoch(model, splits["train"], train_order, optimizer, device, amp_dtype, scaler)
        _, _, val_metrics = evaluate(model, splits["val"], device, amp_dtype)
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
        torch.save(model.state_dict(), out_dir / "lastmodel.pth")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_order": train_order,
                 "np_random_state": np.random.get_state(), "elapsed": time.time() - started}
        torch.save(state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)

    final = {}
    for name in ("train", "val", "test"):
        y_true, y_pred, metrics = evaluate(model, splits[name], device, amp_dtype)
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
