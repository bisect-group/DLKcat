import argparse
import datetime
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_RESULTS_DIRNAME,
    append_csv_row,
    hparams_cache_dir,
    load_hparams,
    metric_direction,
    regression_metrics,
    resolve_amp_dtype,
    resolve_single_split_job,
    save_json,
    set_seed,
)
from emulator_bench.dataset import DLKcatCachedDataset, dlkcat_collate
from emulator_bench.feature_pipeline import load_dictionaries, manifest_path, split_manifest_path
from emulator_bench.modeling import build_model


LOG10_OF_2 = math.log10(2.0)


def _autocast_context(device: torch.device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _monitor_value(metric_name: str, metrics: dict):
    if metric_name == "loss":
        return float(metrics["loss"])
    return float(metrics[metric_name])


def evaluate(model, loader, device, autocast_dtype, desc="eval", show_progress=False):
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    pred_log10_values = []
    true_log10_values = []
    iterator = tqdm(loader, desc=desc, leave=False, unit="batch") if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = _move_batch(batch, device)
            with _autocast_context(device, autocast_dtype):
                pred_log2 = model(batch)
                loss = loss_fn(pred_log2, batch["target_log2"])
            total_loss += float(loss.item()) * int(pred_log2.numel())
            total_count += int(pred_log2.numel())
            pred_log10_values.append((pred_log2.detach().float().cpu().numpy()) * LOG10_OF_2)
            true_log10_values.append(batch["target_log10"].detach().float().cpu().numpy())
    y_pred = np.concatenate(pred_log10_values) if pred_log10_values else np.asarray([], dtype=np.float32)
    y_true = np.concatenate(true_log10_values) if true_log10_values else np.asarray([], dtype=np.float32)
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_count)
    return y_true, y_pred, metrics


def train_one_epoch(model, loader, optimizer, device, autocast_dtype, scaler, clip_grad: float):
    model.train()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    iterator = tqdm(loader, desc="train", leave=False, unit="batch")
    for batch in iterator:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, autocast_dtype):
            pred_log2 = model(batch)
            loss = loss_fn(pred_log2, batch["target_log2"])
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()
        total_loss += float(loss.item()) * int(pred_log2.numel())
        total_count += int(pred_log2.numel())
        iterator.set_postfix(loss=f"{float(loss.item()):.4f}")
    return {"loss": total_loss / max(1, total_count)}


def _resolve_manifest_paths(args, hparams):
    cache_dir = hparams_cache_dir(Path(args.base_dir), hparams, Path(args.embeddings_dir) if args.embeddings_dir else None)
    if args.train_manifest and args.val_manifest and args.test_manifest:
        return Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest), cache_dir, None
    job = resolve_single_split_job(Path(args.base_dir), args.split_group, threshold=args.threshold)
    train_manifest = split_manifest_path(cache_dir, job["split_group"], job["split_name"], "train")
    val_manifest = split_manifest_path(cache_dir, job["split_group"], job["split_name"], "val")
    test_manifest = split_manifest_path(cache_dir, job["split_group"], job["split_name"], "test")
    return train_manifest, val_manifest, test_manifest, cache_dir, job


def _default_out_dir(args, job):
    if args.out_dir:
        return Path(args.out_dir)
    if job is None:
        raise ValueError("--out_dir is required when explicit manifest paths are used.")
    return Path(job["root_dir"]) / args.results_dirname / f"seed_{args.seed}"


def _save_predictions(path: Path, y_true, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def _save_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train DLKcat on one explicit EMULaToR train/val/test split.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--split_group", type=str, default="random_splits_grouped_sequence")
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--test_manifest", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--task_name", type=str, default="dlkcat_retrain")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--decay_interval", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--monitor_metric", choices=["rmse", "mae", "mse", "r2_score", "pearson", "spearman", "loss"], default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--cache_items", type=int, default=8192)
    args = parser.parse_args()

    hparams = load_hparams(Path(args.hparams_json))
    for key in ("batch_size", "epochs", "lr", "lr_decay", "decay_interval", "weight_decay", "monitor_metric", "val_every", "patience", "clip_grad"):
        override = getattr(args, key)
        if override is not None:
            hparams[key] = override

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

    train_manifest, val_manifest, test_manifest, cache_dir, job = _resolve_manifest_paths(args, hparams)
    if not manifest_path(cache_dir).exists():
        raise FileNotFoundError(f"Missing cache manifest {manifest_path(cache_dir)}. Run cache_embeddings.py first.")
    out_dir = _default_out_dir(args, job)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionaries = load_dictionaries(cache_dir)
    n_fingerprint = len(dictionaries["fingerprint_dict"])
    n_word = len(dictionaries["word_dict"])
    if n_fingerprint <= 0 or n_word <= 0:
        raise ValueError("DLKcat dictionaries are empty; rebuild caches.")

    train_dataset = DLKcatCachedDataset(train_manifest, cache_dir, cache_items=args.cache_items)
    val_dataset = DLKcatCachedDataset(val_manifest, cache_dir, cache_items=args.cache_items)
    test_dataset = DLKcatCachedDataset(test_manifest, cache_dir, cache_items=args.cache_items)
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) <= 0:
        raise ValueError("Train/val/test manifests must all contain at least one row after DLKcat filtering.")

    loader_kwargs = {
        "batch_size": int(hparams["batch_size"]),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory or device.type == "cuda"),
        "collate_fn": dlkcat_collate,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    eval_loader_kwargs = dict(loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)

    model = build_model(hparams, n_fingerprint=n_fingerprint, n_word=n_word).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))

    started = time.time()
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monitor_metric = str(hparams.get("monitor_metric", "rmse"))
    direction = metric_direction(monitor_metric)
    best_metric = float("inf") if direction == "minimize" else float("-inf")
    best_checkpoint_path = out_dir / "bestmodel.pth"
    best_state_dict_path = out_dir / "bestmodel_state_dict.pth"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    log_path = out_dir / "logfile.csv"
    patience = int(hparams.get("patience", 0) or 0)
    no_improve = 0

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(
            f"CUDA device: {torch.cuda.get_device_name(index)} | capability: {torch.cuda.get_device_capability(index)} | precision: {precision_mode}",
            flush=True,
        )
    else:
        print(f"Device: {device} | precision: {precision_mode}", flush=True)

    for epoch in tqdm(range(1, int(hparams["epochs"]) + 1), desc="epochs", unit="epoch"):
        if int(hparams["decay_interval"]) > 0 and epoch % int(hparams["decay_interval"]) == 0:
            optimizer.param_groups[0]["lr"] *= float(hparams["lr_decay"])
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            clip_grad=float(hparams.get("clip_grad", 0.0) or 0.0),
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "elapsed_seconds": time.time() - started,
        }

        should_validate = epoch == int(hparams["epochs"]) or (int(hparams.get("val_every", 1)) > 0 and epoch % int(hparams.get("val_every", 1)) == 0)
        if should_validate:
            _val_true, _val_pred, val_metrics = evaluate(model, val_loader, device, autocast_dtype, desc=f"val epoch {epoch}")
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            current = _monitor_value(monitor_metric, val_metrics)
            improved = current < best_metric if direction == "minimize" else current > best_metric
            if improved:
                best_metric = current
                no_improve = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_metric": best_metric,
                    "monitor_metric": monitor_metric,
                    "hparams": hparams,
                    "args": vars(args),
                    "precision_mode": precision_mode,
                    "n_fingerprint": n_fingerprint,
                    "n_word": n_word,
                }
                torch.save(checkpoint, best_checkpoint_path)
                torch.save(model.state_dict(), best_state_dict_path)
            else:
                no_improve += 1
        append_csv_row(log_path, row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_metric": best_metric,
                "monitor_metric": monitor_metric,
                "hparams": hparams,
                "args": vars(args),
                "precision_mode": precision_mode,
                "n_fingerprint": n_fingerprint,
                "n_word": n_word,
            },
            last_checkpoint_path,
        )
        if patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} after {no_improve} non-improving validation checks.", flush=True)
            break

    if not best_checkpoint_path.exists():
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_metric": best_metric}, best_checkpoint_path)
        torch.save(model.state_dict(), best_state_dict_path)
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_true, train_pred, train_final = evaluate(model, train_loader, device, autocast_dtype, desc="final train", show_progress=True)
    val_true, val_pred, val_final = evaluate(model, val_loader, device, autocast_dtype, desc="final val", show_progress=True)
    test_true, test_pred, test_final = evaluate(model, test_loader, device, autocast_dtype, desc="final test", show_progress=True)
    _save_predictions(out_dir / "pred_label_train.csv", train_true, train_pred)
    _save_predictions(out_dir / "pred_label_val.csv", val_true, val_pred)
    _save_predictions(out_dir / "pred_label_test.csv", test_true, test_pred)
    _save_metrics(out_dir / "final_results_train.csv", train_final)
    _save_metrics(out_dir / "final_results_val.csv", val_final)
    _save_metrics(out_dir / "final_results_test.csv", test_final)

    summary = {
        "task_name": args.task_name,
        "started_at": started_at,
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "precision_mode": precision_mode,
        "best_epoch": int(best_checkpoint.get("epoch", epoch)),
        "monitor_metric": monitor_metric,
        "best_val_metric": float(best_checkpoint.get("best_val_metric", best_metric)),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "test_manifest": str(test_manifest),
        "cache_dir": str(cache_dir),
        "final_train_metrics": train_final,
        "final_val_metrics": val_final,
        "final_test_metrics": test_final,
        "hparams": hparams,
        "args": vars(args),
    }
    save_json(out_dir / "run_summary.json", summary)
    pd.DataFrame([{key: value for key, value in summary.items() if key not in {"final_train_metrics", "final_val_metrics", "final_test_metrics", "hparams", "args"}}]).to_csv(
        out_dir / "run_summary.csv",
        index=False,
    )
    print(f"Saved run outputs to {out_dir}")


if __name__ == "__main__":
    main()
