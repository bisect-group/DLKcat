"""Batch-size study for DLKcat: the same network and settings as production, trained with batch size B.

Production (train_single_target_tvt.py) takes one Adam step per sample on the loss (pred - log2 kcat)^2.
Here one Adam step uses the mean of that loss over B samples, so B = 1 is the production loss. Graphs and sequences are
padded to the longest in the batch. Padded atoms and words are masked, so each sample sees the values of the unpadded
network. Learning rate, weight decay, decay schedule, epoch count, fp32 and the last-epoch model are unchanged.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HPARAMS_PATH,
    append_csv_row,
    hparams_cache_dir,
    load_hparams,
    regression_metrics,
    save_json,
    set_seed,
)
from emulator_bench.feature_pipeline import load_dictionaries, load_molecule, load_protein  # noqa: E402
from emulator_bench.modeling import KcatPrediction  # noqa: E402
from emulator_bench.train_single_target_tvt import LOG10_OF_2, _save_metrics, _save_predictions  # noqa: E402


class BatchedKcatPrediction(KcatPrediction):
    """KcatPrediction with the same parameters, computed for a padded batch.

    A batch of one sample is unpadded, so it runs the original per-sample forward. Batched kernels (bmm, sum) pick other
    float summation orders for small shapes and would not match the original bitwise.
    """

    def forward_batch(self, fingerprints, adjacency, n_atoms, words, n_words):
        if len(fingerprints) == 1:
            return self(fingerprints[0], adjacency[0], words[0])
        atom_mask = (torch.arange(fingerprints.shape[1], device=fingerprints.device)[None] < n_atoms[:, None]).unsqueeze(-1).float()
        word_mask = (torch.arange(words.shape[1], device=words.device)[None] < n_words[:, None]).unsqueeze(-1).float()
        xs = self.embed_fingerprint(fingerprints)
        for layer in self.W_gnn:
            xs = xs + torch.matmul(adjacency, torch.relu(layer(xs)))
        compound = (xs * atom_mask).sum(1) / n_atoms[:, None]
        xs = (self.embed_word(words) * word_mask).unsqueeze(1)
        for layer in self.W_cnn:
            xs = torch.relu(layer(xs)) * word_mask.unsqueeze(1)
        xs = xs.squeeze(1)
        h = torch.relu(self.W_attention(compound))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(torch.matmul(hs, h.unsqueeze(-1)))
        protein = (weights * hs * word_mask).sum(1) / n_words[:, None]
        out = torch.cat((compound, protein), 1)
        for layer in self.W_out:
            out = torch.relu(layer(out))
        return self.W_interaction(out)


def build_batched_model(hparams, n_fingerprint: int, n_word: int) -> BatchedKcatPrediction:
    return BatchedKcatPrediction(
        n_fingerprint, n_word, *(int(hparams[key]) for key in ("dim", "layer_gnn", "window", "layer_cnn", "layer_output"))
    )


class BatchedSplit:
    """One split on the device: tensors per unique molecule and sequence, and row indices into them."""

    def __init__(self, manifest_path: Path, cache_dir: Path, device: torch.device):
        frame = pd.read_csv(manifest_path, usecols=["smiles", "sequence", "target_log2", "target_log10"])
        smiles, mol_row = np.unique(frame["smiles"].to_numpy(), return_inverse=True)
        sequences, prot_row = np.unique(frame["sequence"].to_numpy(), return_inverse=True)
        molecules = [load_molecule(cache_dir, s) for s in smiles]
        proteins = [load_protein(cache_dir, q)["words"] for q in sequences]
        self.n_atoms = torch.tensor([len(m["fingerprints"]) for m in molecules], device=device)
        self.n_words = torch.tensor([len(p) for p in proteins], device=device)
        self.fingerprints = torch.zeros(len(molecules), int(self.n_atoms.max()), dtype=torch.long)
        for i, m in enumerate(molecules):
            self.fingerprints[i, : len(m["fingerprints"])] = torch.from_numpy(m["fingerprints"].astype(np.int64))
        # Adjacency matrices are stored as one flat array of n x n blocks. The last element is a zero for padding.
        blocks = [torch.from_numpy(m["adjacency"].astype(np.float32)).reshape(-1) for m in molecules]
        self.adjacency = torch.cat([*blocks, torch.zeros(1)]).to(device)
        self.adjacency_offset = torch.tensor(np.cumsum([0] + [len(b) for b in blocks[:-1]]), device=device)
        self.words = torch.zeros(len(proteins), int(self.n_words.max()), dtype=torch.long)
        for i, p in enumerate(proteins):
            self.words[i, : len(p)] = torch.from_numpy(p.astype(np.int64))
        self.fingerprints, self.words = self.fingerprints.to(device), self.words.to(device)
        self.mol_row, self.prot_row = torch.from_numpy(mol_row).to(device), torch.from_numpy(prot_row).to(device)
        self.target = torch.tensor(frame["target_log2"].to_numpy(np.float32), device=device).view(-1, 1)
        self.target_log10 = frame["target_log10"].to_numpy(np.float32)

    def __len__(self):
        return len(self.target)

    def batch(self, rows: torch.Tensor):
        mol, prot = self.mol_row[rows], self.prot_row[rows]
        n_atoms, n_words = self.n_atoms[mol], self.n_words[prot]
        a, w = int(n_atoms.max()), int(n_words.max())
        n = n_atoms[:, None, None]
        row, col = torch.arange(a, device=n.device)[None, :, None], torch.arange(a, device=n.device)[None, None]
        index = torch.where((row < n) & (col < n), self.adjacency_offset[mol][:, None, None] + row * n + col, len(self.adjacency) - 1)
        return self.fingerprints[mol, :a], self.adjacency[index], n_atoms, self.words[prot, :w], n_words


def evaluate(model, split: BatchedSplit, batch_size: int):
    """Predictions in row order. Rows are sorted by sequence length so padding stays small."""
    model.eval()
    order = torch.argsort(split.n_words[split.prot_row])
    preds = torch.empty(len(split), device=order.device)
    with torch.no_grad():
        for start in range(0, len(split), batch_size):
            rows = order[start : start + batch_size]
            preds[rows] = model.forward_batch(*split.batch(rows)).view(-1)
    loss = float(F.mse_loss(preds, split.target.view(-1), reduction="sum")) / len(split)
    y_pred = preds.cpu().numpy() * LOG10_OF_2
    metrics = regression_metrics(split.target_log10, y_pred)
    metrics["loss"] = loss
    return split.target_log10, y_pred, metrics


def train_one_epoch(model, split: BatchedSplit, order: np.ndarray, optimizer, batch_size: int) -> float:
    """Mean per-sample loss over the epoch. One Adam step per batch on the batch-mean loss."""
    model.train()
    order = torch.from_numpy(order).to(split.target.device)
    total = torch.zeros((), device=order.device)
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model.forward_batch(*split.batch(rows)), split.target[rows])
        loss.backward()
        optimizer.step()
        total += loss.detach() * len(rows)
    return float(total) / len(order)


def main():
    parser = argparse.ArgumentParser(description="Train DLKcat with batch size B on one split.")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--hparams_json", default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test override of the epoch count.")
    args = parser.parse_args()

    hparams = load_hparams(Path(args.hparams_json))
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = hparams_cache_dir(hparams, args.embeddings_dir, args.base_dir)
    manifests = {name: Path(getattr(args, f"{name}_manifest")) for name in ("train", "val", "test")}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint_last.pt"
    if not checkpoint.exists():
        (out_dir / "logfile.csv").unlink(missing_ok=True)

    dictionaries = load_dictionaries(cache_dir)
    splits = {name: BatchedSplit(path, cache_dir, device) for name, path in manifests.items()}
    model = build_batched_model(hparams, len(dictionaries["fingerprint_dict"]), len(dictionaries["word_dict"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))
    train_order = np.arange(len(splits["train"]))

    start_epoch, elapsed_before = 1, 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        train_order[:] = state["train_order"]
        np.random.set_state(state["np_random_state"])
        start_epoch, elapsed_before = state["epoch"] + 1, state["elapsed"]

    started = time.time() - elapsed_before
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for epoch in range(start_epoch, int(hparams["epochs"]) + 1):
        if epoch % int(hparams["decay_interval"]) == 0:
            optimizer.param_groups[0]["lr"] *= float(hparams["lr_decay"])
        np.random.shuffle(train_order)
        train_loss = train_one_epoch(model, splits["train"], train_order, optimizer, args.batch_size)
        _, _, val_metrics = evaluate(model, splits["val"], 256)
        append_csv_row(
            out_dir / "logfile.csv",
            {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
             **{f"val_{key}": value for key, value in val_metrics.items()}, "elapsed_seconds": time.time() - started},
        )
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_order": train_order,
                 "np_random_state": np.random.get_state(), "elapsed": time.time() - started}
        torch.save(state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
    torch.save(model.state_dict(), out_dir / "lastmodel.pth")

    final = {}
    for name in ("train", "val", "test"):
        y_true, y_pred, metrics = evaluate(model, splits[name], 256)
        final[name] = metrics
        _save_predictions(out_dir / f"pred_label_{name}.csv", y_true, y_pred)
        _save_metrics(out_dir / f"final_results_{name}.csv", metrics)
    save_json(
        out_dir / "run_summary.json",
        {"batch_size": args.batch_size, "seed": args.seed, "started_at": started_at,
         "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": time.time() - started,
         "epochs": int(hparams["epochs"]), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
         "rows": {name: len(split) for name, split in splits.items()},
         **{f"final_{name}_metrics": metrics for name, metrics in final.items()}, "hparams": hparams},
    )
    print(f"Saved run outputs to {out_dir}")


if __name__ == "__main__":
    main()
