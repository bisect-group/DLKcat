# ruff: noqa: F811
"""The batch-size study code against the original per-sample DLKcat code (CPU, fp32, 64/16/16 fixture rows, seed 666).

B = 1 must equal the original training bitwise. For B > 1 the padded forward pass must equal the per-sample forward
pass, and the gradient of the batch-mean loss must equal the mean of the per-sample gradients, up to float noise.
"""

import numpy as np
import torch
import torch.nn.functional as F

from emulator_bench import batched_study
from emulator_bench.feature_pipeline import load_dictionaries
from test_original_equivalence import EPOCHS, HPARAMS, SEED, SPLIT, original_run, wrapper_cache  # noqa: F401, F811


def load(wrapper_cache):
    _base, cache_dir = wrapper_cache
    dictionaries = load_dictionaries(cache_dir)
    split = batched_study.BatchedSplit(cache_dir / "split_manifests" / SPLIT / "all" / "train.csv", cache_dir, torch.device("cpu"))
    torch.manual_seed(SEED)
    model = batched_study.build_batched_model(HPARAMS, len(dictionaries["fingerprint_dict"]), len(dictionaries["word_dict"]))
    return model, split


def test_batch_size_one_matches_original(wrapper_cache, original_run):
    model, split = load(wrapper_cache)
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
    order = np.arange(len(split))
    np.random.seed(SEED)
    losses = []
    for epoch in range(1, EPOCHS + 1):
        if epoch % HPARAMS["decay_interval"] == 0:
            optimizer.param_groups[0]["lr"] *= HPARAMS["lr_decay"]
        np.random.shuffle(order)
        losses.append(batched_study.train_one_epoch(model, split, order, optimizer, 1))
    assert np.allclose(losses, [row["loss_sum"] / original_run["n_train"] for row in original_run["log"]], rtol=1e-6, atol=0)
    assert all(torch.equal(value, original_run["model"].state_dict()[key]) for key, value in model.state_dict().items())


def test_padded_batch_matches_per_sample(wrapper_cache):
    model, split = load(wrapper_cache)
    rows = torch.arange(8)
    target = split.target[rows]
    batch_loss = F.mse_loss(model.forward_batch(*split.batch(rows)), target)
    batch_loss.backward()
    batch_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for row in rows:
        loss = F.mse_loss(model.forward_batch(*split.batch(row.view(1))), split.target[row.view(1)]) / len(rows)
        loss.backward()
    assert all(torch.allclose(a, p.grad, rtol=1e-4, atol=1e-6) for a, p in zip(batch_grads, model.parameters()))
    assert split.n_atoms[split.mol_row[rows]].unique().numel() > 1 and split.n_words[split.prot_row[rows]].unique().numel() > 1
