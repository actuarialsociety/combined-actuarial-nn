"""Shared PyTorch helpers: seeding, device choice, a Poisson training loop with early stopping."""

from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    """Training device: `DAS_DEVICE` if set (cpu, cuda, mps), else CUDA when available.

    Models train there and are moved back to the CPU afterwards, so prediction, metrics and
    figures run unchanged. GPU and CPU results differ in the last digits, which the
    network tolerances in conf/base/parameters_benchmarks.yml cover."""
    name = os.environ.get("DAS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def train_poisson(
    model: nn.Module,
    train: tuple,
    val: tuple,
    *,
    epochs: int,
    batch_size: int,
    lr: float | None = None,
    shuffle: bool = False,
    patience: int | None = None,
) -> dict:
    """Minimise the Poisson NLL of `model(X, offset)` (expected counts).

    train/val are (X, y, offset, exposure) tensors. Keeps the weights with the
    lowest validation loss (early stopping by checkpoint). With `patience`, training
    also stops once the validation loss has not improved for that many epochs
    (Keras `EarlyStopping(patience=..., restore_best_weights=True)`). Returns the loss
    history per epoch (per unit of exposure).
    """
    dev = device()
    model.to(dev)
    X, y, off, w = (t.to(dev) for t in train)
    Xv, yv, offv, wv = (t.to(dev) for t in val)
    opt = torch.optim.NAdam(model.parameters(), **({"lr": lr} if lr else {}))
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    best, best_state, since = float("inf"), copy.deepcopy(model.state_dict()), 0
    history = {"loss": [], "val_loss": []}
    n = len(X)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
        total = 0.0
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            loss = loss_fn(model(X[b], off[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv, offv), yv).item() / float(wv.sum())
        history["loss"].append(total / float(w.sum()))
        history["val_loss"].append(val_loss)
        if val_loss < best:
            best, best_state, since = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if patience is not None and since >= patience:
                break
    model.load_state_dict(best_state)
    model.cpu()
    history["best_epoch"] = int(np.argmin(history["val_loss"])) if epochs else -1
    return history


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
