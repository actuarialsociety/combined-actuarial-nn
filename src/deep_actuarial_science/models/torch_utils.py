"""Shared PyTorch helpers: seeding, a Poisson training loop with early stopping."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
) -> dict:
    """Minimise the Poisson NLL of `model(X, offset)` (expected counts).

    train/val are (X, y, offset, exposure) tensors. Keeps the weights with the
    lowest validation loss (early stopping by checkpoint), as in the reference
    notebooks. Returns the loss history per epoch (per unit of exposure).
    """
    X, y, off, w = train
    Xv, yv, offv, wv = val
    opt = torch.optim.NAdam(model.parameters(), **({"lr": lr} if lr else {}))
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history = {"loss": [], "val_loss": []}
    n = len(X)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n) if shuffle else torch.arange(n)
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
            best, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) if epochs else -1
    return history


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
