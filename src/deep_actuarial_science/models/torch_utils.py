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

    Apple-silicon GPUs (`DAS_DEVICE=mps`) are supported but opt-in: these networks are
    small (hundreds to a few thousand weights, batches of 4 to 7,000 rows), so kernel-launch
    overhead dominates and on an M1 MPS trained 1.0-2.5x slower than the CPU (see README).
    All tensors are float32, which MPS requires.

    Models train there and are moved back to the CPU afterwards, so prediction, metrics and
    figures run unchanged. GPU and CPU results differ in the last digits, which the
    network tolerances in conf/base/parameters_benchmarks.yml cover."""
    name = os.environ.get("DAS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DAS_DEVICE=mps, but this PyTorch build or machine has no MPS")
    return torch.device(name)


# Keras optimiser epsilon (PyTorch defaults to 1e-8). The benchmark papers train in Keras.
KERAS_EPS = 1e-7


def keras_init(module: nn.Module) -> nn.Module:
    """Re-initialise every layer with Keras's defaults, as in the benchmark papers (Keras/R
    keras): Glorot-uniform kernels and zero biases for dense and convolutional layers (PyTorch
    uses Kaiming-uniform), uniform(-0.05, 0.05) embeddings (PyTorch: standard normal).
    Layers with a `keras_init` method (the Keras-style LSTM) set their own extras. Call it
    before any model-specific initialisation, such as zero output weights."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
    # After the generic pass: modules() visits a layer before its sub-layers, so the generic
    # pass would otherwise overwrite these (e.g. the LSTM's forget-gate bias).
    for m in module.modules():
        if m is not module and hasattr(m, "keras_init"):
            m.keras_init()
    return module


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
    opt = torch.optim.NAdam(model.parameters(), lr=lr or 0.001, eps=KERAS_EPS)
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    best, best_state, since = float("inf"), copy.deepcopy(model.state_dict()), 0
    history = {"loss": [], "val_loss": []}
    n = len(X)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
        total = torch.zeros((), device=dev)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            loss = loss_fn(model(X[b], off[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()  # no per-batch .item(): a GPU sync on every step
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv, offv), yv).item() / float(wv.sum())
        history["loss"].append(total.item() / float(w.sum()))
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
