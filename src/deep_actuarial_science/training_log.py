"""Per-epoch training curves of every trained model, for MLflow and W&B.

Each model's `fit` node returns its training history in its own shape: a frequency model
object (`histories`, one per seed, or one `history`), mortality's `{sex: {..., "runs": [...]}}`,
reserving's single history. `curves` turns any of them into one long table

    group  run  epoch  metric    value
    Male   3    41     val_loss  0.00021

(`group` is the sex for mortality, "" elsewhere; `run` is the seed index), and the tracking
hooks (`hooks.py`, `wandb_hook.py`) log it after the node runs: every seed's training and
validation loss, and the learning rate where it changes, per epoch.
"""

from __future__ import annotations

import pandas as pd

CURVES = ("loss", "val_loss", "lr", "refit_loss")
COLUMNS = ["group", "run", "epoch", "metric", "value"]


def _one(hist: dict, group: str, run: int) -> list[dict]:
    rows = []
    for metric in CURVES:
        for epoch, value in enumerate(hist.get(metric) or []):
            rows.append(
                {"group": group, "run": run, "epoch": epoch, "metric": metric, "value": value}
            )
    return rows


def _histories(hist: dict) -> list[dict]:
    """A history dict holding every seed (`runs`) or a single run."""
    return list(hist["runs"]) if isinstance(hist.get("runs"), list) else [hist]


def curves(obj) -> pd.DataFrame:
    """Long table of per-epoch curves from a fit node's output, empty if there are none."""
    rows: list[dict] = []
    if hasattr(obj, "histories") and obj.histories:  # multi-seed frequency network
        for r, h in enumerate(obj.histories):
            rows += _one(h, "", r)
    elif hasattr(obj, "history") and isinstance(obj.history, dict):  # GBM, single fits
        rows += _one(obj.history, "", 0)
    elif isinstance(obj, dict) and any(k in obj for k in CURVES):  # reserving
        for r, h in enumerate(_histories(obj)):
            rows += _one(h, "", r)
    elif isinstance(obj, dict):  # mortality: one history per sex
        for group, hist in obj.items():
            if isinstance(hist, dict):
                for r, h in enumerate(_histories(hist)):
                    rows += _one(h, str(group), r)
    return pd.DataFrame(rows, columns=COLUMNS)


def fit_outputs(node, outputs: dict) -> list[tuple[str, pd.DataFrame]]:
    """(model namespace, curves) for each output of a `<model>.fit` node that has any."""
    if not node.name.endswith(".fit") or not node.namespace:
        return []
    found = []
    for value in outputs.values():
        df = curves(value)
        if len(df):
            found.append((node.namespace, df))
    return found


def series_key(ns: str, group: str, run: int, metric: str) -> str:
    """Metric name of one curve, e.g. `lstm/train/Male/seed_03/val_loss`."""
    parts = [ns, "train", group, f"seed_{run:02d}", metric]
    return "/".join(p for p in parts if p)
