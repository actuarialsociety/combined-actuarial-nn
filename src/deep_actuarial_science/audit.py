"""Audit trail: what each pipeline step received and produced, for review and reconciliation.

For every node of a run, `AuditTrail` records how long it took and, for each data frame it
read or wrote, the row count and the totals of the columns an actuary reconciles between steps:
claim counts, exposure, predicted claims, premium, paid losses and reserves. A total that
changes where it should not (rows dropped in a join, exposure lost in a split) shows up as a
break between consecutive steps.

The tracking hooks (`hooks.py`, `wandb_hook.py`) log each step as it finishes and the whole
table at the end of the run (`audit/steps.csv`). `version()` is the model version every
report states: the package version plus the git commit it ran from.
"""

from __future__ import annotations

import subprocess
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

# Columns whose totals are reconciled between steps (any frame that has them).
TOTALS = {
    "ClaimNb": "claims",  # frequency: observed claim counts
    "Exposure": "exposure",  # frequency: years at risk
    "pred": "predicted",  # every domain: model prediction (claims, log m, incremental paid)
    "premium": "premium",  # reserving: earned premium
    "incr_paid": "paid",  # reserving: incremental paid losses
    "predicted_reserve": "reserve_predicted",
    "actual_reserve": "reserve_actual",
}


def version() -> dict[str, str]:
    """Package version (from pyproject.toml, via the installed metadata) and git commit."""
    try:
        pkg = metadata.version("deep-actuarial-science")
    except metadata.PackageNotFoundError:
        pkg = "unknown"
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True
    )
    return {"package_version": pkg, "git_commit": out.stdout.strip() or "unknown"}


def frame_totals(df: pd.DataFrame) -> dict[str, float]:
    """Row count and the reconciled totals of one data frame."""
    out = {"rows": float(len(df))}
    for col, name in TOTALS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out[name] = float(df[col].sum())
    return out


class AuditTrail:
    """Collects one row per (node, dataset) for a run; `start`/`finish` wrap each node."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._t0: dict[str, float] = {}

    def start(self, node_name: str) -> None:
        self._t0[node_name] = time.perf_counter()

    def finish(self, node_name: str, inputs: dict, outputs: dict) -> list[dict]:
        """Record the node's duration and the totals of every frame in or out; return the
        new rows (one per frame, or one with only the duration if no frame passed)."""
        seconds = time.perf_counter() - self._t0.pop(node_name, time.perf_counter())
        new = []
        for side, data in (("in", inputs), ("out", outputs)):
            for name, value in data.items():
                if isinstance(value, pd.DataFrame):
                    new.append(
                        {"node": node_name, "side": side, "dataset": name, **frame_totals(value)}
                    )
        if not new:
            new.append({"node": node_name, "side": "", "dataset": ""})
        for row in new:
            row["seconds"] = seconds
        self.rows += new
        return new

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def metric_key(row: dict, field: str) -> str:
    """MLflow/W&B name of one audit number, e.g. `audit/fnn.predict/out/rows`."""
    parts = ["audit", row["node"], row.get("side") or "", field]
    return "/".join(p for p in parts if p)
