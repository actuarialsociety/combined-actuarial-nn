"""Scoring rules used across lessons. Deviances follow Wüthrich & Merz (2023): × 100."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_poisson_deviance


def poisson_deviance(y, mu, v) -> float:
    """Exposure-weighted Poisson deviance of frequencies, in units of 10^-2.

    y: claim counts, mu: expected claim counts (already × exposure), v: exposure.
    Matches the AI Tools for Actuaries notebooks exactly.
    """
    y, mu, v = (np.asarray(a, dtype=float) for a in (y, mu, v))
    return 100 * mean_poisson_deviance(y / v, mu / v, sample_weight=v)


def poisson_deviance_per_policy(y, mu, v) -> float:
    """Poisson deviance divided by the number of policies n (not total exposure), in 10^-2:
    the scale of Wüthrich & Merz (2023), eq. (5.28), Tables 5.5 and 7.x, and of the papers
    built on it (Richman & Wüthrich 2023; Brauer 2024)."""
    v = np.asarray(v, dtype=float)
    return poisson_deviance(y, mu, v) * v.sum() / len(v)


def balance_ratio(y, mu) -> float:
    """Sum of predicted / sum of observed claims. 1.0 = balance property holds."""
    return float(np.sum(mu) / np.sum(y))


def gini_index(y, mu, v) -> float:
    """Exposure-weighted Gini index of the ordered Lorenz curve (higher = better ranking)."""
    y, mu, v = (np.asarray(a, dtype=float) for a in (y, mu, v))
    order = np.argsort(-(mu / v), kind="stable")
    cum_v = np.cumsum(v[order]) / v.sum()
    cum_y = np.cumsum(y[order]) / y.sum()
    area = np.trapezoid(np.r_[0, cum_y], np.r_[0, cum_v])
    return float(2 * area - 1)


def rmse(y, y_hat) -> float:
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def to_mlflow(metrics: dict[str, float]) -> dict[str, dict]:
    """Shape a flat metric dict for kedro-mlflow's MlflowMetricsHistoryDataset."""
    return {k: {"value": float(v), "step": 0} for k, v in metrics.items() if v is not None}
