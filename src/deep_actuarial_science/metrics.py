"""Scoring rules used across lessons. Deviances follow Wüthrich & Merz (2023): × 100."""

from __future__ import annotations

import numpy as np
from scipy.special import xlogy
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


def poisson_unit_deviance(y, mu) -> np.ndarray:
    """Each policy's contribution to `poisson_deviance_per_policy` (their mean equals it):
    100 * 2 * (mu - y + y log(y / mu)). The exposure cancels."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    return 200 * (mu - y + xlogy(y, y / mu))


def balance_ratio(y, mu) -> float:
    """Sum of predicted / sum of observed claims. 1.0 = balance property holds."""
    return float(np.sum(mu) / np.sum(y))


def gini_index(y, mu, v) -> float:
    """Exposure-weighted Gini index of the ordered Lorenz curve (higher = better ranking).
    Policies with the same predicted rate are one point of the curve, so a constant
    prediction scores 0 whatever the order of the data."""
    y, mu, v = (np.asarray(a, dtype=float) for a in (y, mu, v))
    # riskiest rate first; rounding makes equal rates equal despite float error in mu / v
    _, group = np.unique(-np.round(np.log(mu / v), 10), return_inverse=True)
    cum_v = np.cumsum(np.bincount(group, weights=v)) / v.sum()
    cum_y = np.cumsum(np.bincount(group, weights=y)) / y.sum()
    area = np.trapezoid(np.r_[0, cum_y], np.r_[0, cum_v])
    return float(2 * area - 1)


def rmse(y, y_hat) -> float:
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def to_mlflow(metrics: dict[str, float]) -> dict[str, dict]:
    """Shape a flat metric dict for kedro-mlflow's MlflowMetricsHistoryDataset."""
    return {k: {"value": float(v), "step": 0} for k, v in metrics.items() if v is not None}
