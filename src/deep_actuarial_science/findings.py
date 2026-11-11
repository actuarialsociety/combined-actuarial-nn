"""Findings: is a gap between two models real, or noise?

Published tables rank models on one test set to three decimals. Two checks say what such a
ranking is worth:

- `gaps`: a gap between two models is the mean of paired per-unit loss differences (per test
  policy, per company), so it has a standard error over units and a 95% interval. A gap whose
  interval contains 0 would not survive another test sample. The intervals are per pair; the
  `verdict` is Holm-adjusted over every pair in the table (`p_holm` < 0.05), so comparing many
  models does not manufacture "real" gaps. The interval treats each fitted model as fixed: it
  is test-sample noise only. For a network it is the mean over seeds; for a model fitted once
  (GBM, Lee-Carter) a "real" gap holds for that fit, not for every fit of the method.
- `seeds`: a network published as a single training run is one draw from its seed
  distribution. The table places that run among our seeds, and our seeds against a baseline.

Each pipeline's compare node writes its table to `data/08_reporting/<pipeline>/findings.csv`;
`reference/` holds the tables of the frozen reference run (`scripts/freeze_reference.py`).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from deep_actuarial_science.metrics import poisson_unit_deviance, rmse

REAL, NOISE = "real", "not distinguishable"
COLUMNS = ["worse", "better", "loss_worse", "loss_better", "gap", "ci_low", "ci_high"]
COLUMNS += ["p_value", "p_holm", "verdict", "units", "adjacent"]
AVERAGED = re.compile(r"_x\d+$")  # `fnn_x15`: the average of the 15 seeds' predictions of `fnn`


def gaps(losses: dict[str, np.ndarray]) -> pd.DataFrame:
    """Every pair of models: mean loss of `worse` minus `better`, with the 95% t-interval of
    the paired differences over units and the two-sided p-value, Holm-adjusted over all pairs
    (`p_holm`, which sets the verdict). `adjacent` marks neighbours on the leaderboard: those
    rows, read top to bottom, say which steps of the ranking are real."""
    names = sorted(losses, key=lambda m: -float(np.mean(losses[m])))
    rows = []
    for i, worse in enumerate(names):
        for j in range(i + 1, len(names)):
            d = np.asarray(losses[worse], float) - np.asarray(losses[names[j]], float)
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
            gap = float(d.mean())
            p = float(2 * stats.t.sf(abs(gap) / se, n - 1)) if n > 1 and se > 0 else np.nan
            rows.append(
                {
                    "worse": worse,
                    "better": names[j],
                    "loss_worse": float(np.mean(losses[worse])),
                    "loss_better": float(np.mean(losses[names[j]])),
                    "gap": gap,
                    "ci_low": gap - half,
                    "ci_high": gap + half,
                    "p_value": p,
                    "units": n,
                    "adjacent": j == i + 1,
                }
            )
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["p_holm"] = holm(out.p_value.to_numpy())
    out["verdict"] = np.where((out.p_holm < 0.05) & (out.gap > 0), REAL, NOISE)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order]).clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def ladder(table: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The rows of `gaps` that step down the ranking of `names` only, worst pair first."""
    loss = pd.concat([table.set_index("worse").loss_worse, table.set_index("better").loss_better])
    loss = loss[~loss.index.duplicated()].reindex(names).dropna().sort_values(ascending=False)
    pairs = pd.MultiIndex.from_arrays([loss.index[:-1], loss.index[1:]], names=["worse", "better"])
    return table.set_index(["worse", "better"]).reindex(pairs).reset_index()


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """The reading order of a `gaps` table: the ranking of the models step by step, then what
    averaging each network's seeds adds (`<model>` to `<model>_xN`)."""
    names = pd.unique(table[["worse", "better"]].to_numpy().ravel())
    single = [n for n in names if not AVERAGED.search(n)]
    averaging = table[[b.startswith(f"{w}_x") for w, b in zip(table.worse, table.better)]]
    return pd.concat([ladder(table, single), averaging], ignore_index=True)


def frequency(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gaps in test Poisson deviance per policy (10^-2), paired over test policies. A
    multi-seed network enters twice: as the mean of its single fits (what the papers publish)
    and as `<model>_xN`, the average of its N predictions."""
    losses, ids = {}, None
    for m, p in preds.items():
        t = p[p.split == "test"]
        if ids is not None and not np.array_equal(ids, t.IDpol.to_numpy()):
            raise ValueError(f"{m}: predictions are from another test set; re-run the models")
        ids, y = t.IDpol.to_numpy(), t.ClaimNb.to_numpy()
        runs = [c for c in t.columns if c.startswith("pred_run_")]
        if runs:
            losses[m] = np.mean([poisson_unit_deviance(y, t[c]) for c in runs], axis=0)
            losses[f"{m}_x{len(runs)}"] = poisson_unit_deviance(y, t.pred)
        else:
            losses[m] = poisson_unit_deviance(y, t.pred)
    return gaps(losses)


def reserving(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gaps in the absolute percentage error of ultimate loss (its mean is the MAPE of Kuo
    2019), paired over companies."""
    from deep_actuarial_science.models.reserving import ultimates

    losses = {}
    for m, p in preds.items():
        u = ultimates(p)
        losses[m] = np.abs(u.predicted_ultimate / u.actual_ultimate - 1).to_numpy()
    return gaps(losses)


def seed_rmse(results: pd.DataFrame) -> np.ndarray:
    """Out-of-sample RMSE of log m for each seed of one model (one value for a single fit)."""
    out = results[results.split == "test"]
    runs = [c for c in out.columns if c.startswith("pred_run_")] or ["pred"]
    return np.array([rmse(out.logmx, out[c]) for c in runs])


def seeds(
    per_seed: dict[str, np.ndarray],
    ensemble: dict[str, float],
    published: dict[str, float | None],
    baseline: str,
) -> pd.DataFrame:
    """Where the published single run falls among our seeds, and how many seeds beat the
    baseline model. `ensemble` is the score of the seed-averaged forecast."""
    base = float(per_seed[baseline].mean()) if baseline in per_seed else np.nan
    rows = []
    for m, s in per_seed.items():
        pub = published.get(m)
        if pub is None or len(s) < 2:
            place = ""
        elif pub < s.min():
            place = "better than every seed"
        elif pub > s.max():
            place = "worse than every seed"
        else:
            place = "inside the seed range"
        rows.append(
            {
                "model": m,
                "published": pub,
                "seeds": len(s),
                "seed_min": float(s.min()),
                "seed_mean": float(s.mean()),
                "seed_max": float(s.max()),
                "seeds_at_or_below_published": int((s <= pub).sum()) if place else np.nan,
                "published_run": place,
                f"seeds_beating_{baseline}": int((s < base).sum()) if m != baseline else np.nan,
                "ensemble": ensemble[m],
            }
        )
    return pd.DataFrame(rows)


def mortality(results: dict[str, pd.DataFrame], published: dict[str, float | None]):
    """`seeds` on the out-of-sample RMSE of log m; Lee–Carter is the baseline. Also returns
    the per-seed scores (for the figure)."""
    per_seed = {m: seed_rmse(r) for m, r in results.items()}
    ensemble = {
        m: rmse(r[r.split == "test"].logmx, r[r.split == "test"].pred) for m, r in results.items()
    }
    return seeds(per_seed, ensemble, published, "lee_carter"), per_seed
