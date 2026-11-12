"""Reconcile pipeline results to published reference values.

Benchmarks live in `conf/base/parameters_benchmarks.yml`. Each entry gives the
expected value of a metric, a tolerance and its source. A benchmark only
applies when the run uses the reference setup (e.g. the full dataset).

Tolerances: a number (closed-form fits, to rounding); "2sd", for a single published run,
which must lie within 2 sd of our per-seed values; "2se", for a published mean over `n`
seeds with sd `sd`, which must lie within 2 standard errors of the difference of the two
means, sqrt(sd_pub^2 / n_pub + sd_ours^2 / n_ours).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def reconcile(
    metrics: dict, benchmark: dict | list | None, applicable: bool, model: str
) -> pd.DataFrame:
    """One row per benchmarked metric. `benchmark` is one entry or a list of entries
    (e.g. a published table plus a software cross-check), each with its own source."""
    if isinstance(benchmark, list):
        parts = [reconcile(metrics, b, applicable, model) for b in benchmark]
        return (
            pd.concat(parts, ignore_index=True)
            if parts
            else reconcile(metrics, None, applicable, model)
        )
    rows = []
    expected = (benchmark or {}).get("expected", {})
    tolerance = (benchmark or {}).get("tolerance", 0.0)
    source = (benchmark or {}).get("source", "")
    if isinstance(source, list):  # [paper, table]
        source = ", ".join(source)
    for name, value in expected.items():
        tol = tolerance.get(name, 0.0) if isinstance(tolerance, dict) else tolerance
        if tol == "2sd":  # a single published run: within 2 sd of our seed distribution
            sd = metrics.get(f"{name}_sd")
            tol = None if sd is None else 2 * sd
        elif tol == "2se":  # published mean over n seeds vs ours: 2 standard errors of the gap
            sd, pub_sd = metrics.get(f"{name}_sd"), (benchmark.get("sd") or {}).get(name)
            n_pub, n_ours = benchmark.get("n"), metrics.get("n_runs")
            tol = (
                None
                if None in (sd, pub_sd, n_pub, n_ours)
                else 2 * float(np.sqrt(pub_sd**2 / n_pub + sd**2 / n_ours))
            )
        actual = metrics.get(name)
        if not applicable:
            status = "SKIPPED (not reference setup)"
        elif actual is None or tol is None:
            status = "MISSING"
        else:
            status = "PASS" if abs(actual - value) <= tol else "FAIL"
        rows.append(
            {
                "model": model,
                "metric": name,
                "expected": value,
                "actual": actual,
                "difference": None if actual is None else actual - value,
                "tolerance": tol,
                "status": status,
                "source": source,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "metric",
            "expected",
            "actual",
            "difference",
            "tolerance",
            "status",
            "source",
        ],
    )
    for r in df.itertuples():
        level = logging.WARNING if r.status == "FAIL" else logging.INFO
        log.log(
            level,
            "Benchmark %s.%s: expected %s, got %s -> %s",
            model,
            r.metric,
            r.expected,
            r.actual,
            r.status,
        )
    return df
