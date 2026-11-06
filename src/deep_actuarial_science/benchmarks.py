"""Reconcile pipeline results to published reference values.

Benchmarks live in `conf/base/parameters_benchmarks.yml`. Each entry gives the
expected value of a metric, a tolerance and its source. A benchmark only
applies when the run uses the reference setup (e.g. the full dataset).
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def reconcile(metrics: dict, benchmark: dict | None, applicable: bool, model: str) -> pd.DataFrame:
    rows = []
    expected = (benchmark or {}).get("expected", {})
    tol = (benchmark or {}).get("tolerance", 0.0)
    source = (benchmark or {}).get("source", "")
    for name, value in expected.items():
        actual = metrics.get(name)
        if not applicable:
            status = "SKIPPED (not reference setup)"
        elif actual is None:
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
