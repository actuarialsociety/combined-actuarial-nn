#!/usr/bin/env python3
"""Fill lesson result tables from the latest pipeline outputs (never type numbers by hand).

In a lesson, write a placeholder ⟪model.metric⟫. This script replaces it with the value
wrapped in invisible markers, <!--r:model.metric-->47.179<!--/r-->, so re-running after
a new pipeline run refreshes every number in place.

Metrics come from data/08_reporting/<pipeline>/<model>/metrics.json. Special keys:
  <model>.status        PASS / FAIL / SKIPPED from reconciliation.csv
  <model>.package_diff  actual value of the chain-ladder vs package reconciliation
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "08_reporting"
PLACEHOLDER = re.compile(r"⟪([\w.]+)⟫|<!--r:([\w.]+)-->.*?<!--/r-->")


def collect() -> dict[str, str]:
    values: dict[str, str] = {}
    for mfile in REPORT.glob("*/*/metrics.json"):
        model = mfile.parent.name
        for k, v in json.loads(mfile.read_text()).items():
            values[f"{model}.{k}"] = fmt(k, v)
        rec = mfile.parent / "reconciliation.csv"
        if rec.exists():
            r = pd.read_csv(rec)
            if len(r):
                s = sorted(set(r.status.str.split(" ").str[0]))
                values[f"{model}.status"] = "**FAIL**" if "FAIL" in s else " / ".join(s)
                pkg = r[r.metric == "max_rel_diff_vs_package"]
                if len(pkg):
                    values[f"{model}.package_diff"] = f"{pkg.actual.iloc[0]:.1e}"
            else:
                values[f"{model}.status"] = "n/a"
    return values


def fmt(key: str, v) -> str:
    if v is None:
        return "n/a"
    if key.startswith("rmse"):
        return f"{v:.4f}"
    if key.startswith("deviance") or key.startswith("gini"):
        return f"{v:.3f}"
    if key.startswith("balance"):
        return f"{v:.4f}"
    if key.endswith("_pct"):
        return f"{v:+.1f}" if key.startswith("total") else f"{v:.1f}"
    if key.startswith("share"):
        return f"{100 * v:.0f}%"
    if key in ("n_params", "n_companies", "best_epoch"):
        return f"{int(v):,}"
    if key == "fit_seconds":
        return f"{v:.0f}"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main() -> None:
    values = collect()
    missing = set()
    for lesson in sorted((ROOT / "lessons").glob("*.md")):
        text = lesson.read_text(encoding="utf-8")

        def repl(m: re.Match) -> str:
            key = m.group(1) or m.group(2)
            if key not in values:
                missing.add(key)
                return m.group(0)
            return f"<!--r:{key}-->{values[key]}<!--/r-->"

        new = PLACEHOLDER.sub(repl, text)
        if new != text:
            lesson.write_text(new, encoding="utf-8")
            print("updated", lesson.relative_to(ROOT))
    for k in sorted(missing):
        print("no value for", k, "(run the pipeline first)")


if __name__ == "__main__":
    main()
