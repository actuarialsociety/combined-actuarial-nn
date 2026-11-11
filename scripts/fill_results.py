#!/usr/bin/env python3
"""Fill every number in the lessons and the README from the frozen reference run in
reference/ (never type numbers by hand). `--check` changes nothing and fails if a file is
out of date or a key is unknown (CI).

In a lesson, write a placeholder ⟪model.metric⟫. This script replaces it with the value
wrapped in invisible markers, <!--r:model.metric-->47.179<!--/r-->, so a new reference run
(`scripts/freeze_reference.py`) refreshes every number in place. Keys:

  <model>.<metric>          reference/metrics.csv, and the columns of mortality_findings.csv
  <model>.status            PASS / FAIL from reconciliation.csv
  <model>.package_diff      the chain-ladder's largest difference to the chainladder package
  gap.<worse>.<better>      a gap between two models with its 95% interval (findings.gaps)
  verdict.<worse>.<better>  "real" or "not distinguishable"
  count.PASS, count.FAIL    benchmarked metrics over all models
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
PLACEHOLDER = re.compile(r"⟪([\w.]+)⟫|<!--r:([\w.]+)-->.*?<!--/r-->")


def collect() -> dict[str, str]:
    values: dict[str, str] = {}
    for r in pd.read_csv(REF / "metrics.csv").itertuples():
        values[f"{r.model}.{r.metric}"] = fmt(r.metric, r.value)
    rec = pd.read_csv(REF / "reconciliation.csv")
    for model, r in rec.groupby("model"):
        values[f"{model}.status"] = "**FAIL**" if "FAIL" in set(r.status) else "PASS"
    pkg = rec[rec.metric == "max_rel_diff_vs_package"]
    values[f"{pkg.model.iloc[0]}.package_diff"] = f"{pkg.actual.iloc[0]:.1e}"
    for status in ("PASS", "FAIL"):
        values[f"count.{status}"] = str(int((rec.status == status).sum()))
    for pipeline, digits in (("frequency", 3), ("reserving", 4)):
        for r in pd.read_csv(REF / f"{pipeline}_findings.csv").itertuples():
            pair = f"{r.worse}.{r.better}"
            values[f"gap.{pair}"] = (
                f"{r.gap:.{digits}f} ({r.ci_low:.{digits}f} to {r.ci_high:.{digits}f})"
            )
            values[f"verdict.{pair}"] = r.verdict
    for row in pd.read_csv(REF / "mortality_findings.csv").to_dict("records"):
        for k, v in row.items():
            if k != "model" and pd.notna(v):
                values[f"{row['model']}.{k}"] = v if isinstance(v, str) else fmt_seed(k, v)
    return values


def fmt_seed(key: str, v: float) -> str:
    return f"{int(v)}" if key.startswith("seeds") else f"{v:.4f}"


def fmt(key: str, v) -> str:
    if pd.isna(v):
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
    if key in ("n_params", "n_companies", "best_epoch", "n_runs"):
        return f"{int(v):,}"
    if key == "fit_seconds":
        return f"{v:.0f}"
    return f"{v:.4g}"


def main(check: bool = False) -> int:
    values = collect()
    missing, stale = set(), []
    for doc in [ROOT / "README.md", *sorted((ROOT / "lessons").glob("*.md"))]:
        text = doc.read_text(encoding="utf-8")

        def repl(m: re.Match) -> str:
            key = m.group(1) or m.group(2)
            if key not in values:
                missing.add(key)
                return m.group(0)
            return f"<!--r:{key}-->{values[key]}<!--/r-->"

        new = PLACEHOLDER.sub(repl, text)
        if new != text:
            stale.append(doc.relative_to(ROOT))
            if not check:
                doc.write_text(new, encoding="utf-8")
                print("updated", doc.relative_to(ROOT))
    for k in sorted(missing):
        print("no value for", k, "(not in reference/)")
    if check and stale:
        print("out of date (run scripts/fill_results.py):", *stale)
    return int(bool(missing) or (check and bool(stale)))


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
