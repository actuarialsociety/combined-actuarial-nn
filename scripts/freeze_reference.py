#!/usr/bin/env python3
"""Freeze the latest reference run into reference/ (committed), the single source of every
number in the lessons, the README and the notebooks' "reference run" columns.

    kedro run                                # the reference setup (conf/base), every pipeline
    python3 scripts/freeze_reference.py      # data/08_reporting -> reference/, then the
                                             # lesson figures (export_figures.py) and every
                                             # number in the lessons and README (fill_results.py)

Refuses to freeze a run that is not the reference setup (a benchmark reports SKIPPED).

  metrics.csv           every metric of every model (metrics.json)
  reconciliation.csv    paper value, this repo's value, tolerance, PASS/FAIL, source
  <pipeline>_findings.csv   which gaps are real; where the published run falls among our seeds
  environment.json      date, commit, device and package versions of the freeze
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT, OUT = ROOT / "data" / "08_reporting", ROOT / "reference"
PIPELINES = ["frequency", "mortality", "reserving"]
PACKAGES = ["kedro", "torch", "numpy", "pandas", "scikit-learn", "statsmodels", "lightgbm",
            "chainladder"]  # fmt: skip


def main() -> int:
    rec, rows = [], []
    for p in PIPELINES:
        for mfile in sorted((REPORT / p).glob("*/metrics.json")):
            model = mfile.parent.name
            for k, v in json.loads(mfile.read_text()).items():
                rows.append({"pipeline": p, "model": model, "metric": k, "value": v})
            rec.append(pd.read_csv(mfile.parent / "reconciliation.csv").assign(pipeline=p))
    rec = pd.concat(rec, ignore_index=True)
    skipped = rec[~rec.status.isin(["PASS", "FAIL"])]
    if len(skipped):
        print("Not the reference setup, nothing frozen:")
        print(skipped[["pipeline", "model", "metric", "status"]].to_string(index=False))
        return 1
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    rec[["pipeline", *rec.columns[:-1]]].to_csv(OUT / "reconciliation.csv", index=False)
    for p in PIPELINES:
        pd.read_csv(REPORT / p / "findings.csv").to_csv(OUT / f"{p}_findings.csv", index=False)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    env = {
        "frozen": date.today().isoformat(),
        "commit": commit,
        "package_version": metadata.version("deep-actuarial-science"),
        "device": os.environ.get("DAS_DEVICE", "default (CUDA if present, else CPU)"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: metadata.version(p) for p in PACKAGES},
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(rec.status.value_counts().to_string())
    print("frozen to", OUT.relative_to(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_figures
    import fill_results

    export_figures.main()
    return fill_results.main()


if __name__ == "__main__":
    sys.exit(main())
