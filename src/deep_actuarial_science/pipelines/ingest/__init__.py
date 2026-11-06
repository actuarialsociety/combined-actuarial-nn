"""Download public datasets once and cache them under data/01_raw.

- AI Tools for Actuaries `Data.zip` (Wüthrich et al. 2025): freMTPL2 motor claims
  (678,007 policies, fixed learn/test split) and Swiss HMD mortality rates.
- CAS Loss Reserving Database (Meyers & Shi 2011): NAIC Schedule P paid triangles for
  six US lines of business, including the realised lower triangle for back-testing.
"""

from __future__ import annotations

import io
import logging
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

log = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "deep-actuarial-science"})
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    return dest


def read_hmd_mx(text: str) -> pd.DataFrame:
    """Parse an HMD `Mx_1x1.txt` file into Year, Age, Female, Male, Total."""
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=2)
    df.columns = ["Year", "Age", "Female", "Male", "Total"]
    df["Age"] = df["Age"].astype(str).str.replace("+", "", regex=False)
    return df.replace(".", np.nan).apply(pd.to_numeric, errors="coerce")


def fetch_aitools(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    zpath = _download(params["url"], Path(params["cache_dir"]) / "Data.zip")
    with zipfile.ZipFile(zpath) as z:
        freq = pd.read_parquet(io.BytesIO(z.read(params["frequency_member"])))
        mort = read_hmd_mx(z.read(params["mortality_member"]).decode("utf-8"))
    log.info("freMTPL2: %s rows; mortality: %s rows", len(freq), len(mort))
    return freq, mort


def fetch_cas(params: dict) -> pd.DataFrame:
    frames = []
    for lob in params["lines"]:
        f = _download(params["url"].format(lob=lob), Path(params["cache_dir"]) / f"{lob}_pos.csv")
        d = pd.read_csv(f)
        # Column suffixes differ by line (_B, _C, _D, _F2, _h1, _R1): drop them.
        d.columns = [re.sub(r"_[A-Za-z]\d?$", "", c) for c in d.columns]
        frames.append(d.assign(lob=lob))
    out = pd.concat(frames, ignore_index=True)
    log.info("CAS Schedule P: %s rows, %s companies", len(out), out.GRCODE.nunique())
    return out


aitools_node = node(
    fetch_aitools, "params:ingest.aitools", ["freq_raw", "mortality_raw"], name="fetch_aitools_data"
)
cas_node = node(fetch_cas, "params:ingest.cas", "cas_raw", name="fetch_cas_schedule_p")


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([aitools_node, cas_node], tags="ingest")
