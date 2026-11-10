#!/usr/bin/env python3
"""Exploratory data analysis of every dataset the models use. Local only.

    python scripts/eda.py                       # all three datasets
    python scripts/eda.py frequency mortality   # a subset

Writes `data/10_eda/<dataset>/report.md` with PNG figures and CSV tables, and an index at
`data/10_eda/README.md` (git-ignored, like the rest of `data/`).

This is deliberately not a Kedro pipeline: no session starts, so the MLflow and W&B hooks never
fire and nothing is logged to either. Each dataset is rebuilt from the cached downloads with the
pipelines' own `prepare` functions and the same config, so the EDA describes exactly what the
models are given, and never reads files a concurrent `kedro run` is rewriting.

| Dataset                      | Models                                                 |
| ---------------------------- | ------------------------------------------------------ |
| freMTPL2 (frequency)         | homogeneous, glm, glm3, gbm, fnn, cann, localglmnet    |
| Swiss HMD m(x,t) (mortality) | lee_carter, cnn, cnn_tanh, lstm, transformer           |
| CAS Schedule P (reserving)   | chain_ladder, neural_odp                               |
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # the ingest cache paths in conf/ are relative to the project root

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kedro.config import OmegaConfigLoader  # noqa: E402

from deep_actuarial_science import plots  # noqa: E402  (also applies the figure style)
from deep_actuarial_science.models import mortality as mm  # noqa: E402
from deep_actuarial_science.models import reserving as rm  # noqa: E402
from deep_actuarial_science.models.frequency import CLIP_UPPER, NUMERIC  # noqa: E402
from deep_actuarial_science.pipelines.frequency.nodes import prepare as freq_prepare  # noqa: E402
from deep_actuarial_science.pipelines.ingest import fetch_aitools, fetch_cas  # noqa: E402

log = logging.getLogger("eda")
OUT = ROOT / "data" / "10_eda"
INK, P = plots.INK, plots.PALETTE
MARKERS = ["o", "s", "^", "D", "v", "P"]  # second encoding: amber/green are close under CVD
SEQ, DIV = "Blues", "RdBu_r"
plt.rcParams["figure.constrained_layout.use"] = True  # titles and shared colour bars


def load_params(env: str) -> dict:
    # Same merge as settings.py, which is not imported because it pulls in the tracking hooks.
    return OmegaConfigLoader(
        conf_source="conf",
        env=env,
        base_env="base",
        default_run_env="local",
        merge_strategy={"parameters": "soft"},
    )["parameters"]


class Report:
    """One dataset's report.md, with its figures and tables alongside."""

    def __init__(self, name: str, title: str, intro: str, root: Path = OUT):
        self.dir = root / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", "", intro, ""]

    def h(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def note(self, text: str) -> None:
        self.lines += [text, ""]

    def table(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        """Save the full table as CSV and print it in the report."""
        df.to_csv(self.dir / f"{name}.csv", index=index)
        d = df.reset_index() if index else df
        cells = d.map(
            lambda v: (
                ("n/a" if pd.isna(v) else f"{v:,.4g}")
                if isinstance(v, float | np.floating)
                else f"{v:,}"
                if isinstance(v, int | np.integer)
                else str(v)
            )
        )
        self.lines += [
            "| " + " | ".join(map(str, d.columns)) + " |",
            "|" + " --- |" * d.shape[1],
            *("| " + " | ".join(r) + " |" for r in cells.to_numpy().tolist()),
            "",
        ]

    def fig(self, fig: plt.Figure, name: str, caption: str) -> None:
        fig.savefig(self.dir / f"{name}.png")
        plt.close(fig)
        self.lines += [f"**{caption}**", "", f"![{caption}]({name}.png)", ""]

    def close(self) -> Path:
        (self.dir / "report.md").write_text("\n".join(self.lines) + "\n")
        return self.dir / "report.md"


def grid(n: int, ncols: int, w: float, h: float):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)[:n]


# --- Frequency: freMTPL2 --------------------------------------------------------------------

# Right-open bins; the last bin of each is open-ended ("150+").
BINS = {
    "DrivAge": [18, 21, 26, 31, 41, 51, 61, 71, 81, 200],
    "BonusMalus": [50, 51, 60, 70, 80, 90, 100, 110, 125, 150, 1000],
    "VehAge": [0, 1, 2, 3, 6, 11, 16, 21, 1000],
    "VehPower": [4, 5, 6, 7, 8, 9, 10, 12, 100],
}
ORDERED = [*BINS, "Density", "Area"]
NOMINAL = ["VehGas", "VehBrand", "Region"]


def _bin(s: pd.Series, feature: str) -> pd.Series:
    if feature == "Density":  # log scale, as the GLMs and networks use it
        edges = list(range(0, 12))
        labels = [f"ln {a}-{a + 1}" for a in edges[:-1]]
        return pd.cut(np.log(s), edges, right=False, labels=labels, include_lowest=True)
    if feature in BINS:
        e = BINS[feature]
        labels = [
            str(a) if b == a + 1 else f"{a}+" if i == len(e) - 2 else f"{a}-{b - 1}"
            for i, (a, b) in enumerate(zip(e, e[1:]))
        ]
        return pd.cut(s, e, right=False, labels=labels)
    return s.astype(str)


def one_way(learn: pd.DataFrame, test: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Exposure, claims and observed frequency per feature level, learn and test."""
    parts = {}
    for split, d in (("learn", learn), ("test", test)):
        g = d.groupby(_bin(d[feature], feature), observed=False)
        t = g[["Exposure", "ClaimNb"]].sum()
        t["freq"] = t.ClaimNb / t.Exposure
        t["se"] = np.sqrt(t.ClaimNb) / t.Exposure  # Poisson standard error
        t["share"] = t.Exposure / t.Exposure.sum()
        parts[split] = t
    out = pd.concat([t.add_suffix(f"_{s}") for s, t in parts.items()], axis=1)
    if feature in NOMINAL:
        out = out.sort_values("Exposure_learn", ascending=False)
    return out[out.Exposure_learn.fillna(0) + out.Exposure_test.fillna(0) > 0]


def psi(p: pd.Series, q: pd.Series) -> float:
    """Population stability index between two exposure distributions."""
    p, q = p.fillna(0).clip(lower=1e-6), q.fillna(0).clip(lower=1e-6)
    return float(((p - q) * np.log(p / q)).sum())


def frequency_eda(raw: pd.DataFrame, params: dict) -> Path:
    learn, test = freq_prepare(raw, params["frequency"]["data"])
    all_ = pd.concat([learn, test])
    r = Report(
        "frequency",
        "EDA: freMTPL2 claim frequency",
        "Used by `homogeneous`, `glm`, `glm3`, `gbm`, `fnn`, `cann` and `localglmnet`, on the "
        "fixed learn/test split. Claim frequency is claims per policy-year (`ClaimNb / "
        "Exposure`); every rate here is exposure-weighted.",
    )

    r.h("Overview")
    r.table(
        pd.DataFrame(
            [
                {
                    "split": s,
                    "policies": len(d),
                    "exposure_years": float(d.Exposure.sum()),
                    "claims": int(d.ClaimNb.sum()),
                    "frequency": float(d.ClaimNb.sum() / d.Exposure.sum()),
                    "share_with_claim": float((d.ClaimNb > 0).mean()),
                    "mean_exposure": float(d.Exposure.mean()),
                }
                for s, d in (("learn", learn), ("test", test))
            ]
        ),
        "overview",
    )

    r.h("Data quality")
    checks = {
        "missing cells": int(all_.isna().sum().sum()),
        "duplicated IDpol": int(all_.IDpol.duplicated().sum()),
        "IDpol in both learn and test": len(set(learn.IDpol) & set(test.IDpol)),
        "Exposure <= 0": int((all_.Exposure <= 0).sum()),
        "Exposure > 1 year": int((all_.Exposure > 1).sum()),
        "max Exposure": float(all_.Exposure.max()),
        "max ClaimNb": int(all_.ClaimNb.max()),
        "policies with ClaimNb >= 5": int((all_.ClaimNb >= 5).sum()),
        "RandN mean / sd": f"{all_.RandN.mean():.3f} / {all_.RandN.std():.3f}",
        "RandU mean / sd": f"{all_.RandU.mean():.3f} / {all_.RandU.std():.3f}",
    }
    for col, upper in zip(NUMERIC, CLIP_UPPER):
        checks[f"{col} above the network clip ({upper})"] = int((all_[col] > upper).sum())
    r.table(pd.DataFrame({"check": list(checks), "value": list(checks.values())}), "quality")

    r.h("Target and exposure")
    counts = (
        pd.DataFrame(
            {
                s: d.ClaimNb.clip(upper=5).astype(int).value_counts(normalize=True)
                for s, d in (("learn", learn), ("test", test))
            }
        )
        .sort_index()
        .rename_axis("claims")
    )
    counts.index = [str(i) if i < 5 else "5+" for i in counts.index]
    r.table(counts.rename_axis("claims").reset_index(), "claim_counts")
    fig, axes = grid(2, 2, 5, 3.4)
    axes[0].hist(learn.Exposure, bins=50, color=P[0], edgecolor="none")
    axes[0].set(title="Policy exposure, learn", xlabel="exposure (years)", ylabel="policies")
    x = np.arange(len(counts))
    axes[1].bar(x - 0.2, counts.learn, 0.38, color=P[0], label="learn")
    axes[1].bar(x + 0.2, counts.test, 0.38, color=P[1], label="test")
    axes[1].set(title="Claim counts", xlabel="claims on the policy", ylabel="share of policies")
    axes[1].set_xticks(x, counts.index)
    axes[1].set_yscale("log")
    axes[1].legend()
    r.fig(fig, "target_exposure", "Exposure distribution and claim counts")

    r.h("One-way views")
    r.note(
        "Observed frequency per level with ±2 Poisson standard errors. Learn and test should "
        "agree where exposure is large; a level with a wide bar has little data behind it."
    )
    features = [*ORDERED, *NOMINAL]
    tables = {f: one_way(learn, test, f) for f in features}
    fig, axes = grid(len(features), 3, 5, 3.3)
    for ax, f in zip(axes, features):
        t = tables[f]
        x = np.arange(len(t))
        style = {"ls": "-"} if f in ORDERED else {"ls": "none"}
        for off, s, c, m in ((-0.08, "learn", P[0], "o"), (0.08, "test", P[1], "s")):
            ax.errorbar(
                x + off,
                t[f"freq_{s}"],
                2 * t[f"se_{s}"],
                color=c,
                marker=m,
                ms=4,
                lw=1.5,
                elinewidth=1,
                label=s,
                **style,
            )
        ax.set_xticks(x, t.index, rotation=60, ha="right", fontsize=8)
        ax.set(title=f, ylabel="claims per year")
    axes[0].legend()
    r.fig(fig, "one_way_frequency", "Observed claim frequency by feature")

    fig, axes = grid(len(features), 3, 5, 3.0)
    for ax, f in zip(axes, features):
        t = tables[f]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.share_learn, 0.38, color=P[0], label="learn")
        ax.bar(x + 0.2, t.share_test, 0.38, color=P[1], label="test")
        ax.set_xticks(x, t.index, rotation=60, ha="right", fontsize=8)
        ax.set(title=f, ylabel="exposure share")
    axes[0].legend()
    r.fig(fig, "exposure_by_feature", "Exposure share by feature, learn vs test")

    rows = []
    for f, t in tables.items():
        big = t[t.share_learn >= 0.01]  # ignore thin levels when ranking
        lo, hi = big.freq_learn.idxmin(), big.freq_learn.idxmax()
        rows.append(
            {
                "feature": f,
                "lowest_level": lo,
                "lowest_freq": float(big.freq_learn[lo]),
                "highest_level": hi,
                "highest_freq": float(big.freq_learn[hi]),
                "ratio": float(big.freq_learn[hi] / big.freq_learn[lo]),
                "psi_learn_vs_test": psi(t.share_learn, t.share_test),
            }
        )
    r.note(
        "Spread of learn frequency across levels holding at least 1% of exposure, and the "
        "population stability index between the learn and test exposure mix (PSI below 0.1 is "
        "conventionally read as a stable mix)."
    )
    r.table(pd.DataFrame(rows).sort_values("ratio", ascending=False), "one_way_summary")
    pd.concat(tables, names=["feature", "level"]).to_csv(r.dir / "one_way.csv")

    r.h("Dependence between features")
    cols = [*NUMERIC, "Density", "Exposure", "ClaimNb", "RandU", "RandN"]
    corr = learn[cols].corr(method="spearman")
    corr.round(4).to_csv(r.dir / "spearman.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(v) > 0.6 else "#333333",
        )
    fig.colorbar(im, ax=ax, label="Spearman correlation", shrink=0.8)
    ax.set_title("Rank correlation, learn")
    r.fig(fig, "spearman", "Spearman correlation of numeric features, target and noise controls")
    r.note(
        "`RandU` and `RandN` are the LocalGLMnet pure-noise controls: their correlation with "
        "everything should be about zero."
    )
    return r.close()


# --- Mortality: Swiss HMD --------------------------------------------------------------------

SEXES = ["Male", "Female"]
AGE_BANDS = {"0-14": (0, 14), "15-44": (15, 44), "45-64": (45, 64), "65-89": (65, 89)}


def _surface(ax, piv: pd.DataFrame, obs_year: int, **kw):
    im = ax.imshow(
        piv.to_numpy(),
        aspect="auto",
        origin="lower",
        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
        **kw,
    )
    ax.axvline(obs_year + 0.5, color=INK, lw=1.2, ls="--")
    ax.grid(False)
    ax.set(xlabel="calendar year", ylabel="age")
    return im


def _trend(d: pd.DataFrame, first: int, last: int) -> float:
    """OLS slope of band-average log m per year, x100: about % change in the rate per year."""
    g = d[d.Year.between(first, last)].groupby("Year").logmx.mean()
    return float(np.polyfit(g.index, g.to_numpy(), 1)[0] * 100)


def mortality_eda(raw: pd.DataFrame, base: dict, up: dict) -> Path:
    cfg = dict(up["mortality"]["data"], sexes=SEXES)
    d = mm.prepare(raw, cfg)
    oy, floor = cfg["obs_year"], np.exp(-11)
    first, last = int(d.Year.min()), int(d.Year.max())
    piv = {s: d[d.Sex == s].pivot(index="Age", columns="Year", values="logmx") for s in SEXES}
    r = Report(
        "mortality",
        "EDA: Swiss mortality rates (HMD)",
        f"Used by `lee_carter`, `cnn`, `cnn_tanh`, `lstm` and `transformer`. Log central death "
        f"rates, ages 0-{int(d.Age.max())}, {first}-{last}; learn up to {oy}, forecast after. "
        f"The reference setup uses {', '.join(base['mortality']['data']['sexes'])}; scale_up "
        "fits both sexes.",
    )

    r.h("Data quality")
    in_window = raw[raw.Year.between(first, last)]
    r.note(
        f"Raw file: {len(raw):,} rows, {int(raw.Year.min())}-{int(raw.Year.max())}, ages "
        f"{int(raw.Age.min())}-{int(raw.Age.max())}. The models drop years outside the window "
        f"and ages {cfg['max_age']}+ ({int((in_window.Age >= cfg['max_age']).sum()):,} "
        "rows inside the window)."
    )
    rows = []
    for s in SEXES:
        x = d[d.Sex == s]
        rows.append(
            {
                "sex": s,
                "cells": len(x),
                "expected_cells": x.Year.nunique() * x.Age.nunique(),
                "missing_mx": int(x.mx.isna().sum()),
                "floored_mx": int((x.mx < floor).sum()),
                "floored_learn": int(((x.mx < floor) & (x.split == "learn")).sum()),
                "floored_test": int(((x.mx < floor) & (x.split == "test")).sum()),
                "oldest_floored_age": x[x.mx < floor].Age.max(),
                "min_positive_mx": float(x.mx[x.mx > 0].min()),
                "max_mx": float(x.mx.max()),
            }
        )
    r.table(pd.DataFrame(rows), "quality")
    r.note(
        "`floored_mx` counts rates below exp(-11), which `prepare` sets to exp(-11) before "
        "taking logs (zero deaths at young ages in a small population)."
    )

    r.h("The mortality surface")
    fig, axes = grid(2, 2, 5.4, 3.8)
    kw = dict(cmap=SEQ, vmin=-11, vmax=-0.5)
    for ax, s in zip(axes, SEXES):
        im = _surface(ax, piv[s], oy, **kw)
        ax.set_title(f"{s}: log m(x,t)")
    fig.colorbar(im, ax=list(axes), label="log m(x,t)", shrink=0.9)
    r.fig(fig, "surface", "Log mortality by age and year (dashed: end of learn)")

    fig, ax = plt.subplots(figsize=(7, 4))
    im = _surface(ax, piv["Male"] - piv["Female"], oy, cmap=DIV, vmin=-1.5, vmax=1.5)
    fig.colorbar(im, ax=ax, label="log m male − log m female")
    ax.set_title("Sex gap in log mortality")
    r.fig(fig, "sex_gap", "Male minus female log mortality (0 = equal rates)")

    r.h("Age and time structure")
    years = [y for y in (first, 1950, oy, last) if y in piv["Male"].columns]
    fig, axes = grid(2, 2, 5, 3.4)
    for ax, s in zip(axes, SEXES):
        for i, y in enumerate(years):
            ax.plot(piv[s].index, piv[s][y], color=P[i], lw=1.6, label=str(y))
        ax.set(title=f"{s}: age profile", xlabel="age", ylabel="log m(x,t)")
    axes[0].legend()
    r.fig(fig, "age_profiles", "Age profile of log mortality in selected years")

    ages = [a for a in (0, 20, 40, 65, 85) if a in piv["Male"].index]
    fig, axes = grid(2, 2, 5, 3.4)
    for ax, s in zip(axes, SEXES):
        for i, a in enumerate(ages):
            ax.plot(piv[s].columns, piv[s].loc[a], color=P[i], lw=1.3, label=f"age {a}")
        ax.axvline(oy + 0.5, color=INK, lw=1, ls="--")
        ax.set(title=f"{s}: log m(x,t) over time", xlabel="calendar year", ylabel="log m(x,t)")
    axes[0].legend(ncols=2, fontsize=8)
    r.fig(fig, "time_trends", "Log mortality over time at selected ages (dashed: end of learn)")

    r.h("Is the forecast period like the learning period?")
    r.note(
        "Slope of band-average log m per year, x100 (about the % change in the rate per year). "
        "A forecast extrapolates the learning period, so a change in slope in the test window "
        "is forecast risk that no in-sample fit can reveal."
    )
    rows = []
    for s in SEXES:
        for band, (lo, hi) in AGE_BANDS.items():
            x = d[(d.Sex == s) & d.Age.between(lo, hi)]
            rows.append(
                {
                    "sex": s,
                    "ages": band,
                    f"learn_{first}_{oy}": _trend(x, first, oy),
                    f"learn_{oy - 29}_{oy}": _trend(x, oy - 29, oy),
                    f"test_{oy + 1}_{last}": _trend(x, oy + 1, last),
                }
            )
    r.table(pd.DataFrame(rows), "trend_by_period")

    r.h("Mortality shocks")
    r.note(
        "Largest one-year rises in all-age average log m. Shock years are the observations a "
        "smooth trend model cannot fit, and they distort a fit on the years around them."
    )
    avg = d.groupby(["Sex", "Year"]).logmx.mean().unstack("Sex")
    rise = avg.diff()
    shock = pd.concat(
        rise[s].nlargest(5).rename("rise_in_mean_log_m").reset_index().assign(sex=s) for s in SEXES
    )
    r.table(
        shock.assign(Year=shock.Year.astype(str))[["sex", "Year", "rise_in_mean_log_m"]], "shocks"
    )

    r.h("One-factor structure (Lee-Carter)")
    r.note(
        "Share of variance in the learn-period, age-centred log m explained by each singular "
        "vector. Lee-Carter keeps only the first."
    )
    rows = []
    for s in SEXES:
        m = piv[s].loc[:, piv[s].columns <= oy].to_numpy()
        sv = np.linalg.svd(m - m.mean(axis=1, keepdims=True), compute_uv=False)
        share = sv**2 / (sv**2).sum()
        rows.append({"sex": s, **{f"component_{i + 1}": float(share[i]) for i in range(5)}})
    r.table(pd.DataFrame(rows), "svd_variance_share")
    return r.close()


# --- Reserving: CAS Schedule P ---------------------------------------------------------------

LINES = ["ppauto", "wkcomp", "comauto", "medmal", "prodliab", "othliab"]


def _quality(tri: pd.DataFrame, setup: str) -> pd.DataFrame:
    rows = []
    for lob, x in tri.groupby("lob"):
        u = x[x.upper]
        rows.append(
            {
                "setup": setup,
                "line": lob,
                "companies": x.company.nunique(),
                "cells": len(x),
                "upper_cells": int(x.upper.sum()),
                "missing": int(x[["cum_paid", "premium", "incr_paid"]].isna().sum().sum()),
                "premium_le_0": int((x.premium <= 0).sum()),
                "cum_paid_at_floor": int((x.cum_paid <= 1).sum()),
                "negative_incr_share_upper": float((u.incr_paid < 0).mean()),
                "zero_incr_share_upper": float((u.incr_paid == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _age_to_age(tri: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted age-to-age factors, using the upper triangle only."""
    u = tri[tri.upper][["lob", "company", "ay", "dev", "cum_paid"]]
    nxt = u.assign(dev=u.dev - 1).rename(columns={"cum_paid": "next"})
    g = u.merge(nxt, on=["lob", "company", "ay", "dev"]).groupby(["lob", "dev"])
    return (g.next.sum() / g.cum_paid.sum()).unstack("lob")


def reserving_eda(raw: pd.DataFrame, base: dict, up: dict) -> Path:
    ref = rm.prepare(raw, base["reserving"]["data"])
    tri = rm.prepare(raw, up["reserving"]["data"])
    lines = [x for x in LINES if x in tri.lob.unique()]
    r = Report(
        "reserving",
        "EDA: CAS Schedule P paid-loss triangles",
        "Used by `chain_ladder` and `neural_odp`. Accident years 1988-1997, development lags "
        "1-10. Models see only the upper triangle (calendar year up to 1997); the lower "
        "triangle is the realised outcome held back for testing. The reference setup is "
        f"{ref.company.nunique()} private passenger auto companies (Meyers 2015); scale_up uses "
        "every company in all six lines that passes the data checks. Anything computed from "
        "the lower triangle is labelled *realised* and must not inform modelling choices.",
    )

    r.h("Which companies reach the models")
    raw_g = raw.groupby(["lob", "GRCODE"]).size()
    r.table(
        pd.DataFrame(
            [
                {
                    "line": lob,
                    "raw_companies": int(raw_g[lob].size),
                    "complete_100_cells": int((raw_g[lob] == 100).sum()),
                    "scale_up_selected": tri[tri.lob == lob].company.nunique(),
                    "reference_selected": ref[ref.lob == lob].company.nunique(),
                }
                for lob in LINES
            ]
        ),
        "selection",
    )

    r.h("Data quality")
    r.table(pd.concat([_quality(ref, "reference"), _quality(tri, "scale_up")]), "quality")
    r.note(
        "Duplicated rows in the raw file: "
        f"{int(raw.duplicated(['lob', 'GRCODE', 'AccidentYear', 'DevelopmentLag']).sum()):,}. "
        "`cum_paid_at_floor` counts values at 1 or less, where `prepare` applies "
        "Meyers' floor of 1."
    )

    r.h("Company size")
    size = tri[tri.dev == 1].groupby(["lob", "company"]).premium.sum()
    r.table(
        size.groupby("lob")
        .describe(percentiles=[0.5])
        .loc[lines]
        .rename_axis("line")
        .reset_index()
        .rename(columns={"50%": "median"})[["line", "count", "min", "median", "mean", "max"]],
        "company_premium",
    )
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.boxplot(
        [size[x].to_numpy() for x in lines],
        tick_labels=lines,
        widths=0.5,
        medianprops={"color": P[0], "lw": 2},
        flierprops={"ms": 3, "mec": INK},
    )
    ax.set_yscale("log")
    ax.set(title="Net earned premium per company, 1988-1997", ylabel="premium (log scale)")
    r.fig(fig, "company_size", "Company size by line (scale_up companies)")

    r.h("Development")
    a2a = _age_to_age(tri)[lines]
    r.table(a2a.rename_axis("dev").reset_index().rename(columns={"dev": "from_dev"}), "age_to_age")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, x in enumerate(lines):
        ax.plot(a2a.index, a2a[x], color=P[i], marker=MARKERS[i], ms=5, lw=1.6, label=x)
    ax.set_yscale("log")
    ax.set(
        title="Age-to-age factors, upper triangle",
        xlabel="from development lag",
        ylabel="volume-weighted factor (log scale)",
    )
    ax.legend(ncols=2)
    r.fig(fig, "age_to_age", "Chain-ladder development factors by line (no realised data)")

    g = tri.groupby(["lob", "dev"]).cum_paid.sum().unstack("lob")[lines]
    paid = g / g.loc[10]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, x in enumerate(lines):
        ax.plot(paid.index, paid[x], color=P[i], marker=MARKERS[i], ms=5, lw=1.6, label=x)
    ax.set(
        title="Cumulative paid as a share of lag-10 paid (realised)",
        xlabel="development lag",
        ylabel="share of lag-10 paid",
    )
    ax.legend()
    r.fig(fig, "paid_pattern", "Payment speed by line (realised, includes the lower triangle)")
    paid.rename_axis("dev").reset_index().to_csv(r.dir / "paid_pattern.csv", index=False)

    r.note(
        "Incremental paid over premium by accident year and lag; the step separates what "
        "models see (above-left) from the realised lower triangle."
    )
    s = tri.groupby(["lob", "ay", "dev"])[["incr_paid", "premium"]].sum()
    s = (s.incr_paid / s.premium).rename("r").reset_index()
    fig, axes = grid(len(lines), 3, 4.2, 3.2)
    vmax = s.r.quantile(0.98)
    for ax, x in zip(axes, lines):
        m = s[s.lob == x].pivot(index="ay", columns="dev", values="r")
        im = ax.imshow(
            m.to_numpy(),
            cmap=SEQ,
            vmin=0,
            vmax=vmax,
            aspect="auto",
            extent=[0.5, 10.5, m.index.max() + 0.5, m.index.min() - 0.5],
        )
        ax.grid(False)
        for ay in m.index:
            b = 1998.5 - ay
            ax.plot([b, b], [ay - 0.5, ay + 0.5], color=INK, lw=1.4)
            if ay < m.index.max():
                ax.plot([b - 1, b], [ay + 0.5, ay + 0.5], color=INK, lw=1.4)
        ax.set(title=x, xlabel="development lag", ylabel="accident year")
    fig.colorbar(im, ax=list(axes), label="incremental paid / premium", shrink=0.8)
    r.fig(fig, "triangles", "Incremental paid / premium (realised cells below the step)")

    r.h("Loss ratios (realised)")
    lag10 = tri[tri.dev == 10]
    ay = lag10.groupby(["lob", "ay"])[["cum_paid", "premium"]].sum()
    lr = (ay.cum_paid / ay.premium).unstack("lob")[lines]
    lr.rename_axis("ay").reset_index().to_csv(r.dir / "loss_ratio_by_ay.csv", index=False)
    co = lag10.groupby(["lob", "company"])[["cum_paid", "premium"]].sum()
    co_lr = co.cum_paid / co.premium
    fig, axes = grid(2, 2, 5.2, 3.6)
    for i, x in enumerate(lines):
        axes[0].plot(lr.index, lr[x], color=P[i], marker=MARKERS[i], ms=5, lw=1.6, label=x)
    axes[0].set(
        title="Lag-10 paid loss ratio by accident year",
        xlabel="accident year",
        ylabel="paid / premium",
    )
    axes[0].legend(ncols=2, fontsize=8)
    axes[1].boxplot(
        [co_lr[x].to_numpy() for x in lines],
        tick_labels=lines,
        widths=0.5,
        medianprops={"color": P[0], "lw": 2},
        flierprops={"ms": 3, "mec": INK},
    )
    axes[1].set(title="Lag-10 paid loss ratio per company", ylabel="paid / premium")
    axes[1].tick_params(axis="x", labelsize=8)
    r.fig(fig, "loss_ratios", "Realised loss ratios: over accident years and across companies")
    return r.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="frequency, mortality, reserving (default: all)")
    all_datasets = ["frequency", "mortality", "reserving"]
    wanted = ap.parse_args().datasets or all_datasets
    if unknown := set(wanted) - set(all_datasets):
        ap.error(f"unknown dataset: {', '.join(sorted(unknown))}")

    base, up = load_params("local"), load_params("scale_up")
    freq_raw, mort_raw = fetch_aitools(base["ingest"]["aitools"])
    reports = {}
    if "frequency" in wanted:
        reports["frequency"] = frequency_eda(freq_raw, base)
    if "mortality" in wanted:
        reports["mortality"] = mortality_eda(mort_raw, base, up)
    if "reserving" in wanted:
        reports["reserving"] = reserving_eda(fetch_cas(base["ingest"]["cas"]), base, up)

    index = OUT / "README.md"
    previous = sorted(p.parent.name for p in OUT.glob("*/report.md"))
    index.write_text(
        "# Exploratory data analysis\n\nGenerated by `scripts/eda.py`. Local only: not logged to "
        "MLflow or W&B.\n\n" + "".join(f"- [{n}]({n}/report.md)\n" for n in previous)
    )
    for n, p in reports.items():
        log.info("%s: %s", n, p.relative_to(ROOT))
    log.info("index: %s", index.relative_to(ROOT))


if __name__ == "__main__":
    main()
