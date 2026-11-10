#!/usr/bin/env python3
"""Model analysis: how each fitted model behaves, beyond its benchmark PASS/FAIL. Local only.

    python scripts/model_analysis.py                       # all three domains
    python scripts/model_analysis.py frequency             # a subset

Reads the saved predictions in `data/07_model_output` and writes
`data/11_model_analysis/<domain>/report.md` with PNG figures and CSV tables (git-ignored).
Nothing starts a Kedro session, so nothing is logged to MLflow or W&B.

Run it after a full run, not during one: `data/07_model_output` is rewritten by each pipeline.
It analyses the most recent run of each pipeline. After `scripts/full_run.sh` that is the
reference setup for frequency and the scale_up setup for mortality and reserving, whose reports
say so. Published-benchmark comparisons come from `conf/base/parameters_benchmarks.yml`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import eda  # scripts/eda.py: report writer, figure style and feature bins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eda import DIV, INK, MARKERS, ROOT, SEQ, P, Report, grid, load_params

from deep_actuarial_science.metrics import (
    balance_ratio,
    gini_index,
    poisson_deviance_per_policy,
    rmse,
)
from deep_actuarial_science.models import reserving as rm
from deep_actuarial_science.pipelines.frequency import MODELS as FREQUENCY_MODELS
from deep_actuarial_science.pipelines.mortality import MODELS as MORTALITY_MODELS
from deep_actuarial_science.pipelines.reserving import MODELS as RESERVING_MODELS

log = logging.getLogger("model_analysis")
INPUT = ROOT / "data" / "07_model_output"
OUT = ROOT / "data" / "11_model_analysis"
SEGMENTS = ["DrivAge", "BonusMalus", "VehAge", "Area", "VehGas", "VehBrand", "Region"]
HORIZONS = {"2000-04": (2000, 2004), "2005-09": (2005, 2009), "2010-14": (2010, 2014),
            "2015-19": (2015, 2019), "2020-21": (2020, 2021)}  # fmt: skip


def seed_columns(d: pd.DataFrame) -> list[str]:
    return sorted((c for c in d.columns if c.startswith("pred_run_")), key=lambda c: int(c[9:]))


def annotated(ax, m: pd.DataFrame, fmt: str, threshold: float, **kw):
    """Heatmap of a table, with each cell's value printed."""
    im = ax.imshow(m.to_numpy(dtype=float), aspect="auto", **kw)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                color="white" if abs(v) > threshold else "#333333")  # fmt: skip
    return im


def strip(ax, rows: list[dict], metric: str) -> None:
    """Per-seed results as dots, with the ensemble and the published value marked."""
    rng = np.random.default_rng(0)  # jitter only
    for i, row in enumerate(rows):
        s = np.asarray(row["seeds"])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, s=22, color=P[0], alpha=0.7,
                   label="each seed" if i == 0 else None)  # fmt: skip
        ax.scatter(i, row["ensemble"], s=70, color=P[1], marker="D",
                   label="seed-averaged forecast" if i == 0 else None, zorder=3)  # fmt: skip
        if row.get("published") is not None:
            ax.hlines(row["published"], i - 0.3, i + 0.3, color=INK, lw=2.5, zorder=3,
                      label="published" if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(rows)), [r["model"] for r in rows])
    ax.set_ylabel(metric)
    ax.legend()


# --- Frequency -------------------------------------------------------------------------------


def frequency_analysis(base: dict) -> Path:
    models = list(FREQUENCY_MODELS)
    preds = {m: pd.read_parquet(INPUT / "frequency" / f"{m}.parquet") for m in models}
    published = base["benchmarks"]["frequency"]
    r = Report(
        "frequency",
        "Model analysis: claim frequency",
        "Reference setup: freMTPL2, fixed learn/test split. Deviances are per policy in "
        "10^-2 (Wüthrich & Merz 2023, eq. 5.28). `homogeneous` is the no-features baseline.",
        root=OUT,
    )

    def dev(d: pd.DataFrame, col: str = "pred") -> float:
        return poisson_deviance_per_policy(d.ClaimNb, d[col], d.Exposure)

    null_test = dev(preds["homogeneous"].query("split == 'test'"))
    rows = []
    for m, d in preds.items():
        le, te = d[d.split == "learn"], d[d.split == "test"]
        rate = te.pred / te.Exposure
        rows.append(
            {
                "model": m,
                "deviance_learn": dev(le),
                "deviance_test": dev(te),
                "test_minus_learn": dev(te) - dev(le),
                "deviance_explained_test_pct": 100 * (1 - dev(te) / null_test),
                "balance_learn": balance_ratio(le.ClaimNb, le.pred),
                "balance_test": balance_ratio(te.ClaimNb, te.pred),
                "gini_test": np.nan  # a constant rate has no ranking
                if m == "homogeneous"
                else gini_index(te.ClaimNb, te.pred, te.Exposure),
                "rate_p01": float(rate.quantile(0.01)),
                "rate_p99": float(rate.quantile(0.99)),
            }
        )
    r.h("Scorecard")
    r.note(
        "Multi-seed networks are scored here on their seed-averaged prediction, which is why "
        "they differ from the per-seed means reconciled to the papers (see the next section). "
        "`balance` is predicted over observed claims. `rate_p01`/`rate_p99` are the 1st and "
        "99th percentiles of predicted claims per year on the test set."
    )
    r.table(pd.DataFrame(rows).sort_values("deviance_test"), "scorecard")

    r.h("Stability across seeds")
    strips, rows = [], []
    for m, d in preds.items():
        cols = seed_columns(d)
        if not cols:
            continue
        te = d[d.split == "test"]
        seeds = [dev(te, c) for c in cols]
        spread = te[cols].std(axis=1, ddof=1) / te[cols].mean(axis=1)
        rows.append(
            {
                "model": m,
                "seeds": len(cols),
                "test_mean": float(np.mean(seeds)),
                "test_sd": float(np.std(seeds, ddof=1)),
                "test_best": float(np.min(seeds)),
                "test_worst": float(np.max(seeds)),
                "test_ensemble": dev(te),
                "ensemble_gain": float(np.mean(seeds)) - dev(te),
                "policy_cv_median": float(spread.median()),
                "policy_cv_p99": float(spread.quantile(0.99)),
            }
        )
        expected = (published.get(m) or {}).get("expected", {})
        strips.append(
            {
                "model": m,
                "seeds": seeds,
                "ensemble": dev(te),
                "published": expected.get("deviance_test"),
            }  # fmt: skip
        )
    r.note(
        "`ensemble_gain` is how much test deviance averaging the seeds removes. "
        "`policy_cv` is the coefficient of variation of one policy's prediction across seeds: "
        "the disagreement a single-seed model would hide."
    )
    r.table(pd.DataFrame(rows), "seed_stability")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    strip(ax, strips, "test deviance (10^-2)")
    ax.set_title("Test deviance per seed")
    r.fig(fig, "seed_stability", "Per-seed test deviance against the published mean")

    r.h("Calibration across models")
    fitted = [m for m in models if m != "homogeneous"]  # a constant has no deciles
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, m in enumerate(fitted):
        te = preds[m].query("split == 'test'")
        bin_ = pd.qcut((te.pred / te.Exposure).rank(method="first"), 10, labels=False) + 1
        g = te.groupby(bin_)[["ClaimNb", "pred"]].sum()
        ax.plot(g.index, g.ClaimNb / g.pred, color=P[i % len(P)],
                marker=MARKERS[i % len(MARKERS)], ms=5, lw=1.5, label=m)  # fmt: skip
    ax.axhline(1, color=INK, lw=1)
    ax.set(title="Actual / predicted by decile of predicted rate, test",
           ylabel="actual / predicted", xlabel="predicted-rate decile")  # fmt: skip
    ax.set_xticks(range(1, 11))
    ax.legend(ncols=2, fontsize=8)
    r.fig(fig, "calibration_by_decile", "Auto-calibration of every fitted model on the test set")
    claims = float(preds["glm3"].query("split == 'test'").ClaimNb.sum())
    r.note(
        f"Each decile holds about {claims / 10:,.0f} claims, so Poisson noise alone moves a "
        f"point by about ±{200 / np.sqrt(claims / 10):.0f}% (2 sd): read the overall shape, "
        "not single points."
    )

    r.h("Accuracy by segment")
    r.note(
        "Exposure-weighted mean of |actual/predicted − 1| over the levels of each feature "
        "holding at least 1% of test exposure. The `noise_floor` row is what a perfect model "
        "would score from Poisson noise alone, so a value close to it means there is no "
        "detectable segment bias."
    )
    seg, floors = {}, {}
    for m, d in preds.items():
        te = d[d.split == "test"]
        scores = {}
        for f in SEGMENTS:
            g = te.groupby(eda._bin(te[f], f), observed=True)[["ClaimNb", "pred", "Exposure"]].sum()
            g = g[g.Exposure / g.Exposure.sum() >= 0.01]
            share = g.Exposure / g.Exposure.sum()
            scores[f] = float((share * (g.ClaimNb / g.pred - 1).abs()).sum())
            if m == "glm3":  # E|A/E - 1| for a Poisson count, normal approximation
                floors[f] = float((share * np.sqrt(2 / (np.pi * g.pred))).sum())
        seg[m] = scores
    table = pd.DataFrame(seg).T[SEGMENTS]
    table.loc["noise_floor"] = pd.Series(floors)
    r.table(table.rename_axis("model").reset_index(), "segment_error")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    im = annotated(ax, 100 * table, ".1f", 100 * table.to_numpy().max() * 0.6, cmap=SEQ, vmin=0)
    fig.colorbar(im, ax=ax, label="mean |A/E − 1| (%)")
    ax.set_title("Segment error on the test set")
    r.fig(fig, "segment_error", "Exposure-weighted mean |A/E − 1| by model and feature (%)")

    r.h("Where models disagree")
    rates = pd.DataFrame({m: preds[m].query("split == 'test'").pred.to_numpy() for m in fitted})
    logs = np.log(rates)
    gap = pd.DataFrame(
        [[float((logs[a] - logs[b]).abs().mean()) for b in fitted] for a in fitted],
        index=fitted, columns=fitted,
    )  # fmt: skip
    gap.round(4).to_csv(r.dir / "pairwise_disagreement.csv")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = annotated(ax, 100 * gap, ".1f", 100 * gap.to_numpy().max() * 0.6, cmap=SEQ, vmin=0)
    fig.colorbar(im, ax=ax, label="mean |log ratio| (%)")
    ax.set_title("Pairwise difference in predicted rates, test")
    r.fig(fig, "pairwise_disagreement", "Mean absolute log ratio between models' test predictions")
    return r.close()


# --- Mortality -------------------------------------------------------------------------------


def mortality_analysis(base: dict) -> Path:
    res = {m: pd.read_parquet(INPUT / "mortality" / f"{m}.parquet") for m in MORTALITY_MODELS}
    data, train = base["mortality"]["data"], base["mortality"]["training"]
    first_in = data["first_year"] + train["lookback"]
    published = base["benchmarks"]["mortality"]
    sexes = sorted(next(iter(res.values())).Sex.unique(), reverse=True)  # Male first
    r = Report(
        "mortality",
        "Model analysis: Swiss mortality forecasts",
        f"The latest run's forecasts ({', '.join(sexes)}), each sex fitted separately. RMSE of "
        f"log m(x,t): in-sample from {first_in}, out-of-sample after {data['obs_year']}. "
        "Published values (Richman, Scognamiglio & Wüthrich 2026, Ch. 8) are for Swiss males.",
        root=OUT,
    )

    rows, strips = [], []
    lc_out = {}
    for m, d in res.items():
        for s in sexes:
            x = d[d.Sex == s]
            ins, out = x[(x.split == "learn") & (x.Year >= first_in)], x[x.split == "test"]
            cols = seed_columns(x) or ["pred"]
            out_seeds = [rmse(out.logmx, out[c]) for c in cols]
            pre_covid = out[out.Year < 2020]
            row = {
                "sex": s,
                "model": m,
                "seeds": len(cols),
                "rmse_in": float(np.mean([rmse(ins.logmx, ins[c]) for c in cols])),
                "rmse_out": float(np.mean(out_seeds)),
                "rmse_out_sd": float(np.std(out_seeds, ddof=1)) if len(cols) > 1 else 0.0,
                "rmse_out_best": float(np.min(out_seeds)),
                "rmse_out_worst": float(np.max(out_seeds)),
                "rmse_out_ensemble": rmse(out.logmx, out.pred),
                "rmse_out_before_2020": float(
                    np.mean([rmse(pre_covid.logmx, pre_covid[c]) for c in cols])
                ),
            }
            row["out_over_in"] = row["rmse_out"] / row["rmse_in"]
            if m == "lee_carter":
                lc_out[s] = row["rmse_out"]
            rows.append(row)
            if s == "Male" and len(cols) > 1:
                strips.append(
                    {
                        "model": m,
                        "seeds": out_seeds,
                        "ensemble": row["rmse_out_ensemble"],
                        "published": (published.get(m) or {}).get("expected", {}).get("rmse_out"),
                    }  # fmt: skip
                )
    score = pd.DataFrame(rows)
    score["vs_lee_carter_pct"] = 100 * (score.rmse_out / score.sex.map(lc_out) - 1)
    r.h("Scorecard")
    r.note(
        "`rmse_in`, `rmse_out` and `rmse_out_before_2020` are per-seed means for networks. "
        "`out_over_in` above 1 is the usual forecast deterioration; a large value signals "
        "overfitting or a change in trend. `vs_lee_carter_pct` is negative when a model beats "
        "Lee-Carter out of sample."
    )
    r.table(score, "scorecard")

    r.h("Stability across seeds")
    if strips:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        strip(ax, strips, "RMSE of log m, 2000-2021")
        ax.set_title("Male: out-of-sample RMSE per seed")
        r.fig(fig, "seed_stability", "Per-seed out-of-sample RMSE against the book's single run")
        r.note(
            "The book publishes one run per network, so a published value can sit anywhere in "
            "this spread and still be consistent with it."
        )

    r.h("Errors by age")
    r.note("Mean signed error of the forecast, seed-averaged, 2000-2021. Positive means the "
           "model forecasts higher mortality than was observed.")  # fmt: skip
    bias, spread = {}, {}
    for m, d in res.items():
        for s in sexes:
            x = d[(d.Sex == s) & (d.split == "test")]
            b = {}
            e = {}
            for band, (lo, hi) in eda.AGE_BANDS.items():
                y = x[x.Age.between(lo, hi)]
                b[band], e[band] = float((y.pred - y.logmx).mean()), rmse(y.logmx, y.pred)
            bias[(s, m)], spread[(s, m)] = b, e
    bias_t = pd.DataFrame(bias).T.rename_axis(["sex", "model"])
    rmse_t = pd.DataFrame(spread).T.rename_axis(["sex", "model"])
    r.table(bias_t.reset_index(), "bias_by_age")
    rmse_t.reset_index().to_csv(r.dir / "rmse_by_age.csv", index=False)
    fig, axes = grid(len(sexes), 2, 4.8, 3.2)
    lim = float(bias_t.abs().to_numpy().max())
    for ax, s in zip(axes, sexes):
        im = annotated(ax, bias_t.loc[s], ".2f", lim * 0.6, cmap=DIV, vmin=-lim, vmax=lim)
        ax.set_title(f"{s}: mean forecast error")
    fig.colorbar(im, ax=axes, label="forecast − observed log m", shrink=0.9)
    r.fig(fig, "bias_by_age", "Mean signed forecast error by age band, 2000-2021")

    r.h("Errors by forecast horizon")
    rows = []
    for m, d in res.items():
        for s in sexes:
            x = d[(d.Sex == s) & (d.split == "test")]
            rows.append(
                {
                    "sex": s,
                    "model": m,
                    **{
                        h: rmse(x[x.Year.between(*yr)].logmx, x[x.Year.between(*yr)].pred)
                        for h, yr in HORIZONS.items()
                    },
                }  # fmt: skip
            )
    r.note("RMSE of the seed-averaged forecast. 2020-21 includes the COVID-19 years.")
    r.table(pd.DataFrame(rows), "rmse_by_horizon")
    return r.close()


# --- Reserving -------------------------------------------------------------------------------


def reserving_analysis(base: dict) -> Path:
    preds = {m: pd.read_parquet(INPUT / "reserving" / f"{m}.parquet") for m in RESERVING_MODELS}
    lines = [x for x in eda.LINES if x in next(iter(preds.values())).lob.unique()]
    n = next(iter(preds.values())).groupby("lob").company.nunique().sum()
    r = Report(
        "reserving",
        "Model analysis: paid-loss reserves",
        f"Back-test of the latest run ({n} company-line triangles across {', '.join(lines)}) "
        "against the realised lower triangle. `neural_odp` fits all triangles of a run jointly, "
        "so after a scale_up run its results differ from the 50-company reference fit that the "
        "reconciliation table reports.",
        root=OUT,
    )
    frames = []
    for m, d in preds.items():
        rs = rm.reserves(d).merge(rm.ultimates(d), on=rm.KEY)
        rs["model"] = m
        frames.append(rs)
    a = pd.concat(frames, ignore_index=True)
    a["reserve_error"] = a.predicted_reserve - a.actual_reserve
    a["ultimate_pe"] = a.predicted_ultimate / a.actual_ultimate - 1

    def summary(x: pd.DataFrame) -> pd.Series:
        pos = x[x.actual_reserve > 0]
        rel = pos.reserve_error / pos.actual_reserve
        return pd.Series(
            {
                "companies": len(x),
                "mape_ultimate": x.ultimate_pe.abs().mean(),
                "rmspe_ultimate": np.sqrt((x.ultimate_pe**2).mean()),
                "total_reserve_error_pct": 100
                * (x.predicted_reserve.sum() / x.actual_reserve.sum() - 1),
                "median_abs_reserve_error_pct": 100 * rel.abs().median(),
                "share_within_10pct": (rel.abs() <= 0.10).mean(),
            }
        )

    r.h("Scorecard by line")
    r.note(
        "`mape_ultimate` and `rmspe_ultimate` are unweighted percentage errors of total ultimate "
        "loss over companies (Kuo 2019). Reserve errors use companies with a positive realised "
        "reserve; a total close to 0 can hide offsetting company errors."
    )
    order = {x: i for i, x in enumerate(lines)}
    byline = a.groupby(["lob", "model"]).apply(summary, include_groups=False).reset_index()
    byline = byline.sort_values(
        ["lob", "model"], key=lambda s: s.map(order) if s.name == "lob" else s
    )
    r.table(byline.astype({"companies": int}).rename(columns={"lob": "line"}), "scorecard_by_line")

    r.h("Does accuracy depend on company size?")
    labels = ["small", "mid", "large"]
    company = a.drop_duplicates(rm.KEY).set_index(rm.KEY).premium  # one row per company
    bands = company.groupby("lob").transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=labels).astype(str)
    )
    a["size_band"] = pd.Categorical(
        pd.MultiIndex.from_frame(a[rm.KEY]).map(bands), labels, ordered=True
    )
    r.note("Size bands are premium terciles within each line.")
    by_size = (
        a.groupby(["model", "size_band"], observed=True).apply(summary, include_groups=False)
        .reset_index()[["model", "size_band", "companies", "mape_ultimate",
                        "median_abs_reserve_error_pct"]]
    )  # fmt: skip
    r.table(by_size.astype({"companies": int}), "accuracy_by_size")

    r.h("Where in the triangle the errors are")
    r.note("Signed error of the projected lower triangle, summed over companies, as a share of "
           "premium. Positive means over-reserved; a diagonal band would suggest a calendar-year "
           "effect that development models do not capture.")  # fmt: skip
    fig, axes = grid(len(preds), 2, 5, 3.6)
    cells = {}
    for m, d in preds.items():
        low = d[~d.upper]
        g = low.groupby(["ay", "dev"]).agg(pred=("pred", "sum"), incr=("incr_paid", "sum"))
        prem = d[d.dev == 1].groupby("ay").premium.sum()
        err = (g.pred - g.incr) / g.index.get_level_values("ay").map(prem)
        cells[m] = err.unstack("dev")
        err.rename("error_share_of_premium").reset_index().to_csv(
            r.dir / f"triangle_error_{m}.csv", index=False
        )
    lim = float(max(c.abs().max().max() for c in cells.values()))
    for ax, (m, c) in zip(axes, cells.items()):
        c = c.reindex(index=range(1988, 1998), columns=range(1, 11))
        im = ax.imshow(c.to_numpy(), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                       extent=[0.5, 10.5, 1997.5, 1987.5])  # fmt: skip
        ax.grid(False)
        ax.set(title=m, xlabel="development lag", ylabel="accident year")
    fig.colorbar(im, ax=axes, label="(predicted − realised) / premium", shrink=0.9)
    r.fig(fig, "triangle_error", "Signed reserve error by accident year and lag")

    r.h("Largest misses")
    a["error_share_of_premium"] = a.reserve_error / a.premium
    worst = (
        a.assign(abs_err=a.error_share_of_premium.abs())
        .sort_values("abs_err", ascending=False)
        .groupby("model", sort=False).head(5)
        .sort_values(["model", "abs_err"], ascending=[True, False])
    )  # fmt: skip
    r.table(
        worst[["model", "lob", "company", "premium", "predicted_reserve", "actual_reserve",
               "error_share_of_premium"]].rename(columns={"lob": "line"}),
        "largest_misses",
    )  # fmt: skip
    return r.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("domains", nargs="*", help="frequency, mortality, reserving (default: all)")
    all_domains = {
        "frequency": frequency_analysis,
        "mortality": mortality_analysis,
        "reserving": reserving_analysis,
    }
    wanted = ap.parse_args().domains or list(all_domains)
    if unknown := set(wanted) - set(all_domains):
        ap.error(f"unknown domain: {', '.join(sorted(unknown))}")
    base = load_params("local")
    for name in wanted:
        log.info("%s: %s", name, all_domains[name](base).relative_to(ROOT))
    done = sorted(p.parent.name for p in OUT.glob("*/report.md"))
    (OUT / "README.md").write_text(
        "# Model analysis\n\nGenerated by `scripts/model_analysis.py`. Local only: not logged to "
        "MLflow or W&B.\n\n" + "".join(f"- [{n}]({n}/report.md)\n" for n in done)
    )


if __name__ == "__main__":
    main()
