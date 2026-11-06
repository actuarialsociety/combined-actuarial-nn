"""Diagnostic figures. Every function returns a matplotlib Figure (Kedro saves it,
kedro-mlflow logs it, the lessons embed it)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelcolor": INK,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 10,
        "legend.frameon": False,
    }
)


def named(figs: dict) -> dict:
    """Kedro's MatplotlibDataset writes dict keys as file names: add the extension."""
    return {k if k.endswith(".png") else f"{k}.png": v for k, v in figs.items()}


def _fig(w=6.4, h=4.0, ncols=1):
    return plt.subplots(1, ncols, figsize=(w, h), squeeze=False)


# --- Frequency ------------------------------------------------------------------------------


def calibration(pred: pd.DataFrame, bins: int = 10, title: str = "") -> plt.Figure:
    """Actual vs predicted frequency by predicted-frequency decile (test set): auto-calibration."""
    d = pred.assign(f=pred.pred / pred.Exposure)
    d["bin"] = pd.qcut(d.f.rank(method="first"), bins, labels=False) + 1
    g = d.groupby("bin").agg(y=("ClaimNb", "sum"), mu=("pred", "sum"), v=("Exposure", "sum"))
    fig, ax = _fig()
    ax = ax[0, 0]
    ax.plot(g.index, g.y / g.v, "o-", color=INK, label="actual")
    ax.plot(g.index, g.mu / g.v, "s--", color=PALETTE[0], label="predicted")
    ax.set(
        yscale="log",
        xlabel="predicted-frequency decile",
        ylabel="claim frequency",
        title=title or "Calibration by decile (test)",
    )
    ax.legend()
    return fig


def lorenz(
    preds: dict[str, pd.DataFrame], title: str = "Ordered Lorenz curve (test)"
) -> plt.Figure:
    fig, ax = _fig(5, 4.5)
    ax = ax[0, 0]
    for i, (name, p) in enumerate(preds.items()):
        o = np.argsort(-(p.pred / p.Exposure).to_numpy(), kind="stable")
        cv = np.r_[0, np.cumsum(p.Exposure.to_numpy()[o]) / p.Exposure.sum()]
        cy = np.r_[0, np.cumsum(p.ClaimNb.to_numpy()[o]) / p.ClaimNb.sum()]
        ax.plot(cv, cy, color=PALETTE[i % len(PALETTE)], label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set(xlabel="share of exposure (riskiest first)", ylabel="share of claims", title=title)
    ax.legend()
    return fig


def marginal(pred: pd.DataFrame, feature: str, title: str = "") -> plt.Figure:
    """Exposure-weighted actual vs predicted frequency by a covariate, with exposure bars."""
    g = pred.groupby(feature, observed=True).agg(
        y=("ClaimNb", "sum"), mu=("pred", "sum"), v=("Exposure", "sum")
    )
    g = g[g.v > g.v.sum() * 0.001]
    fig, ax = _fig()
    ax = ax[0, 0]
    numeric = pd.api.types.is_numeric_dtype(g.index.dtype)
    x = g.index.to_numpy() if numeric else np.arange(len(g))
    ax2 = ax.twinx()
    ax2.bar(x, g.v, color=GRID, width=0.8 if len(g) < 30 else 1.0, zorder=0)
    ax2.set(ylabel="exposure", yticks=[])
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(x, g.y / g.v, "o", ms=3, color=INK, label="actual")
    ax.plot(x, g.mu / g.v, "-", color=PALETTE[1], lw=2, label="predicted")
    if not numeric:
        ax.set_xticks(x, g.index.astype(str), rotation=90 if len(g) > 12 else 0)
    ax.set(xlabel=feature, ylabel="claim frequency", title=title or f"Marginal fit: {feature}")
    ax.legend(loc="upper center")
    return fig


def training(history: dict | None, title: str = "Training history") -> plt.Figure:
    fig, ax = _fig()
    ax = ax[0, 0]
    if not history:
        ax.text(
            0.5,
            0.5,
            "closed-form fit: no training history",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.axvline(
        history.get("best_epoch", np.argmin(history["val_loss"])),
        color=MUTED,
        ls="--",
        label="early-stopping point",
    )
    ax.set(xlabel="epoch / boosting round", ylabel="loss", title=title)
    ax.legend()
    return fig


def leaderboard(
    table: pd.DataFrame,
    metric: str,
    bench_col: str | None = None,
    title: str = "",
    lower_is_better: bool = True,
) -> plt.Figure:
    t = table.sort_values(metric, ascending=not lower_is_better)
    fig, ax = _fig(6.4, 0.45 * len(t) + 1.2)
    ax = ax[0, 0]
    y = np.arange(len(t))
    ax.barh(y, t[metric], color=PALETTE[0], height=0.6, label="this run")
    if bench_col and bench_col in t:
        ax.plot(t[bench_col], y, "D", color=PALETTE[1], ms=7, label="published benchmark")
    lo = np.nanmin(t[[c for c in (metric, bench_col) if c and c in t]].to_numpy())
    hi = np.nanmax(t[[c for c in (metric, bench_col) if c and c in t]].to_numpy())
    pad = (hi - lo) * 0.15 or abs(hi) * 0.01
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(y, t["model"])
    for yi, v in zip(y, t[metric]):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=9, color=INK)
    ax.set(xlabel=metric, title=title or f"Leaderboard: {metric}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return fig


def attention(
    att: pd.DataFrame, X: pd.DataFrame, features: list[str], n: int = 5000, seed: int = 0
) -> plt.Figure:
    """LocalGLMnet attention weights beta_j(x) against x_j; noise band from RandN."""
    s = att.sample(min(n, len(att)), random_state=seed)
    xs = X.loc[s.index]
    band = 2.576 * att["RandN"].std() if "RandN" in att else None  # 99% band
    k = len(features)
    cols = 4
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, f in zip(axes.flat, features):
        ax.scatter(xs[f], s[f], s=2, alpha=0.3, color=PALETTE[0])
        if band is not None:
            ax.axhspan(-band, band, color=PALETTE[1], alpha=0.08)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f, fontsize=9)
    for ax in list(axes.flat)[k:]:
        ax.set_axis_off()
    fig.suptitle(
        r"LocalGLMnet attention weights $\beta_j(x)$ (shaded: 99% noise band)", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def importance(
    att: pd.DataFrame, title: str = "Variable importance: mean |beta_j(x)|"
) -> plt.Figure:
    imp = att.abs().mean().sort_values()
    fig, ax = _fig(5.5, 0.35 * len(imp) + 1)
    ax = ax[0, 0]
    colors = [PALETTE[1] if c == "RandN" else PALETTE[0] for c in imp.index]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set(title=title, xlabel="mean absolute attention")
    return fig


# --- Mortality ------------------------------------------------------------------------------


def mortality_surface(
    res: pd.DataFrame, col: str, title: str, obs_year: int, residual: bool = False
) -> plt.Figure:
    piv = res.pivot(index="Age", columns="Year", values=col)
    if residual:
        piv = res.pivot(index="Age", columns="Year", values="logmx") - piv
    fig, ax = _fig(7, 4)
    ax = ax[0, 0]
    kw = (
        dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        if residual
        else dict(cmap="viridis", vmin=-11, vmax=-0.5)
    )
    im = ax.imshow(
        piv.to_numpy(),
        aspect="auto",
        origin="lower",
        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
        **kw,
    )
    ax.axvline(obs_year + 0.5, color="white" if not residual else INK, lw=1.5, ls="--")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="observed − fitted log m" if residual else "log m(x,t)")
    ax.set(xlabel="calendar year", ylabel="age", title=title)
    return fig


def age_profile(res: pd.DataFrame, models: list[str], year: int) -> plt.Figure:
    d = res[res.Year == year]
    fig, ax = _fig()
    ax = ax[0, 0]
    ax.plot(d.Age, d.logmx, "o", ms=3, color=INK, label=f"observed {year}")
    for i, m in enumerate(models):
        ax.plot(d.Age, d[m], color=PALETTE[i % len(PALETTE)], label=m)
    ax.set(xlabel="age", ylabel="log m(x,t)", title=f"Forecast age profile, {year}")
    ax.legend()
    return fig


def rmse_by_year(res: pd.DataFrame, models: list[str], obs_year: int) -> plt.Figure:
    fig, ax = _fig()
    ax = ax[0, 0]
    d = res[res.Year > obs_year]
    for i, m in enumerate(models):
        e = d.assign(e=(d.logmx - d[m]) ** 2).groupby("Year").e.mean() ** 0.5
        ax.plot(e.index, e.values, color=PALETTE[i % len(PALETTE)], label=m)
    ax.set(xlabel="forecast year", ylabel="RMSE of log m", title="Forecast error by horizon")
    ax.legend()
    return fig


# --- Reserving ------------------------------------------------------------------------------


def reserve_scatter(res: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, ax = _fig(5.2, 5)
    ax = ax[0, 0]
    for i, m in enumerate(models):
        ax.scatter(
            res.actual_reserve, res[m], s=14, alpha=0.7, color=PALETTE[i % len(PALETTE)], label=m
        )
    lim = [max(1, res.actual_reserve.min()), res.actual_reserve.max()]
    ax.plot(lim, lim, color=MUTED, ls=":")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="actual outstanding (realised)",
        ylabel="predicted reserve",
        title="Reserve back-test by company",
    )
    ax.legend()
    return fig


def reserve_errors(res: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, ax = _fig()
    ax = ax[0, 0]
    res = res[res.actual_reserve > 0]
    data = [(res[m] - res.actual_reserve) / res.actual_reserve * 100 for m in models]
    ax.boxplot(data, tick_labels=models, showfliers=False)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(ylabel="reserve error (%)", title="Reserve error distribution across companies")
    return fig


def development(tri: pd.DataFrame, title: str = "Incremental paid pattern") -> plt.Figure:
    g = tri.groupby("dev").apply(
        lambda d: d.incr_paid.sum() / d.premium.sum(), include_groups=False
    )
    fig, ax = _fig()
    ax = ax[0, 0]
    ax.bar(g.index, g.values, color=PALETTE[0])
    ax.set(xlabel="development year", ylabel="incremental paid / premium", title=title)
    return fig
