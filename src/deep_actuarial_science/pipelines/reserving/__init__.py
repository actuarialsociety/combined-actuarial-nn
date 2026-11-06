"""Reserving pipeline: chain-ladder vs neural ODP, back-tested on realised CAS outcomes.

kedro run --pipeline reserving
kedro run --pipeline reserving --env scale_up       # all six lines, all companies
"""

from __future__ import annotations

import logging

import chainladder as cl
import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

from deep_actuarial_science import plots
from deep_actuarial_science.benchmarks import reconcile as _reconcile
from deep_actuarial_science.metrics import to_mlflow
from deep_actuarial_science.models import reserving as rm
from deep_actuarial_science.pipelines.ingest import cas_node

log = logging.getLogger(__name__)
MODELS = ["chain_ladder", "neural_odp"]


def fit_predict(tri: pd.DataFrame, params: dict):
    if params["kind"] == "chain_ladder":
        return rm.chain_ladder(tri)
    return rm.neural_odp(tri, params)


def evaluate(pred: pd.DataFrame, params: dict, history: dict | None = None):
    """Back-test against the realised lower triangle. Relative company errors use only
    companies with a positive realised reserve; the premium-relative error uses all."""
    r = rm.reserves(pred)
    pos = r[r.actual_reserve > 0]
    err = (pos.predicted_reserve - pos.actual_reserve) / pos.actual_reserve
    low = pred[~pred.upper]
    m = {
        "total_reserve_error_pct": 100 * (r.predicted_reserve.sum() / r.actual_reserve.sum() - 1),
        "median_abs_error_pct": 100 * float(np.median(np.abs(err))),
        "share_within_10pct": float(np.mean(np.abs(err) <= 0.10)),
        "mean_abs_error_pct_premium": 100
        * float(np.mean(np.abs(r.predicted_reserve - r.actual_reserve) / r.premium)),
        "cell_rmse_per_premium": float(
            np.sqrt(
                np.mean(
                    ((low.pred - low.incr_paid) / low.groupby(rm.KEY).premium.transform("sum")) ** 2
                )
            )
        ),
        "n_companies": len(r),
    }
    if history and "best_epoch" in history:
        m["best_epoch"] = history["best_epoch"]
    log.info(
        "%s: total reserve error %.2f%%, median |error| %.1f%%",
        params["kind"],
        m["total_reserve_error_pct"],
        m["median_abs_error_pct"],
    )
    return m, to_mlflow(m)


def package_chain_ladder(tri: pd.DataFrame) -> pd.Series:
    """IBNR per company from the `chainladder` package, for reconciliation."""
    up = tri[tri.upper].assign(
        origin=lambda x: x.ay.astype(str), development=lambda x: x.calendar.astype(str)
    )
    t = cl.Triangle(
        up,
        origin="origin",
        development="development",
        columns="cum_paid",
        index=rm.KEY,
        cumulative=True,
    )
    ibnr = cl.Chainladder().fit(t).ibnr_.sum("origin").to_frame(keepdims=True)
    return (
        ibnr.reset_index().set_index(rm.KEY)["cum_paid"]
        if "cum_paid" in ibnr
        else ibnr.reset_index().set_index(rm.KEY).iloc[:, -1]
    )


def reconcile(pred: pd.DataFrame, tri: pd.DataFrame, benchmark: dict | None, params: dict):
    if params["kind"] != "chain_ladder":
        return _reconcile({}, benchmark, True, params["kind"])
    ours = rm.reserves(pred).set_index(rm.KEY).predicted_reserve
    pkg = package_chain_ladder(tri).reindex(ours.index)
    diff = float(np.max(np.abs(ours / pkg - 1)))
    return _reconcile({"max_rel_diff_vs_package": diff}, benchmark, True, params["kind"])


def compare(*args):
    k = len(MODELS)
    metrics, preds, tri = args[:k], args[k : 2 * k], args[-1]
    table = pd.DataFrame([{"model": n, **m} for n, m in zip(MODELS, metrics)])
    res = rm.reserves(preds[0]).rename(columns={"predicted_reserve": MODELS[0]})
    for n, p in zip(MODELS[1:], preds[1:]):
        res[n] = rm.reserves(p).predicted_reserve.to_numpy()
    figs = {
        "reserve_scatter": plots.reserve_scatter(res, MODELS),
        "reserve_errors": plots.reserve_errors(res, MODELS),
        "development_pattern": plots.development(tri[tri.upper]),
        "leaderboard_median_error": plots.leaderboard(
            table, "median_abs_error_pct", title="Median |reserve error| (%)"
        ),
    }
    return table, res, plots.named(figs)


def model_pipeline(name: str) -> Pipeline:
    template = pipeline(
        [
            node(
                fit_predict,
                ["res_triangles", "params:model"],
                ["res_predictions", "res_history"],
                name="fit",
            ),
            node(
                evaluate,
                ["res_predictions", "params:model", "res_history"],
                ["res_metrics", "res_metrics_mlflow"],
                name="evaluate",
            ),
            node(
                reconcile,
                ["res_predictions", "res_triangles", "params:benchmark", "params:model"],
                "res_reconciliation",
                name="reconcile",
            ),
        ]
    )
    return pipeline(
        template,
        namespace=name,
        inputs={"res_triangles"},
        parameters={
            "params:model": f"params:reserving.models.{name}",
            "params:benchmark": f"params:benchmarks.reserving.{name}",
        },
    )


def create_pipeline(**kwargs) -> Pipeline:
    prep = pipeline(
        [
            cas_node,
            node(
                rm.prepare,
                ["cas_raw", "params:reserving.data"],
                "res_triangles",
                name="prepare_reserving_data",
            ),
        ]
    )
    models = sum((model_pipeline(m) for m in MODELS), Pipeline([]))
    comp = pipeline(
        [
            node(
                compare,
                [f"{m}.res_metrics" for m in MODELS]
                + [f"{m}.res_predictions" for m in MODELS]
                + ["res_triangles"],
                ["res_leaderboard", "res_company_reserves", "res_comparison_figures"],
                name="compare_reserving_models",
            )
        ]
    )
    return pipeline(prep + models + comp, tags="reserving")
