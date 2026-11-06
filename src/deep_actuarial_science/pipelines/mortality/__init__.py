"""Mortality forecasting pipeline: Lee-Carter vs neural forecasters on HMD data.

kedro run --pipeline mortality
kedro run --pipeline mortality --namespaces lee_carter,lstm
kedro run --pipeline mortality --env scale_up       # both sexes (list values: use an env)
"""

from __future__ import annotations

import logging

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

from deep_actuarial_science import plots
from deep_actuarial_science.benchmarks import reconcile as _reconcile
from deep_actuarial_science.metrics import rmse, to_mlflow
from deep_actuarial_science.models import mortality as mm
from deep_actuarial_science.pipelines.ingest import aitools_node

log = logging.getLogger(__name__)
MODELS = ["lee_carter", "cnn", "cnn_tanh", "lstm", "transformer"]


def fit_forecast(data: pd.DataFrame, params: dict, training: dict, data_params: dict):
    return mm.fit_forecast(data, params, training, data_params["obs_year"])


def evaluate(res: pd.DataFrame, params: dict, data_params: dict, training: dict):
    """RMSE of log m on the reference evaluation window (in-sample starts after the lookback)."""
    first_in = data_params["first_year"] + training["lookback"]
    ins = res[(res.split == "learn") & (res.Year >= first_in)]
    out = res[res.split == "test"]
    m = {"rmse_in": rmse(ins.logmx, ins.pred), "rmse_out": rmse(out.logmx, out.pred)}
    for sex, d in out.groupby("Sex"):
        m[f"rmse_out_{sex.lower()}"] = rmse(d.logmx, d.pred)
    log.info("%s: RMSE in %.4f, out %.4f", params["kind"], m["rmse_in"], m["rmse_out"])
    return m, to_mlflow(m)


def figures(res: pd.DataFrame, history: dict, params: dict, data_params: dict) -> dict:
    k, oy = params["kind"], data_params["obs_year"]
    d = res[res.Sex == res.Sex.iloc[0]]
    figs = {
        "residuals": plots.mortality_surface(
            d, "pred", f"{k}: residuals (observed − fitted)", oy, residual=True
        )
    }
    first = next(iter(history.values()))
    if "loss" in first:
        figs["training"] = plots.training(first, title=f"{k}: training history (scaled MSE)")
    return plots.named(figs)


def reconcile(
    metrics: dict, benchmark: dict | None, data_params: dict, params: dict, reference: bool = True
):
    ref = reference and data_params["sexes"] == ["Male"] and data_params["obs_year"] == 1999
    return _reconcile(metrics, benchmark, ref, params["kind"])


def compare(*args):
    k = len(MODELS)
    metrics, results = args[:k], args[k : 2 * k]
    data_params = args[-1]
    table = pd.DataFrame([{"model": n, **m} for n, m in zip(MODELS, metrics)])
    oy = data_params["obs_year"]
    wide = results[0][["Sex", "Year", "Age", "logmx"]].copy()
    for n, r in zip(MODELS, results):
        wide[n] = r["pred"].to_numpy()
    male = wide[wide.Sex == wide.Sex.iloc[0]]
    figs = {
        "leaderboard_rmse_out": plots.leaderboard(
            table, "rmse_out", title="Out-of-sample RMSE of log m(x,t)"
        ),
        "observed_surface": plots.mortality_surface(male, "logmx", "Observed log m(x,t)", oy),
        "age_profile_last_year": plots.age_profile(male, MODELS, int(male.Year.max())),
        "rmse_by_horizon": plots.rmse_by_year(male, MODELS, oy),
    }
    return table, plots.named(figs)


def model_pipeline(name: str) -> Pipeline:
    template = pipeline(
        [
            node(
                fit_forecast,
                ["mort_data", "params:model", "params:mortality.training", "params:mortality.data"],
                ["mort_results", "mort_history"],
                name="fit",
            ),
            node(
                evaluate,
                [
                    "mort_results",
                    "params:model",
                    "params:mortality.data",
                    "params:mortality.training",
                ],
                ["mort_metrics", "mort_metrics_mlflow"],
                name="evaluate",
            ),
            node(
                figures,
                ["mort_results", "mort_history", "params:model", "params:mortality.data"],
                "mort_figures",
                name="figures",
            ),
            node(
                reconcile,
                [
                    "mort_metrics",
                    "params:benchmark",
                    "params:mortality.data",
                    "params:model",
                    "params:reference_setup",
                ],
                "mort_reconciliation",
                name="reconcile",
            ),
        ]
    )
    return pipeline(
        template,
        namespace=name,
        inputs={"mort_data"},
        parameters={
            "params:model": f"params:mortality.models.{name}",
            "params:benchmark": f"params:benchmarks.mortality.{name}",
            "params:mortality.training": "params:mortality.training",
            "params:mortality.data": "params:mortality.data",
            "params:reference_setup": "params:reference_setup",
        },
    )


def create_pipeline(**kwargs) -> Pipeline:
    prep = pipeline(
        [
            aitools_node,
            node(
                mm.prepare,
                ["mortality_raw", "params:mortality.data"],
                "mort_data",
                name="prepare_mortality_data",
            ),
        ]
    )
    models = sum((model_pipeline(m) for m in MODELS), Pipeline([]))
    comp = pipeline(
        [
            node(
                compare,
                [f"{m}.mort_metrics" for m in MODELS]
                + [f"{m}.mort_results" for m in MODELS]
                + ["params:mortality.data"],
                ["mort_leaderboard", "mort_comparison_figures"],
                name="compare_mortality_models",
            )
        ]
    )
    return pipeline(prep + models + comp, tags="mortality")
