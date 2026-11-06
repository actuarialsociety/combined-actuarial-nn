"""Nodes for the claim-frequency pipeline. Pure functions: easy to call from a notebook."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deep_actuarial_science import plots
from deep_actuarial_science.benchmarks import reconcile as _reconcile
from deep_actuarial_science.metrics import balance_ratio, gini_index, poisson_deviance, to_mlflow
from deep_actuarial_science.models.frequency import FrequencyModel, LocalGLMnet, build

log = logging.getLogger(__name__)
KEEP = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "DrivAge",
    "BonusMalus",
    "VehAge",
    "VehBrand",
    "Region",
    "Area",
    "VehGas",
]


def prepare(raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed learn/test split of Wüthrich & Merz (2023) plus a pure-noise covariate.

    `scale` < 1 subsamples both sets (same seed) for fast experiments; benchmarks
    only apply at scale 1.0.
    """
    df = raw.copy()
    df["RandN"] = np.random.default_rng(params.get("noise_seed", 500)).normal(0, 1, len(df))
    learn, test = df[df.LearnTest == "L"].copy(), df[df.LearnTest == "T"].copy()
    scale = params.get("scale", 1.0)
    if scale < 1.0:
        learn = learn.sample(frac=scale, random_state=params.get("sample_seed", 1))
        test = test.sample(frac=scale, random_state=params.get("sample_seed", 1))
    log.info("Learn %s rows, test %s rows (scale %.3f)", len(learn), len(test), scale)
    return learn, test


def fit_model(learn: pd.DataFrame, params: dict, base: FrequencyModel | None = None):
    model = build(params["kind"]).fit(learn, params, base)
    log.info(
        "Fitted %s in %.1fs (%s parameters)", params["kind"], model.fit_seconds, model.n_params()
    )
    return model


def predict(model: FrequencyModel, learn: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = []
    for split, d in (("learn", learn), ("test", test)):
        out.append(d[KEEP].assign(split=split, pred=model.predict(d)))
    return pd.concat(out, ignore_index=True)


def evaluate(model: FrequencyModel, pred: pd.DataFrame, params: dict):
    m = {}
    for split in ("learn", "test"):
        d = pred[pred.split == split]
        m[f"deviance_{split}"] = poisson_deviance(d.ClaimNb, d.pred, d.Exposure)
        m[f"balance_{split}"] = balance_ratio(d.ClaimNb, d.pred)
    t = pred[pred.split == "test"]
    m["gini_test"] = gini_index(t.ClaimNb, t.pred, t.Exposure)
    m["n_params"] = model.n_params()
    m["fit_seconds"] = model.fit_seconds
    if model.history:
        m["best_epoch"] = model.history["best_epoch"]
    log.info(
        "%s: deviance learn %.3f, test %.3f, gini %.3f",
        params["kind"],
        m["deviance_learn"],
        m["deviance_test"],
        m["gini_test"],
    )
    return m, to_mlflow(m)


def figures(model: FrequencyModel, pred: pd.DataFrame, test: pd.DataFrame, params: dict) -> dict:
    t = pred[pred.split == "test"].copy()
    t["DrivAge"] = t.DrivAge.clip(upper=90)
    t["BonusMalus"] = t.BonusMalus.clip(upper=150)
    name = params["kind"]
    figs = {
        "calibration": plots.calibration(t, title=f"{name}: calibration by decile (test)"),
        "marginal_drivage": plots.marginal(t, "DrivAge", title=f"{name}: driver age (test)"),
        "marginal_bonusmalus": plots.marginal(t, "BonusMalus", title=f"{name}: bonus-malus (test)"),
        "marginal_vehbrand": plots.marginal(t, "VehBrand", title=f"{name}: vehicle brand (test)"),
        "training": plots.training(model.history, title=f"{name}: training history"),
    }
    if isinstance(model, LocalGLMnet):
        att = model.attentions(test)
        X = model.design(test)
        figs["attention"] = plots.attention(att, X, model.features)
        figs["importance"] = plots.importance(att)
    return plots.named(figs)


def reconcile(
    metrics: dict, benchmark: dict | None, data_params: dict, params: dict, reference: bool = True
) -> pd.DataFrame:
    applicable = reference and data_params.get("scale", 1.0) == 1.0
    return _reconcile(metrics, benchmark, applicable, params["kind"])


def make_compare(names: list[str]):
    """Leaderboard across models, with the published benchmark drawn next to each result."""

    def compare_frequency_models(*args) -> tuple[pd.DataFrame, dict]:
        k = len(names)
        metrics, preds, recs = args[:k], args[k : 2 * k], args[2 * k :]
        table = pd.DataFrame([{"model": n, **m} for n, m in zip(names, metrics)])
        bench = {
            r.model.iloc[0]: r.set_index("metric").expected.get("deviance_test")
            for r in recs
            if len(r)
        }
        table["benchmark_deviance_test"] = table.model.map(bench)
        figs = {
            "leaderboard_test_deviance": plots.leaderboard(
                table,
                "deviance_test",
                "benchmark_deviance_test",
                title="Out-of-sample Poisson deviance (×10⁻²)",
            ),
            "leaderboard_gini": plots.leaderboard(
                table, "gini_test", title="Gini index (test)", lower_is_better=False
            ),
            "lorenz": plots.lorenz({n: p[p.split == "test"] for n, p in zip(names, preds)}),
        }
        return table, plots.named(figs)

    return compare_frequency_models
