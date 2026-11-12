"""Claim-frequency pipeline: one namespace per model, plus a cross-model comparison.

kedro run --pipeline frequency                          # all models
kedro run --pipeline frequency --namespaces glm,cann    # selected models
kedro run --pipeline frequency --params frequency.data.scale=0.1
"""

from __future__ import annotations

from functools import partial

from kedro.pipeline import Pipeline, node, pipeline

from deep_actuarial_science.pipelines.frequency import nodes as n
from deep_actuarial_science.pipelines.ingest import aitools_node

# model namespace -> model it builds on (CANN uses the fitted GLM3 as its offset, as in
# Wüthrich & Merz 2023, Table 7.7, and Brauer 2024)
MODELS = {
    "homogeneous": None,
    "glm": None,  # Poisson GLM1, Wüthrich & Merz (2023), Table 5.5
    "glm3": None,  # Poisson GLM3, same table
    "gbm": None,
    "fnn": None,
    "cann": "glm3",
    "localglmnet": None,
}


def model_pipeline(name: str, base: str | None) -> Pipeline:
    fit_inputs = ["freq_learn", "params:model"] + ([f"{base}.freq_model"] if base else [])
    template = pipeline(
        [
            node(n.fit_model, fit_inputs, "freq_model", name="fit"),
            node(
                n.predict,
                ["freq_model", "freq_learn", "freq_test"],
                "freq_predictions",
                name="predict",
            ),
            node(
                n.evaluate,
                ["freq_model", "freq_predictions", "params:model"],
                ["freq_metrics", "freq_metrics_mlflow"],
                name="evaluate",
            ),
            node(
                n.figures,
                ["freq_model", "freq_predictions", "freq_test", "params:model"],
                "freq_figures",
                name="figures",
            ),
            node(
                partial(n.reconcile, name=name),
                [
                    "freq_metrics",
                    "params:benchmark",
                    "params:frequency.data",
                    "params:reference_setup",
                ],
                "freq_reconciliation",
                name="reconcile",
            ),
        ]
    )
    return pipeline(
        template,
        namespace=name,
        inputs={"freq_learn", "freq_test", *([f"{base}.freq_model"] if base else [])},
        parameters={
            "params:model": f"params:frequency.models.{name}",
            "params:benchmark": f"params:benchmarks.frequency.{name}",
            "params:frequency.data": "params:frequency.data",
            "params:reference_setup": "params:reference_setup",
        },
    )


def create_pipeline(**kwargs) -> Pipeline:
    prep = pipeline(
        [
            aitools_node,
            node(
                n.prepare,
                ["freq_raw", "params:frequency.data"],
                ["freq_learn", "freq_test"],
                name="prepare_frequency_data",
            ),
        ]
    )
    models = sum((model_pipeline(m, b) for m, b in MODELS.items()), Pipeline([]))
    compare = pipeline(
        [
            node(
                n.make_compare(list(MODELS)),
                [
                    f"{m}.{d}"
                    for d in ("freq_metrics", "freq_predictions", "freq_reconciliation")
                    for m in MODELS
                ],
                ["freq_leaderboard", "freq_findings", "freq_comparison_figures"],
                name="compare_frequency_models",
            )
        ]
    )
    return pipeline(prep + models + compare, tags="frequency")
