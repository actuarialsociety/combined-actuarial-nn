"""MLflow logging beyond kedro-mlflow's metrics (kedro-mlflow starts the run): every figure
dict a node saves, and every model's per-epoch training curves (`training_log.py`)."""

from __future__ import annotations

import logging

import mlflow
from kedro.framework.hooks import hook_impl
from matplotlib.figure import Figure
from mlflow.entities import Metric

from deep_actuarial_science.training_log import fit_outputs, series_key

log = logging.getLogger(__name__)
BATCH = 1000  # MLflow's limit of metrics per log_batch call


class MlflowFigureHook:
    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, node) -> None:
        if not isinstance(data, dict) or mlflow.active_run() is None:
            return
        for key, fig in data.items():
            if isinstance(fig, Figure):
                mlflow.log_figure(fig, f"figures/{dataset_name}/{key}")

    @hook_impl
    def after_node_run(self, node, outputs: dict) -> None:
        run = mlflow.active_run()
        if run is None:
            return
        try:
            for ns, df in fit_outputs(node, outputs):
                self._log_curves(run.info.run_id, ns, df)
        except Exception as e:  # noqa: BLE001 - tracking must never fail a pipeline
            log.warning("MLflow training-curve logging failed: %s", e)

    @staticmethod
    def _log_curves(run_id: str, ns: str, df) -> None:
        client = mlflow.MlflowClient()
        metrics = [
            Metric(series_key(ns, r.group, r.run, r.metric), float(r.value), 0, int(r.epoch))
            for r in df.itertuples()
        ]
        for i in range(0, len(metrics), BATCH):
            client.log_batch(run_id, metrics=metrics[i : i + BATCH])
        mlflow.log_text(df.to_csv(index=False), f"training/{ns}.csv")
        log.info("MLflow: %s per-epoch points of %s", len(metrics), ns)
