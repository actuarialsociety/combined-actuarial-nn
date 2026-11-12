"""Tracking beyond kedro-mlflow's metrics (kedro-mlflow starts the MLflow run):

- `MlflowFigureHook`: every figure dict a node saves, and every model's per-epoch training
  curves (`training_log.py`).
- `AuditHook`: each step's duration, row counts and reconciled totals (`audit.py`), to MLflow
  and, when a W&B run is open, to W&B; plus the model version as tags.

kedro-mlflow's own hooks run before the project's, so its run is already closed when ours see
`after_pipeline_run`: the audit table is therefore re-logged after every node."""

from __future__ import annotations

import logging

import mlflow
from kedro.framework.hooks import hook_impl
from matplotlib.figure import Figure
from mlflow.entities import Metric

from deep_actuarial_science.audit import AuditTrail, metric_key, version
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


class AuditHook:
    """Per-step audit trail (time, rows, totals) and the model version, on every run."""

    def __init__(self) -> None:
        self.trail = AuditTrail()
        self.version = version()

    @hook_impl
    def before_pipeline_run(self) -> None:
        self.trail = AuditTrail()  # one table per run
        if mlflow.active_run() is not None:
            mlflow.set_tags(self.version)

    @hook_impl
    def before_node_run(self, node) -> None:
        self.trail.start(node.name)

    @hook_impl
    def after_node_run(self, node, inputs: dict, outputs: dict) -> None:
        try:
            rows = self.trail.finish(node.name, inputs, outputs)
            numbers = {
                metric_key(r, k): float(v)
                for r in rows
                for k, v in r.items()
                if k not in ("node", "side", "dataset") and v == v  # skip NaN
            }
            if mlflow.active_run() is not None:
                mlflow.log_metrics({k.replace("/", "."): v for k, v in numbers.items()})
                mlflow.log_text(self.trail.table().to_csv(index=False), "audit/steps.csv")
            self._wandb(numbers)
        except Exception as e:  # noqa: BLE001 - auditing must never fail a pipeline
            log.warning("Audit logging of %s failed: %s", node.name, e)

    @staticmethod
    def _wandb(numbers: dict) -> None:
        """Log to the open W&B run, if WandbHook started one."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.run.summary.update(numbers)
