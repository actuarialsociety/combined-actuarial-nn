"""Log every figure dict a node saves to the active MLflow run (kedro-mlflow starts it)."""

from __future__ import annotations

import mlflow
from kedro.framework.hooks import hook_impl
from matplotlib.figure import Figure


class MlflowFigureHook:
    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, node) -> None:
        if not isinstance(data, dict) or mlflow.active_run() is None:
            return
        for key, fig in data.items():
            if isinstance(fig, Figure):
                mlflow.log_figure(fig, f"figures/{dataset_name}/{key}")
