"""Optional Weights & Biases logging, alongside MLflow.

Off unless `WANDB_PROJECT` is set (and `pip install wandb`). Then every `kedro run` (or
notebook `run(...)`) becomes one W&B run that logs, for each model namespace:

- the pipeline's parameters as the run config;
- every metrics dict (`<ns>.<domain>_metrics`) as `<ns>/<metric>` in the summary;
- every figure dict as images under `<dataset>/<figure>`;
- every benchmark reconciliation and leaderboard as a W&B Table.

`WANDB_ENTITY` picks the team, `WANDB_MODE=offline` logs locally for `wandb sync` later, and
`WANDB_API_KEY` (or `wandb login`) authenticates. Keys live in `.env`, never in the repo.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from kedro.framework.hooks import hook_impl
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("WANDB_PROJECT")) and os.environ.get("WANDB_MODE") != "disabled"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: v})
    return out


class WandbHook:
    def __init__(self) -> None:
        self.run = None

    @hook_impl
    def before_pipeline_run(self, run_params: dict, pipeline, catalog) -> None:
        if not enabled():
            return
        try:
            import wandb
        except ImportError:
            log.warning("WANDB_PROJECT is set but wandb is not installed: pip install wandb")
            return
        from deep_actuarial_science.models.torch_utils import device

        names = run_params.get("pipeline_names") or [run_params.get("pipeline_name")]
        name = "+".join(n for n in names if n) or "__default__"
        env = run_params.get("env") or "local"
        try:
            params = catalog.load("parameters")
        except Exception:  # noqa: BLE001 - config is a nice-to-have, never block a run
            params = {}
        domain = params.get(name, params) if isinstance(params, dict) else {}
        self.run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ.get("WANDB_ENTITY"),
            job_type=name,
            group=os.environ.get("WANDB_RUN_GROUP"),
            name=f"{name}-{env}",
            tags=[name, env, f"device:{device().type}"],
            config={
                "env": env,
                "device": device().type,
                "namespaces": run_params.get("namespaces"),
                **_flatten(domain if isinstance(domain, dict) else {}),
            },
            reinit="finish_previous",
        )

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, node) -> None:
        if self.run is None:
            return
        import wandb

        if isinstance(data, dict) and any(isinstance(v, Figure) for v in data.values()):
            self.run.log(
                {
                    f"{dataset_name}/{k}": wandb.Image(f)
                    for k, f in data.items()
                    if isinstance(f, Figure)
                }
            )
        elif dataset_name.endswith("_metrics") and isinstance(data, dict):
            ns = dataset_name.split(".")[0]
            scalars = {
                f"{ns}/{k}": v
                for k, v in data.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self.run.summary.update(scalars)
            self.run.log(scalars)
        elif isinstance(data, pd.DataFrame) and (
            dataset_name.endswith("_reconciliation") or dataset_name.endswith("leaderboard")
        ):
            self.run.log({dataset_name: wandb.Table(dataframe=data.reset_index(drop=True))})

    def _finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None

    @hook_impl
    def after_pipeline_run(self) -> None:
        self._finish()

    @hook_impl
    def on_pipeline_error(self) -> None:
        self._finish(exit_code=1)
