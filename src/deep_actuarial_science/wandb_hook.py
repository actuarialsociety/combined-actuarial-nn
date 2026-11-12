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
import time

import pandas as pd
from kedro.framework.hooks import hook_impl
from matplotlib.figure import Figure

from deep_actuarial_science.audit import version
from deep_actuarial_science.training_log import fit_outputs

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("WANDB_PROJECT")) and os.environ.get("WANDB_MODE") != "disabled"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: v})
    return out


def _init_or_offline(wandb, kwargs: dict, attempts: int = 3):
    """W&B must never sink a pipeline run: retry a failed online start (a busy or unreachable
    server), then log offline, which `wandb sync` uploads later; failing that, no W&B."""
    for i in range(attempts):
        try:
            return wandb.init(**kwargs)
        except Exception as e:  # noqa: BLE001 - auth timeouts, network errors, 5xx
            log.warning("W&B init failed (attempt %d/%d): %s", i + 1, attempts, e)
            time.sleep(5 * 2**i)
    try:
        log.warning("W&B unreachable: logging offline; upload later with `wandb sync`")
        return wandb.init(mode="offline", **kwargs)
    except Exception as e:  # noqa: BLE001
        log.warning("W&B offline init failed too, continuing without W&B: %s", e)
        return None


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
        kwargs = dict(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ.get("WANDB_ENTITY"),
            job_type=name,
            group=os.environ.get("WANDB_RUN_GROUP"),
            name=f"{name}-{env}",
            tags=[name, env, f"device:{device().type}"],
            config={
                **version(),  # package version and git commit: the model version
                "env": env,
                "device": device().type,
                "namespaces": run_params.get("namespaces"),
                **_flatten(domain if isinstance(domain, dict) else {}),
            },
            reinit="finish_previous",
        )
        self.run = _init_or_offline(wandb, kwargs)

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, node) -> None:
        if self.run is None:
            return
        try:
            self._log(dataset_name, data)
        except Exception as e:  # noqa: BLE001 - a logging failure must not fail the node
            log.warning("W&B logging of %s failed: %s", dataset_name, e)

    def _log(self, dataset_name: str, data) -> None:
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

    @hook_impl
    def after_node_run(self, node, outputs: dict) -> None:
        if self.run is None:
            return
        try:
            for ns, df in fit_outputs(node, outputs):
                self._log_curves(ns, df)
        except Exception as e:  # noqa: BLE001 - a logging failure must not fail the node
            log.warning("W&B training-curve logging of %s failed: %s", node.name, e)

    def _log_curves(self, ns: str, df: pd.DataFrame) -> None:
        """Every seed's per-epoch curves: one chart per metric (and sex), plus the table."""
        import wandb

        logged = {f"{ns}/training_curves": wandb.Table(dataframe=df)}
        for (group, metric), d in df.groupby(["group", "metric"]):
            runs = [r for _, r in d.groupby("run")]
            title = " ".join(x for x in (ns, group, metric) if x)
            logged[f"{ns}/train/{group + '/' if group else ''}{metric}"] = wandb.plot.line_series(
                xs=[r.epoch.tolist() for r in runs],
                ys=[r.value.tolist() for r in runs],
                keys=[f"seed_{int(r.run.iloc[0]):02d}" for r in runs],
                title=title,
                xname="epoch",
            )
        self.run.log(logged)

    def _finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as e:  # noqa: BLE001
                log.warning("W&B finish failed: %s", e)
            self.run = None

    @hook_impl
    def after_pipeline_run(self) -> None:
        self._finish()

    @hook_impl
    def on_pipeline_error(self) -> None:
        self._finish(exit_code=1)
