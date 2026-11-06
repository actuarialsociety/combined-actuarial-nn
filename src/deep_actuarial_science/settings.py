"""Kedro project settings. kedro-mlflow registers its hooks through its entry point."""

from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "mlflow": ["mlflow*", "mlflow*/**"],
    },
    # Deep-merge parameters across environments, so `conf/smoke` can override one key.
    "merge_strategy": {"parameters": "soft"},
    # Allow ${oc.env:VAR, default} in mlflow.yml (tracking URI, run name) for containers/CI.
    "custom_resolvers": {"oc.env": oc.env},
}

from deep_actuarial_science.hooks import MlflowFigureHook

HOOKS = (MlflowFigureHook(),)
