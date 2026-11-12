"""Deep learning models in actuarial science: Kedro pipelines behind the lessons."""

from importlib.metadata import PackageNotFoundError, version

# Load PyTorch before LightGBM. On macOS each links its own libomp; with LightGBM's loaded
# first, PyTorch training crashes or hangs. GBM also trains single-threaded on macOS
# (models/frequency.py) so only one OpenMP runtime ever runs.
import torch  # noqa: F401

try:  # one source of truth: `version` in pyproject.toml (bump it with every release)
    __version__ = version("deep-actuarial-science")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"
