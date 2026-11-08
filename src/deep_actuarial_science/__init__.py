"""Deep learning models in actuarial science: Kedro pipelines behind the lessons."""

# Load PyTorch before LightGBM. On macOS each links its own libomp; with LightGBM's loaded
# first, PyTorch training crashes or hangs. GBM also trains single-threaded on macOS
# (models/frequency.py) so only one OpenMP runtime ever runs.
import torch  # noqa: F401

__version__ = "0.1.0"
