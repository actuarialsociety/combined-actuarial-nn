#!/usr/bin/env python3
"""Copy the figures the lessons use from the latest pipeline run into lessons/figures/.

Run after a reference run (`kedro run`) so slides always show real, reproducible output.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data" / "08_reporting"
OUT = ROOT / "lessons" / "figures"

FIGURES = {
    "00-leaderboard.png": "frequency/comparison/leaderboard_test_deviance.png",
    "00-gaps.png": "frequency/comparison/gaps.png",
    "01-glm-calibration.png": "frequency/glm/figures/calibration.png",
    "01-glm-drivage.png": "frequency/glm/figures/marginal_drivage.png",
    "01-gbm-drivage.png": "frequency/gbm/figures/marginal_drivage.png",
    "01-lorenz.png": "frequency/comparison/lorenz.png",
    "02-fnn-training.png": "frequency/fnn/figures/training.png",
    "02-fnn-bonusmalus.png": "frequency/fnn/figures/marginal_bonusmalus.png",
    "02-fnn-calibration.png": "frequency/fnn/figures/calibration.png",
    "03-cann-drivage.png": "frequency/cann/figures/marginal_drivage.png",
    "03-cann-training.png": "frequency/cann/figures/training.png",
    "03-leaderboard.png": "frequency/comparison/leaderboard_test_deviance.png",
    "04-attention.png": "frequency/localglmnet/figures/attention.png",
    "04-importance.png": "frequency/localglmnet/figures/importance.png",
    "05-seeds.png": "mortality/comparison/seeds.png",
    "05-observed.png": "mortality/comparison/observed_surface.png",
    "05-age-profile.png": "mortality/comparison/age_profile_last_year.png",
    "05-horizon.png": "mortality/comparison/rmse_by_horizon.png",
    "05-lc-residuals.png": "mortality/lee_carter/figures/residuals.png",
    "05-lstm-residuals.png": "mortality/lstm/figures/residuals.png",
    "06-scatter.png": "reserving/comparison/reserve_scatter.png",
    "06-errors.png": "reserving/comparison/reserve_errors.png",
    "06-development.png": "reserving/comparison/development_pattern.png",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for dst, src in FIGURES.items():
        s = R / src
        if s.exists():
            shutil.copy2(s, OUT / dst)
        else:
            missing.append(src)
    print(f"Copied {len(FIGURES) - len(missing)} figures to {OUT}")
    for m in missing:
        print("missing (run the pipeline first):", m)


if __name__ == "__main__":
    main()
