#!/usr/bin/env python3
"""Generate the lesson notebooks in notebooks/ (one per lesson).

Notebooks drive the same Kedro pipelines as `kedro run`, so a notebook result is a
pipeline result: tracked in MLflow and reconciled to the benchmarks. Edit this file,
not the .ipynb files, then run:  python3 scripts/make_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = "actuarialsociety/deep-actuarial-science"
OUT = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = f"""# Setup: works in Colab, the dev container and a local clone.
import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/deep-actuarial-science"):
    !git clone -q https://github.com/{REPO} /content/deep-actuarial-science
    !pip install -q -e /content/deep-actuarial-science
if "google.colab" in sys.modules:
    os.chdir("/content/deep-actuarial-science")
    sys.path.insert(0, "src")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from deep_actuarial_science.notebook import run, load, show, leaderboard, parameters, mlflow_ui
FAST = os.environ.get("DAS_ENV") == "smoke"   # CI runs every notebook in smoke mode
"""


def badge(name: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def notebook(name: str, title: str, intro: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n{badge(name)}\n\n{intro}"),
        nbf.v4.new_code_cell(SETUP),
    ]
    for kind, src in cells:
        nb.cells.append(
            nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        )
    nb.cells += [nbf.v4.new_markdown_cell(MLFLOW_MD), nbf.v4.new_code_cell("mlflow_ui()")]
    nbf.write(nb, OUT / name)


MLFLOW_MD = (
    "## Explore the runs in MLflow\n"
    "Parameters, metrics, figures and the benchmark table of every run above. In Colab the UI"
    " opens below and in a new tab through Colab's proxy (MLflow runs on the Colab machine, so"
    " a plain localhost link would point at your own computer)."
)


md, py = "md", "py"

LESSONS = [
    (
        "00_quickstart.ipynb",
        "00 · Quickstart: pipelines, configs, tracking",
        "Run a pipeline, change a config, see it in MLflow. Every lesson works this way.",
        [
            (
                md,
                "## 1. Run a pipeline\nThe GLM and the homogeneous model are closed-form: this takes"
                " about a minute on the full 678k-policy dataset.",
            ),
            (
                py,
                'run("frequency", models=["homogeneous", "glm"])\n'
                'leaderboard("frequency", ["homogeneous", "glm"])',
            ),
            (
                md,
                "## 2. Reconcile to the published benchmark\nEach model has a reconciliation table:"
                " **PASS** means this run reproduces the published result within tolerance.",
            ),
            (py, 'load("glm.freq_reconciliation")'),
            (
                md,
                "## 3. Change a config, re-run\nParameters live in `conf/base/parameters_*.yml`."
                " Override any key at run time, e.g. a different GLM formula:",
            ),
            (
                py,
                'formula = "ClaimNb ~ DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + AreaGLM"\n'
                'run("frequency", models=["glm"], params={} if FAST else {"frequency.models.glm.formula": formula})\n'
                'leaderboard("frequency", ["glm"])',
            ),
            (
                md,
                "## 4. Scale experiments\n`frequency.data.scale` subsamples the data: check how a"
                " model's out-of-sample deviance and stability change with data volume.",
            ),
            (
                py,
                "rows = []\nfor s in ([0.02] if FAST else [0.05, 0.25, 1.0]):\n"
                '    run("frequency", models=["glm"], params={"frequency.data.scale": s})\n'
                '    rows.append({"scale": s, **load("glm.freq_metrics")})\n'
                'import pandas as pd; pd.DataFrame(rows)[["scale", "deviance_learn", "deviance_test", "gini_test"]]',
            ),
            (
                md,
                "## 5. Tracking\nEvery run above is in MLflow (params, metrics, figures, model"
                " pickle, benchmark table). Query it from Python below, or open the UI with"
                " `mlflow_ui()` in the last cell (works in Colab, the container and locally).",
            ),
            (
                py,
                'import mlflow\nmlflow.set_tracking_uri("sqlite:///mlflow.db")\n'
                'mlflow.search_runs(experiment_names=["deep-actuarial-science"])'
                '.filter(regex="start_time|metrics.glm.deviance").head()',
            ),
        ],
    ),
    (
        "01_glm_gbm_baselines.ipynb",
        "01 · Baselines: Poisson GLM and GBM",
        r"Claim frequency $N_i \sim \mathrm{Poi}(v_i\lambda(x_i))$ on freMTPL2. Every deep model"
        " later in the course has to beat these two.",
        [
            (
                py,
                'models = ["homogeneous", "glm", "gbm"]\nrun("frequency", models=models)\nleaderboard("frequency", models)',
            ),
            (md, "## Open the box: GLM coefficients\nThe same `GLM` class the pipeline uses:"),
            (py, 'glm = load("glm.freq_model")\nglm.result.summary().tables[1]'),
            (
                md,
                "## Diagnostics (test set)\nCalibration by decile, marginal fits and the GBM's"
                " early-stopping curve.",
            ),
            (
                py,
                'show("frequency", "glm", ["calibration", "marginal_drivage"])\n'
                'show("frequency", "gbm", ["marginal_drivage", "training"])',
            ),
            (
                md,
                "## Exercise\nAdd `BonusMalusGLM + VehAgeGLM` to the GLM formula (see Quickstart §3)."
                " How close does the GLM get to the GBM? Which feature engineering closes the gap?",
            ),
        ],
    ),
    (
        "02_fnn_embeddings.ipynb",
        "02 · Feed-forward networks with entity embeddings",
        r"$\lambda(x)=\exp\{w_0+\langle w, z^{(d:1)}(x)\rangle\}$ with 2-D embeddings for brand and"
        " region. Full data: about 5-10 minutes on a CPU.",
        [
            (
                py,
                'run("frequency", models=["glm", "fnn"])\nleaderboard("frequency", ["glm", "fnn"])',
            ),
            (py, 'load("fnn.freq_reconciliation")'),
            (
                md,
                "## Open the box: the learned brand embedding\nBrands close together in this plane"
                " have similar effects on frequency.",
            ),
            (
                py,
                'import matplotlib.pyplot as plt\nfnn = load("fnn.freq_model")\n'
                "E = fnn.net.emb[0].weight.detach().numpy()\n"
                'names = fnn.pre.named_transformers_["cats"].categories_[0]\n'
                "fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(E[:, 0], E[:, 1])\n"
                "for n, (a, b) in zip(names, E): ax.annotate(n, (a, b))\n"
                'ax.set_title("VehBrand embedding"); plt.show()',
            ),
            (
                md,
                "## Production checks\n* **Balance**: `balance_learn` should be ≈ 1 (sum of predictions"
                " = sum of claims). Early stopping breaks it; `balance_correction: true` restores it.\n"
                "* **Stability**: refit with another `seed` and compare predictions.",
            ),
            (
                py,
                'show("frequency", "fnn", ["training", "calibration", "marginal_bonusmalus"])\n'
                'load("fnn.freq_metrics")',
            ),
            (
                md,
                "## Exercise\nRe-run with `frequency.models.fnn.seed` set to 3 other values. How much"
                " does the test deviance move? Compare that spread with the gap to the GLM.",
            ),
        ],
    ),
    (
        "03_cann.ipynb",
        "03 · CANN: nesting the GLM in a network",
        r"$\lambda^{\mathrm{CANN}}(x)=\lambda^{\mathrm{GLM}}(x)\cdot\exp\{w_0+\langle w,z^{(d:1)}(x)\rangle\}$,"
        r" initialised at $w=0$: training starts exactly at the GLM.",
        [
            (
                py,
                'run("frequency", models=["glm", "cann"])\nleaderboard("frequency", ["glm", "cann"])',
            ),
            (py, 'load("cann.freq_reconciliation")'),
            (
                md,
                "## Open the box: zero initialisation reproduces the GLM\nBuild a CANN with 0 epochs:"
                " its predictions equal the GLM's to float precision.",
            ),
            (
                py,
                "import numpy as np\nfrom deep_actuarial_science.models.frequency import CANN\n"
                'glm, test = load("glm.freq_model"), load("freq_test")\n'
                'cann0 = CANN().fit(load("freq_learn"), {"epochs": 0}, base=glm)\n'
                "np.abs(cann0.predict(test) / glm.predict(test) - 1).max()",
            ),
            (
                md,
                "## Where does the network correct the GLM?\nThe ratio CANN / GLM by covariate shows"
                " what the GLM misspecifies (e.g. the young-driver and bonus-malus shapes).",
            ),
            (
                py,
                'p = load("cann.freq_predictions").assign(glm=load("glm.freq_predictions").pred)\n'
                'p = p[p.split == "test"]\n'
                'r = p.groupby(p.DrivAge.clip(upper=90))[["pred", "glm"]].sum()\n'
                '(r.pred / r.glm).plot(title="CANN / GLM by driver age", ylabel="multiplicative correction");',
            ),
            (py, 'show("frequency", "cann", ["calibration", "marginal_drivage", "training"])'),
            (
                md,
                "## Exercise\nFreeze less: make the GLM part trainable (hint: subclass `CANN` and make"
                " the offset a learnable linear layer). Does joint training beat the offset version?",
            ),
        ],
    ),
    (
        "04_localglmnet.ipynb",
        "04 · LocalGLMnet: an interpretable network",
        r"$\log\lambda(x)=\beta_0+\sum_j\beta_j(x)\,x_j$: the network outputs one coefficient per"
        " covariate, so it reads like a GLM.",
        [
            (
                py,
                'run("frequency", models=["localglmnet"])\nleaderboard("frequency", ["localglmnet"])',
            ),
            (py, 'load("localglmnet.freq_reconciliation")'),
            (
                md,
                "## Attention weights and variable selection\n`RandN` is pure noise: its attention"
                " weights define the band (shaded) inside which a covariate is not significant.",
            ),
            (py, 'show("frequency", "localglmnet", ["attention", "importance"], width=820)'),
            (
                py,
                'm = load("localglmnet.freq_model"); test = load("freq_test")\n'
                'att = m.attentions(test); band = 2.576 * att["RandN"].std()\n'
                '(att.abs().mean() / band).sort_values().rename("mean |beta| / noise band").to_frame()',
            ),
            (
                md,
                "## Exercise\nDrop the covariates that sit inside the noise band (edit"
                " `frequency.models.localglmnet.features`) and re-run. Does the test deviance change?",
            ),
        ],
    ),
    (
        "05_mortality.ipynb",
        "05 · Mortality forecasting: Lee-Carter to deep learning",
        r"$\log m_{x,t}=a_x+b_xk_t$ versus CNN, LSTM and Transformer forecasters on 10-year windows"
        " of Swiss mortality (HMD). About 5 minutes on a CPU.",
        [
            (
                py,
                'models = ["lee_carter", "cnn", "cnn_tanh", "lstm", "transformer"]\n'
                'run("mortality")\nleaderboard("mortality", models)',
            ),
            (md, "## Open the box: Lee-Carter parameters"),
            (
                py,
                'import matplotlib.pyplot as plt\nh = load("lee_carter.mort_history")["Male"]\n'
                "fig, ax = plt.subplots(1, 3, figsize=(12, 3))\n"
                'ax[0].plot(h["ax"]); ax[0].set_title("a_x"); ax[1].plot(h["bx"]); ax[1].set_title("b_x")\n'
                'ax[2].plot(h["kt"]); ax[2].set_title("k_t (incl. forecast)"); plt.show()',
            ),
            (py, 'show("mortality", None, width=620)'),
            (py, 'show("mortality", "lstm", ["residuals", "training"])'),
            (
                md,
                "## Production checks\n* The reference notebook scales log-mortality using **all**"
                " years, including the test period: a look-ahead. Set"
                " `mortality.training.scaler_fit: learn` and re-run.\n"
                "* Forecasts are recursive: errors compound with the horizon (see `rmse_by_horizon`).",
            ),
            (
                md,
                '## Exercise\nRun both sexes (`run("mortality", env="scale_up")`) and move'
                ' the forecast origin: `params={"mortality.data.obs_year": 2009}`. Which model'
                " degrades least?",
            ),
        ],
    ),
    (
        "06_reserving.ipynb",
        "06 · Claims reserving: chain-ladder to neural ODP",
        "Chain-ladder per company versus a neural ODP model (a CANN with the chain-ladder as offset),"
        " back-tested on the **realised** outcomes in the CAS Schedule P data.",
        [
            (py, 'run("reserving")\nleaderboard("reserving", ["chain_ladder", "neural_odp"])'),
            (py, 'load("chain_ladder.res_reconciliation")'),
            (py, 'show("reserving", None, width=560)'),
            (md, "## Company-level back-test"),
            (
                py,
                'load("res_company_reserves").sort_values("actual_reserve", ascending=False).head(10)',
            ),
            (
                md,
                "## Exercise\nThe neural ODP stays at the chain-ladder: with a per-company"
                " chain-ladder offset there is nothing shared left to learn. Replace the offset with"
                " a **pooled, line-level** chain-ladder (one development pattern per line, scaled by"
                " each company's premium) in `models/reserving.py`, so the network learns company"
                ' deviations. Then scale up with `run("reserving", env="scale_up")`.',
            ),
        ],
    ),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, title, intro, cells in LESSONS:
        notebook(name, title, intro, cells)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
