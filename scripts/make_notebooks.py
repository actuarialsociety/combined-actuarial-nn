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

GPU_GUARD = """# Colab must run on a GPU runtime: stop here otherwise (set DAS_ALLOW_CPU=1 to override).
if "google.colab" in sys.modules and os.environ.get("DAS_ALLOW_CPU") != "1":
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This notebook needs a GPU runtime. In Colab: Runtime > Change runtime type > "
            "T4 GPU > Save, then Runtime > Run all. (To run on CPU anyway: "
            "import os; os.environ['DAS_ALLOW_CPU'] = '1')"
        )
    print("GPU:", torch.cuda.get_device_name(0))
"""

SETUP = f"""# Setup: runs anywhere Jupyter runs. Opened on its own (Colab, Kaggle, SageMaker, Azure ML,
# any hosted Jupyter), this cell clones and installs the repo; in a clone or the container it
# uses that checkout.
import os, sys, importlib.util
{GPU_GUARD}
REPO_DIR = next((d for d in (os.getcwd(), os.path.dirname(os.getcwd()))
                 if os.path.isdir(os.path.join(d, "src", "deep_actuarial_science"))), None)
if REPO_DIR is None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Needs Python 3.10+: choose a newer kernel or image")
    # a writable, persistent home: Colab, Kaggle, SageMaker notebook instances, else ~
    base = next(d for d in ("/content", "/kaggle/working", os.path.expanduser("~/SageMaker"),
                            os.path.expanduser("~")) if os.path.isdir(d) and os.access(d, os.W_OK))
    REPO_DIR = os.path.join(base, "deep-actuarial-science")
    if not os.path.exists(REPO_DIR):  # Kaggle: turn on Internet in the notebook settings
        !git clone -q https://github.com/{REPO} {{REPO_DIR}}
    if importlib.util.find_spec("deep_actuarial_science") is None:
        !pip install -q -e {{REPO_DIR}}
os.chdir(REPO_DIR)
sys.path.insert(0, "src")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from deep_actuarial_science.notebook import run, load, show, leaderboard, parameters, mlflow_ui, device_summary
device_summary()
FAST = os.environ.get("DAS_ENV") == "smoke"   # CI runs every notebook in smoke mode
"""


GITHUB_PATH = f"github/{REPO}/blob/main/notebooks"
# Launchers. Azure ML and SageMaker Studio have no open-from-GitHub link: the SageMaker badge
# opens the Studio console; clone the repo there (or upload the notebook) and the setup cell
# does the rest. (SageMaker Studio Lab, which had one, is closed to new customers.)
LAUNCHERS = {
    "Colab": (
        "https://colab.research.google.com/assets/colab-badge.svg",
        "https://colab.research.google.com/{path}/{name}",
    ),
    "Kaggle": (
        "https://kaggle.com/static/images/open-in-kaggle.svg",
        "https://kaggle.com/kernels/welcome?src=https://{path_gh}/{name}",
    ),
    "SageMaker Studio": (
        "https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900",
        "https://console.aws.amazon.com/sagemaker/home#/studio-landing",
    ),
}


def launch_url(where: str, name: str) -> str:
    path_gh = f"github.com/{REPO}/blob/main/notebooks"
    return LAUNCHERS[where][1].format(path=GITHUB_PATH, path_gh=path_gh, name=name)


def badge(name: str, where: str = "Colab") -> str:
    return f"[![Open In {where}]({LAUNCHERS[where][0]})]({launch_url(where, name)})"


def badges(name: str) -> str:
    return " ".join(badge(name, w) for w in LAUNCHERS)


RUN_ANYWHERE = (
    "_Runs anywhere: **Colab** opens on a GPU runtime (T4) and the first cell stops on a CPU"
    " runtime; on **Kaggle** turn on *GPU T4* and *Internet* in the notebook settings; in"
    " **SageMaker Studio** (the badge opens the console) or **Azure ML**, pick a GPU instance,"
    " clone the repo (Git → Clone) or upload this notebook, and run all: the first cell installs"
    " what it needs. Any other Jupyter works the same way._"
)


def notebook(
    name: str, title: str, intro: str, cells: list[tuple[str, str]], local: bool = False
) -> None:
    """Write one notebook. `local` notebooks (Apple silicon) get no cloud badges or metadata."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        # Colab opens these on a GPU runtime (free T4); training uses it automatically.
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        # Kaggle: GPU and internet (for the clone) when it imports the notebook
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
    }
    if local:
        nb.metadata = {k: nb.metadata[k] for k in ("kernelspec", "language_info")}
    header = f"# {title}\n\n{intro}" if local else f"# {title}\n{badges(name)}\n\n{intro}\n\n{RUN_ANYWHERE}"
    nb.cells = [
        nbf.v4.new_markdown_cell(header),
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
    "Parameters, metrics, figures and the benchmark table of every run above. MLflow runs on"
    " the notebook's machine: in Colab it opens below and in a new tab through Colab's proxy;"
    " in SageMaker and Azure ML through Jupyter's proxy; on Kaggle, which exposes no ports,"
    " query the runs with `mlflow.search_runs` instead."
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
                'run("frequency", models=["glm3", "cann"])\nleaderboard("frequency", ["glm3", "cann"])',
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


REFERENCE_RUN = (
    "reference_run.ipynb",
    "Reference run: reproduce every published benchmark",
    "Runs every pipeline at the reference setup, reconciles each model to its paper's table"
    " (`conf/base/parameters_benchmarks.yml`), refreshes the lesson figures and numbers and"
    " zips the results. Use a **GPU runtime**: the frequency networks fit 15 seeds each. Results come back to the repo as a PR.",
    [
        (md, "## 1. Run every pipeline\nFrequency is the long one (15 seeds for each network)."),
        (py, 'for p in ("frequency", "mortality", "reserving"):\n    run(p)'),
        (md, "## 2. Reconciliation: every model against its paper"),
        (
            py,
            "import glob, pandas as pd\n"
            "rec = pd.concat(pd.read_csv(f) for f in sorted(glob.glob('data/08_reporting/*/*/reconciliation.csv')))\n"
            "rec[['model', 'metric', 'expected', 'actual', 'tolerance', 'status', 'source']]",
        ),
        (md, "## 3. Refresh the lesson figures and numbers"),
        (py, "!python scripts/export_figures.py\n!python scripts/fill_results.py"),
        (
            md,
            "## 4. Download and open a PR\nUnzip `reference_run.zip` at the repo root, check"
            " `git diff lessons/`, then commit on a branch and open a PR.",
        ),
        (
            py,
            "import shutil, sys\n"
            "shutil.make_archive('reference_run', 'zip', '.', 'lessons')\n"
            "if 'google.colab' in sys.modules:\n"
            "    from google.colab import files\n"
            "    files.download('reference_run.zip')\n"
            "else:  # Kaggle: Output tab; SageMaker / Azure ML / Jupyter: file browser\n"
            "    print('Download', os.path.abspath('reference_run.zip'))",
        ),
    ],
)


RUN_ALL_MPS = (
    "run_all_mps.ipynb",
    "Run every lesson on Apple silicon (MPS)",
    "Runs every lesson's pipeline on a Mac with an M-series chip, training the networks on"
    " its GPU through PyTorch's Metal (MPS) backend, then reconciles every model to its paper"
    " and times MPS against the CPU. Open it in a local clone (see *Local* in the README:"
    " `uv venv`, `uv pip install -e \".[dev]\"`, then `jupyter lab`). Elsewhere it falls back"
    " to the default device.\n\n"
    "**Expect MPS to be no faster:** these networks are small, so GPU kernel-launch overhead"
    " outweighs the speedup (section 4 measures it on your machine). The full reference run"
    " fits 15 seeds per frequency network and takes hours; set `ENV = \"smoke\"` below for a"
    " two-minute pass on 2% of the data. To log to Weights & Biases as well, set"
    " `WANDB_PROJECT` (and `WANDB_API_KEY`) before starting Jupyter; `scripts/full_run.sh` is"
    " the same full run from the terminal.",
    [
        (md, "## 1. Train on the Apple GPU"),
        (
            py,
            "import time, torch\n"
            "ENV = \"smoke\" if FAST else \"local\"   # \"smoke\": every pipeline on 2% of the data\n"
            "MPS = torch.backends.mps.is_available()\n"
            "if MPS:\n"
            "    os.environ[\"DAS_DEVICE\"] = \"mps\"   # read by every network at training time\n"
            "else:\n"
            "    print(\"No MPS (not an Apple-silicon Mac, or PyTorch without Metal): using the default device.\")\n"
            "device_summary()",
        ),
        (
            md,
            "## 2. Run every pipeline\nFrequency (GLMs, LightGBM, FNN, CANN, LocalGLMnet),"
            " mortality (Lee–Carter and the deep models) and reserving (chain-ladder, neural"
            " ODP), each tracked in MLflow. LightGBM trains single-threaded on macOS, because its"
            " OpenMP runtime and PyTorch's crash in one process.",
        ),
        (
            py,
            "timings = {}\n"
            "for p in (\"frequency\", \"mortality\", \"reserving\"):\n"
            "    t0 = time.perf_counter()\n"
            "    run(p, env=ENV)\n"
            "    timings[p] = time.perf_counter() - t0\n"
            "    print(f\"{p}: {timings[p] / 60:.1f} min\")",
        ),
        (md, "## 3. Reconciliation: every model against its paper"),
        (
            py,
            "import glob, pandas as pd\n"
            "rec = pd.concat(pd.read_csv(f) for f in sorted(glob.glob('data/08_reporting/*/*/reconciliation.csv')))\n"
            "rec[['model', 'metric', 'expected', 'actual', 'tolerance', 'status', 'source']]",
        ),
        (
            md,
            "## 4. MPS against the CPU\nThe same FNN (one seed, 10% of the data) trained on"
            " each device. GPU and CPU results agree to about 0.002 Poisson deviance.",
        ),
        (
            py,
            "small = {} if FAST else {\"frequency.data.scale\": 0.1, \"frequency.models.fnn.n_runs\": 1}\n"
            "rows = []\n"
            "for dev in ([\"cpu\", \"mps\"] if MPS else [\"cpu\"]):\n"
            "    os.environ[\"DAS_DEVICE\"] = dev\n"
            "    t0 = time.perf_counter()\n"
            "    run(\"frequency\", models=[\"fnn\"], params=small, env=ENV)\n"
            "    m = load(\"fnn.freq_metrics\", env=ENV)\n"
            "    rows.append({\"device\": dev, \"wall_seconds\": time.perf_counter() - t0, **m})\n"
            "os.environ[\"DAS_DEVICE\"] = \"mps\" if MPS else \"cpu\"\n"
            "# fit_seconds is training alone; wall_seconds adds data preparation and logging\n"
            "pd.DataFrame(rows).set_index(\"device\")[[\"fit_seconds\", \"wall_seconds\", \"deviance_test\", \"gini_test\", \"balance_test\"]]",
        ),
    ],
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, title, intro, cells in [*LESSONS, REFERENCE_RUN]:
        notebook(name, title, intro, cells)
        print("wrote", OUT / name)
    notebook(*RUN_ALL_MPS, local=True)
    print("wrote", OUT / RUN_ALL_MPS[0])


if __name__ == "__main__":
    main()
