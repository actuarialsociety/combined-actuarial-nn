# deep-actuarial-science
> *A research-paper benchmark toolkit for deep learning in actuarial science: each model reproduces a **named paper's published table** in a Kedro pipeline and reports PASS or FAIL.*
> * *Each reproduction comes with a short **lesson** (Marp slides with LaTeX, also on the [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki)), a **notebook** that runs on Colab, Kaggle, SageMaker, Azure ML or any Jupyter (GPU when available) and **MLflow** tracking.*
> * *Public data at switchable scale: 678k French motor policies, Swiss HMD mortality, US Schedule P triangles with realised outcomes. See [Open Actuarial Data](./Open%20Actuarial%20Data.md).*
> * *New to Python, VS Code or Git? Start with [ds-training](https://github.com/actuarialsociety/ds-training).*

## Benchmarks: paper → table → reproduction

| Lesson | Model | Reproduces (paper, table) | Notebook |
| --- | --- | --- | --- |
| [00 · Quickstart](./lessons/00%20Quickstart.md) | Pipelines, configs, reconciliation | All of the below | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [01 · Baselines: GLM and GBM](./lessons/01%20Baselines%20-%20GLM%20and%20GBM.md) | Poisson GLM1, GLM3; LightGBM | Wüthrich & Merz (2023), Tables 5.3, 5.5 (GBM: none published) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [02 · Neural networks and embeddings](./lessons/02%20Neural%20Networks%20-%20FNN%20and%20Embeddings.md) | FNN with entity embeddings | Brauer (2024), Table 2 (f), 15 seeds | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [03 · CANN](./lessons/03%20CANN.md) | GLM3 nested in a network | Brauer (2024), Table 2 (g), 15 seeds | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/03_cann.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/03_cann.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [04 · LocalGLMnet](./lessons/04%20LocalGLMnet.md) | Interpretable network, variable selection | Richman & Wüthrich (2023), Table 2 (c) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [05 · Mortality: Lee–Carter to deep learning](./lessons/05%20Mortality%20-%20Lee-Carter%20to%20Deep%20Learning.md) | Lee–Carter, CNNs, LSTM, Transformer | Richman, Scognamiglio & Wüthrich (2026), *AI Tools for Actuaries* Ch. 8, pp. 50, 56 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/05_mortality.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/05_mortality.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |
| [06 · Reserving: chain-ladder to neural ODP](./lessons/06%20Reserving%20-%20Chain-Ladder%20to%20Neural%20ODP.md) | Chain-ladder, neural ODP (CANN) | Kuo (2019), Table 1, on Meyers' (2015) companies (neural ODP: none published) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/06_reserving.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/06_reserving.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) |

Values, tolerances and full citations: [`conf/base/parameters_benchmarks.yml`](./conf/base/parameters_benchmarks.yml). Reproduce everything on a free GPU: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/reference_run.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/reference_run.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing)

## The course in seven ideas

_Click a panel to open it. Each lesson opens with its own **Core ideas** panels that go deeper._

<details>
<summary><b>00 · Same question, same score, reproducible answer</b></summary>

Every model predicts expected claims $\mu_i = v_i\thinspace\lambda(\boldsymbol{x}_ i)$ and is scored by out-of-sample Poisson deviance on one fixed test set. Each result is checked against a published number before anything is changed.
→ [00 · Quickstart](./lessons/00%20Quickstart.md)

</details>

<details>
<summary><b>01 · Know the baseline you have to beat</b></summary>

A Poisson GLM with log link and exposure offset is interpretable and balanced, but its features are made by hand. A tuned GBM learns thresholds and interactions on its own and sets a high bar.
→ [01 · Baselines](./lessons/01%20Baselines%20-%20GLM%20and%20GBM.md)

</details>

<details>
<summary><b>02 · Let the network learn the features</b></summary>

A feed-forward network is a GLM on learned features. Entity embeddings handle categories, early stopping regularises, and a bias shift restores the balance property.
→ [02 · Neural networks](./lessons/02%20Neural%20Networks%20-%20FNN%20and%20Embeddings.md)

</details>

<details>
<summary><b>03 · Keep the GLM, learn what it misses</b></summary>

The CANN uses the frozen GLM as an offset and a zero-initialised network as a multiplicative correction, so training starts exactly at the GLM.
→ [03 · CANN](./lessons/03%20CANN.md)

</details>

<details>
<summary><b>04 · A network that reads like a GLM</b></summary>

LocalGLMnet outputs one coefficient $\beta_j(\boldsymbol{x})$ per covariate and policy. A pure-noise covariate shows which effects are real.
→ [04 · LocalGLMnet](./lessons/04%20LocalGLMnet.md)

</details>

<details>
<summary><b>05 · Forecast mortality from the whole age profile</b></summary>

Lee–Carter extends one time index linearly. Networks map the last ten years of all ages to the next year and forecast recursively, so errors compound with horizon.
→ [05 · Mortality](./lessons/05%20Mortality%20-%20Lee-Carter%20to%20Deep%20Learning.md)

</details>

<details>
<summary><b>06 · Anchor reserves to the chain-ladder, then back-test</b></summary>

The chain-ladder is an over-dispersed Poisson GLM. A neural correction on top starts at it, and the realised lower triangle shows whether the correction helps.
→ [06 · Reserving](./lessons/06%20Reserving%20-%20Chain-Ladder%20to%20Neural%20ODP.md)

</details>

## Reference
- [Open Actuarial Data](./Open%20Actuarial%20Data.md): public pricing, reserving and mortality datasets

## Run it

**In the cloud, no setup:** open any notebook in the benchmarks table; `reference_run.ipynb` reproduces every benchmark. The first cell clones this repo, installs it and runs the same pipeline, wherever it runs.

| Where | How | GPU | MLflow UI |
| --- | --- | --- | --- |
| Colab | Colab badge | T4, requested automatically | inline and new tab (Colab proxy) |
| Kaggle | Kaggle badge; turn on *Internet* in the notebook settings | *GPU T4* in the settings | none (no ports): `mlflow.search_runs()` |
| SageMaker Studio | SageMaker Studio badge opens the console (Studio has no open-from-GitHub link): open JupyterLab, Git → Clone `https://github.com/actuarialsociety/deep-actuarial-science`, open a notebook, run all | choose a GPU instance (e.g. `ml.g4dn.xlarge`) | new tab (Jupyter proxy) |
| Azure Machine Learning | compute instance → terminal → `git clone` the repo, open a notebook, run all (Azure has no open-from-GitHub link) | choose a GPU size (e.g. `Standard_NC4as_T4_v3`) | new tab (Jupyter proxy, needs `jupyter-server-proxy`) |

Agent skills for running on AWS and Azure are vendored in `.claude/skills/` (`scripts/update_aws_skills.sh`, `scripts/update_azure_skills.sh`): SageMaker and Azure ML jobs, AWS sign-in, Azure GPU quotas, and cost estimates on both.

**Container** (recommended locally; also a VS Code dev container):
```bash
docker compose up lab mlflow          # JupyterLab http://localhost:8888, MLflow http://localhost:5000
docker compose run --rm pipeline      # full reference run, tracked in MLflow
```

**Local** with [uv](https://docs.astral.sh/uv/):
```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch && uv pip install -e ".[dev]"
kedro run --env smoke                                    # every pipeline on 2% of the data, ~1-2 min
kedro run --pipeline frequency --namespaces glm,cann     # selected models, full data
kedro run --pipeline frequency --params frequency.data.scale=0.1
mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse runs
```

## How it fits together

| Piece | Where |
| --- | --- |
| Models | `src/deep_actuarial_science/models/` (`frequency.py`, `mortality.py`, `reserving.py`) |
| Pipelines | `src/deep_actuarial_science/pipelines/`: `ingest`, `frequency`, `mortality`, `reserving`; one namespace per model, plus a comparison node |
| Configs | `conf/base/parameters_*.yml` (data scale, model settings); `conf/smoke/` for fast runs |
| Benchmarks | `conf/base/parameters_benchmarks.yml`: published values, tolerances and sources; each model writes a PASS/FAIL `reconciliation.csv` |
| Tracking | kedro-mlflow: parameters, metrics, figures, models and benchmark tables for every run (`mlflow.db`) |
| Outputs | `data/08_reporting/<pipeline>/`: metrics, figures, leaderboards |
| Notebooks | `notebooks/`, generated by `scripts/make_notebooks.py`; they call the pipelines through `deep_actuarial_science.notebook` |
| Slides | `lessons/*.md` are Marp decks: preview with `npx @marp-team/marp-cli@4.5.1 --html -s lessons` (`--html` makes the Core ideas panels clickable); published to GitHub Pages |

## Maintaining these lessons
Browse the lessons as a [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki). It is regenerated from this repo on every merge to `main`.

This repo is set up for [Claude Code](https://code.claude.com/docs/en/overview) to add models and keep lessons current, using [Exa](https://exa.ai/docs/get-started/agent-skills/overview) agent skills for research. Rules are in [AGENTS.md](./AGENTS.md).
1. `cp .env.example .env` and add an [Exa API key](https://dashboard.exa.ai/api-keys), or `export EXA_API_KEY=...`
2. In Claude Code:
   - `/build-research-benchmark-wiki <paper, model or DOI>`: find the paper's results table and reference implementation with Exa, implement it as the paper does, reconcile to the published scores (PASS/FAIL), and publish the lesson, notebook and wiki page
   - `/build-lesson <model>`: a teaching lesson without a paper to reproduce
   - `/refresh-lessons [file|all]`: fix links, re-run benchmarks, swap in newer research
3. CI runs the tests, a smoke run of every pipeline and every notebook on each PR; [`lesson-refresh.yml`](./.github/workflows/lesson-refresh.yml) opens a refresh PR monthly
