# deep-actuarial-science
> *Deep learning models in actuarial science: from new theory to a workable, benchmarked analysis, fast.*
> * *Each lesson is a short **slide deck** (Marp markdown, with LaTeX) backed by a model in a **Kedro pipeline**, a **Colab notebook**, **MLflow** tracking and a **published benchmark** the results reconcile to.*
> * *Realistic public data at switchable scale: 678k French motor policies, Swiss HMD mortality, US Schedule P triangles with realised outcomes. See [Open Actuarial Data](./Open%20Actuarial%20Data.md) for more sources.*
> * *New to Python, VS Code or Git? Start with [ds-training](https://github.com/actuarialsociety/ds-training).*

📖 **Read the lessons as a [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki)**: every page and the sidebar link straight to the lesson's notebook in Colab.

## Lessons

| Lesson | Model | Data | Reconciles to | Notebook |
| --- | --- | --- | --- | --- |
| [00 · Quickstart](./lessons/00%20Quickstart.md) | Pipelines, configs, tracking | freMTPL2 | AI Tools for Actuaries | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) |
| [01 · Baselines: GLM and GBM](./lessons/01%20Baselines%20-%20GLM%20and%20GBM.md) | Poisson GLM, LightGBM | freMTPL2 | GLM deviance (exact) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) |
| [02 · Neural networks and embeddings](./lessons/02%20Neural%20Networks%20-%20FNN%20and%20Embeddings.md) | FNN with entity embeddings | freMTPL2 | FNN deviance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) |
| [03 · CANN](./lessons/03%20CANN.md) | GLM nested in a network | freMTPL2 | CANN deviance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/03_cann.ipynb) |
| [04 · LocalGLMnet](./lessons/04%20LocalGLMnet.md) | Interpretable network, variable selection | freMTPL2 | LocalGLMnet deviance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) |
| [05 · Mortality: Lee–Carter to deep learning](./lessons/05%20Mortality%20-%20Lee-Carter%20to%20Deep%20Learning.md) | Lee–Carter, CNN, LSTM, Transformer | HMD Switzerland | Lee–Carter RMSE (exact) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/05_mortality.ipynb) |
| [06 · Reserving: chain-ladder to neural ODP](./lessons/06%20Reserving%20-%20Chain-Ladder%20to%20Neural%20ODP.md) | Chain-ladder, neural ODP (CANN) | CAS Schedule P | `chainladder` package; realised outcomes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/06_reserving.ipynb) |

## Reference
- [Open Actuarial Data](./Open%20Actuarial%20Data.md): public pricing, reserving and mortality datasets

## Run it

**Colab:** open any notebook above. It clones this repo, installs it and runs the same pipeline.

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
| Slides | `lessons/*.md` are Marp decks: preview with `npx @marp-team/marp-cli@4.5.1 -s lessons`; published to GitHub Pages |

## Maintaining these lessons
Browse the lessons as a [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki). It is regenerated from this repo on every merge to `main`.

This repo is set up for [Claude Code](https://code.claude.com/docs/en/overview) to add models and keep lessons current, using [Exa](https://exa.ai/docs/get-started/agent-skills/overview) agent skills for research. Rules are in [AGENTS.md](./AGENTS.md).
1. `cp .env.example .env` and add an [Exa API key](https://dashboard.exa.ai/api-keys), or `export EXA_API_KEY=...`
2. In Claude Code:
   - `/build-lesson <model or paper>`: research it, implement it in the pipeline, reconcile it to a benchmark, write the deck and notebook
   - `/refresh-lessons [file|all]`: fix links, re-run benchmarks, swap in newer research
3. CI runs the tests, a smoke run of every pipeline and every notebook on each PR; [`lesson-refresh.yml`](./.github/workflows/lesson-refresh.yml) opens a refresh PR monthly
