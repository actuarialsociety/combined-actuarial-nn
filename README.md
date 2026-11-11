# deep-actuarial-science
> *Deep learning in actuarial science, checked against the literature: each model reproduces a **named paper's published table** in a Kedro pipeline and reports PASS or FAIL.*
> * *Each reproduction has a short **lesson** (slides with the theory and the model's exact settings, also on the [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki)), a **notebook** and **MLflow** tracking.*
> * *Public data at switchable scale: 678k French motor policies, Swiss HMD mortality, US Schedule P triangles with realised outcomes. See [Open Actuarial Data](./Open%20Actuarial%20Data.md).*
> * *New to Python, VS Code or Git? Start with [ds-training](https://github.com/actuarialsociety/ds-training).*

## Findings

⟪count.PASS⟫ of the 25 published metrics reproduce; ⟪count.FAIL⟫ does not. A PASS says a number matches; the gaps below say whether a ranking means anything.

| | Finding | Evidence |
| --- | --- | --- |
| 1 | Closed-form models reproduce to rounding | GLM1 ⟪glm.deviance_test⟫, GLM3 ⟪glm3.deviance_test⟫, Lee–Carter ⟪lee_carter.rmse_out⟫, chain-ladder MAPE ⟪chain_ladder.mape_ultimate⟫ |
| 2 | Networks beat the best GLM | GLM3 → FNN: ⟪gap.glm3.fnn⟫ |
| 3 | FNN and CANN cannot be ranked on this test set | FNN → CANN: ⟪gap.fnn.cann⟫, ⟪verdict.fnn.cann⟫ |
| 4 | A plain gradient-boosting model beats every network, even seed-averaged | CANN ×15 → GBM: ⟪gap.cann_x15.gbm⟫ |
| 5 | The mortality networks' published runs are not typical of their seeds | Published run vs our 15 seeds: CNN ⟪cnn.published_run⟫, LSTM ⟪lstm.published_run⟫. Seeds beating Lee–Carter: CNN ⟪cnn.seeds_beating_lee_carter⟫, LSTM ⟪lstm.seeds_beating_lee_carter⟫, Transformer ⟪transformer.seeds_beating_lee_carter⟫, non-linear CNN ⟪cnn_tanh.seeds_beating_lee_carter⟫ |
| 6 | The neural ODP does not improve on the chain-ladder | MAPE gap, chain-ladder ahead: ⟪gap.neural_odp.chain_ladder⟫ |

**How to read a gap.** `A → B: gap (interval)` is A's test loss minus B's (Poisson deviance per policy ×10⁻², or absolute error of ultimate loss), with the 95% interval of the paired differences over test policies or companies. An interval containing 0 means another test sample could reverse the ranking. Method: [lesson 00](./lessons/00%20Quickstart.md), Core ideas ⑥; code: [`findings.py`](./src/deep_actuarial_science/findings.py).

**Where the numbers come from.** [`reference/`](./reference) is the frozen reference run: every metric, the reconciliation, the findings and the package versions. Every number on this page and in the lessons is filled from it, CI fails if one disagrees, and each notebook shows it beside the paper and your own run.

## Benchmarks: paper → table → reproduction

| Lesson | Model | Reproduces (paper, table) | Result | Notebook |
| --- | --- | --- | --- | --- |
| [00 · Quickstart](./lessons/00%20Quickstart.md) | Pipelines, configs, reconciliation, findings | All of the below | | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) |
| [01 · Baselines: GLM and GBM](./lessons/01%20Baselines%20-%20GLM%20and%20GBM.md) | Poisson GLM1, GLM3; LightGBM | Wüthrich & Merz (2023), Tables 5.3, 5.5 (GBM: none published) | ⟪glm.status⟫, ⟪glm3.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) |
| [02 · Neural networks and embeddings](./lessons/02%20Neural%20Networks%20-%20FNN%20and%20Embeddings.md) | FNN with entity embeddings | Brauer (2024), Table 2 (f), 15 seeds | ⟪fnn.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) |
| [03 · CANN](./lessons/03%20CANN.md) | GLM3 nested in a network | Brauer (2024), Table 2 (g), 15 seeds | ⟪cann.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/03_cann.ipynb) |
| [04 · LocalGLMnet](./lessons/04%20LocalGLMnet.md) | Interpretable network, variable selection | Richman & Wüthrich (2023), Table 2 (c) | ⟪localglmnet.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) |
| [05 · Mortality: Lee–Carter to deep learning](./lessons/05%20Mortality%20-%20Lee-Carter%20to%20Deep%20Learning.md) | Lee–Carter, CNNs, LSTM, Transformer | Richman, Scognamiglio & Wüthrich (2026), *AI Tools for Actuaries* Ch. 8, pp. 50, 56 | ⟪lee_carter.status⟫; CNN ⟪cnn.status⟫, non-linear CNN ⟪cnn_tanh.status⟫, LSTM ⟪lstm.status⟫, Transformer ⟪transformer.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/05_mortality.ipynb) |
| [06 · Reserving: chain-ladder to neural ODP](./lessons/06%20Reserving%20-%20Chain-Ladder%20to%20Neural%20ODP.md) | Chain-ladder, neural ODP (CANN) | Kuo (2019), Table 1, on Meyers' (2015) companies (neural ODP: none published) | ⟪chain_ladder.status⟫ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/06_reserving.ipynb) |

Published values, tolerances and full citations: [`conf/base/parameters_benchmarks.yml`](./conf/base/parameters_benchmarks.yml). Every notebook also opens in Kaggle and SageMaker Studio from its first cell's badges. Reproduce everything on a free GPU: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/reference_run.ipynb)

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
- [MAINTENANCE.md](./MAINTENANCE.md): rules for every commit, the reference-run workflow, known issues
- [CHANGELOG.md](./CHANGELOG.md): what changed, and which results moved

## Run it

**In the cloud, no setup:** open a notebook from the table. Its first cell clones this repo, installs it and runs the same pipeline as `kedro run`.

| Where | How | GPU | MLflow UI |
| --- | --- | --- | --- |
| Colab | Colab badge | T4, requested automatically | inline and new tab |
| Kaggle | Kaggle badge; turn on *Internet* | *GPU T4* in the settings | none: `mlflow.search_runs()` |
| SageMaker Studio | badge opens the console; Git → Clone this repo, run a notebook | e.g. `ml.g4dn.xlarge` | new tab (Jupyter proxy) |
| Azure ML | compute instance → `git clone`, run a notebook | e.g. `Standard_NC4as_T4_v3` | new tab (needs `jupyter-server-proxy`) |

**Container** (also a VS Code dev container):
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

- **Apple silicon:** `DAS_DEVICE=mps` trains on the Apple GPU (opt-in: these networks are small, so MPS is no faster than the CPU; results agree to about 0.002 deviance). [`notebooks/run_all_mps.ipynb`](./notebooks/run_all_mps.ipynb) runs and times everything.
- **Weights & Biases (optional):** `uv pip install -e ".[wandb]"`, set `WANDB_PROJECT`; MLflow logging is unchanged.
- **Full run:** `scripts/full_run.sh` runs the reference setup (hours: 15 seeds per network), freezes it into `reference/`, then runs every dataset (`conf/scale_up`). See [MAINTENANCE.md](./MAINTENANCE.md).

## How it fits together

| Piece | Where |
| --- | --- |
| Models | `src/deep_actuarial_science/models/` (`frequency.py`, `mortality.py`, `reserving.py`): each docstring names its paper and every deviation from it |
| Pipelines | `src/deep_actuarial_science/pipelines/`: `ingest`, `frequency`, `mortality`, `reserving`; one namespace per model, plus a comparison node |
| Configs | `conf/base/parameters_*.yml` (data scale, model settings); `conf/smoke/` for fast runs |
| Benchmarks | `conf/base/parameters_benchmarks.yml`: published values, tolerances and sources; each model writes a PASS/FAIL `reconciliation.csv` |
| Findings | `findings.py`: which gaps between models are real, where a published run falls among our seeds; each pipeline writes `findings.csv` |
| Reference run | `reference/`: the frozen full run behind every published number (`scripts/freeze_reference.py`) |
| Tracking | kedro-mlflow: parameters, metrics, figures, models and benchmark tables for every run (`mlflow.db`) |
| Outputs | `data/08_reporting/<pipeline>/`: metrics, figures, leaderboards, findings |
| Notebooks | `notebooks/`, generated by `scripts/make_notebooks.py`; they call the pipelines through `deep_actuarial_science.notebook` and show paper, reference run and your run side by side |
| Slides | `lessons/*.md` are Marp decks: `npx @marp-team/marp-cli@4.5.1 --html -s lessons`; published to GitHub Pages and the [wiki](https://github.com/actuarialsociety/deep-actuarial-science/wiki) |

## Maintaining

Rules for every commit and the reference-run workflow are in [MAINTENANCE.md](./MAINTENANCE.md); agent rules in [AGENTS.md](./AGENTS.md). This repo is set up for [Claude Code](https://code.claude.com/docs/en/overview) with [Exa](https://exa.ai/docs/get-started/agent-skills/overview) research skills (`cp .env.example .env`, add an [Exa API key](https://dashboard.exa.ai/api-keys)):

- `/build-research-benchmark-wiki <paper, model or DOI>`: find the paper's table and reference implementation, implement it as the paper does, reconcile (PASS/FAIL), publish the lesson, notebook and wiki page
- `/build-lesson <model>`: a teaching lesson without a paper to reproduce
- `/refresh-lessons [file|all]`: fix links, re-run benchmarks, swap in newer research
