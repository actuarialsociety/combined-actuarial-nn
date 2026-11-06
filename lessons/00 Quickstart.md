---
marp: true
theme: default
paginate: true
math: katex
title: "00 · Quickstart"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
---

<!-- _class: lead -->
# 00 · Quickstart
### Theory → pipeline → reconciled result, in one command

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) · `kedro run --env smoke`


---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① Every model answers the same question</b></summary>

Claim counts are modelled as $N_i \sim \mathrm{Poi}(\mu_i)$ with $\mu_i = v_i\thinspace\lambda(\boldsymbol{x}_i)$: exposure $v_i$ (policy years) times a frequency $\lambda(\boldsymbol{x})$ that depends on the covariates. The lessons differ only in how $\lambda(\boldsymbol{x})$ is built:

homogeneous (one number) → GLM (hand-made features) → GBM (trees) → FNN (learned features) → CANN (GLM + learned correction) → LocalGLMnet (learned, but readable as coefficients).

Mortality (lesson 05) and reserving (lesson 06) reuse the same pattern on time series and triangles.

</details>

<details>
<summary><b>② One score for all: the Poisson deviance</b></summary>

$$
D = \frac{1}{\sum_i v_i}\sum_i v_i\thinspace d\Big(\frac{N_i}{v_i}, \frac{\mu_i}{v_i}\Big) \times 100,
\qquad d(y,m) = 2\Big(m - y - y\log\frac{m}{y}\Big)
$$

- $d$ is twice the log-likelihood gap to a model that fits every observation exactly, so **minimising deviance = maximum likelihood**
- Lower is better. Values are reported in units of $10^{-2}$, as in the published benchmarks
- It rewards both ranking risks correctly and getting the level right. The other lessons separate the two with Lorenz curves (ranking) and calibration plots (level)

</details>

<details>
<summary><b>③ Learn, validation and test sets</b></summary>

- **Test** (67,801 policies): a fixed hold-out, used only to report results, never to choose anything
- **Learn** (the rest): used to fit
- **Validation**: 10% of *learn*, used by GBMs and networks to decide when to stop training

Using the same split as the published notebooks is what makes the results comparable to their numbers.

</details>

<details>
<summary><b>④ Reconciliation: proving the implementation first</b></summary>

`conf/base/parameters_benchmarks.yml` holds, per model, the published value, a tolerance and the source. After each run:

$$
\text{PASS} \iff |\text{this repo} - \text{published}| \le \text{tolerance}
$$

Benchmarks apply only at the reference setup (e.g. `scale: 1.0`); smaller experiments report `SKIPPED`. A model that reproduces a published number is a trustworthy starting point for your own changes. Where no like-for-like number exists (reserving), the lesson back-tests against realised outcomes instead.

</details>

<details>
<summary><b>⑤ Reproducible by construction</b></summary>

- **Config, not code:** data scale, architectures and epochs live in `conf/base/parameters_*.yml`
- **Seeds** are fixed per model, so a rerun gives the same numbers
- **MLflow** logs parameters, metrics, figures and the reconciliation table for every run
- **Same pipeline everywhere:** Colab, the container and a local clone all run `kedro run`

</details>

<details>
<summary><b>⑥ What a pricing team checks beyond deviance</b></summary>

- **Calibration:** actual vs expected by predicted-risk decile
- **Balance:** total predicted claims = total observed claims on the learning data
- **Ranking:** Lorenz curve and Gini index
- **Stability:** refit with other seeds; average fits before pricing
- **Look-ahead:** nothing from the test period may enter training or scaling

</details>

---

## The workflow every lesson follows

1. **Theory:** one or two equations
2. **Model:** a class in `src/deep_actuarial_science/models/`, registered in a pipeline
3. **Config:** `conf/base/parameters_<domain>.yml` (model settings, data scale)
4. **Run:** `kedro run --pipeline <domain> --namespaces <model>`
5. **Reconcile:** results vs the published benchmark → `reconciliation.csv` (PASS / FAIL)
6. **Track:** parameters, metrics, figures, model and benchmark table in MLflow

---

## Run it three ways

```bash
# Colab: open a notebook from the README (clones, installs, runs the same pipeline)

# Container: JupyterLab + MLflow UI
docker compose up lab mlflow            # :8888 and :5000

# Local
kedro run --env smoke                                   # all pipelines, 2% data, ~2 min
kedro run --pipeline frequency --namespaces glm,cann    # chosen models, full data
kedro run --pipeline frequency --params frequency.data.scale=0.1
```

---

## Every model, one leaderboard

![w:640](figures/00-leaderboard.png)

Out-of-sample Poisson deviance on 67,801 test policies. Diamonds: published benchmark.

---

## Reconciliation: the result is reproducible

| Model | Published test | This repo | Status |
| --- | --- | --- | --- |
| Homogeneous | 47.967 | <!--r:homogeneous.deviance_test-->47.967<!--/r--> | <!--r:homogeneous.status-->PASS<!--/r--> |
| GLM | 47.179 | <!--r:glm.deviance_test-->47.179<!--/r--> | <!--r:glm.status-->PASS<!--/r--> |
| FNN | 44.928 | <!--r:fnn.deviance_test-->44.928<!--/r--> | <!--r:fnn.status-->PASS<!--/r--> |
| CANN | 44.976 | <!--r:cann.deviance_test-->44.976<!--/r--> | <!--r:cann.status-->PASS<!--/r--> |
| LocalGLMnet | 45.034 | <!--r:localglmnet.deviance_test-->45.034<!--/r--> | <!--r:localglmnet.status-->PASS<!--/r--> |

Benchmarks: Wüthrich et al. (2025), *AI Tools for Actuaries*, notebook outputs [1].

---

## Change anything with config, never code

```yaml
# conf/base/parameters_frequency.yml
frequency:
  data:
    scale: 1.0            # 0.01-0.5 for quick experiments; 1.0 = benchmark setup
  models:
    cann:
      hidden: [20, 15, 10]
      epochs: 200
```

`kedro run --params frequency.models.cann.hidden=[32,16]` overrides one key for one run.

---

## Add a new model in four steps

1. Subclass `FrequencyModel` (or the mortality/reserving equivalent) and register it
2. Add `frequency.models.<name>` parameters
3. Add the namespace to the pipeline's `MODELS`, and a published benchmark if one exists
4. `kedro run --pipeline frequency --namespaces <name>`

Or in Claude Code: `/build-lesson <paper or model>`.

---

## Learn more

1. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)
2. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9) (open access)
3. [Kedro documentation](https://docs.kedro.org/) · [MLflow documentation](https://mlflow.org/docs/latest/) · [Marp](https://marp.app/)

_Last reviewed: 2026-11-06_
