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
  section:has(details[open]) details:not([open]) { display: none; }
  section.theory { font-size: 23px; }
  section.theory p { margin: .45em 0; }
  table { font-size: .68em; }
---

<!-- _class: lead -->
# 00 · Quickstart
### Published paper → pipeline → reproduced result, PASS or FAIL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/00_quickstart.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) · `kedro run --env smoke`

---

## How every lesson works

1. **Paper:** a published model and the table it reports
2. **Model:** implemented as in the paper, in `src/deep_actuarial_science/models/`
3. **Config:** `conf/base/parameters_<domain>.yml`; the paper's numbers in `parameters_benchmarks.yml`
4. **Run:** `kedro run --pipeline <domain> --namespaces <model>`
5. **Reconcile:** this repo vs the paper's table → PASS / FAIL, tracked in MLflow

---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① One question, many models</b></summary>

Claim counts $N_i \sim \mathrm{Poi}(\mu_i)$ with $\mu_i = v_i\thinspace\lambda(\boldsymbol{x}_ i)$; lessons differ only in how the frequency $\lambda(\boldsymbol{x})$ is built: homogeneous → GLM → GBM → FNN → CANN → LocalGLMnet. Mortality (05) and reserving (06) reuse the pattern on time series and triangles.

</details>

<details>
<summary><b>② One score: the Poisson deviance</b></summary>

**Poisson deviance per policy**, the scale of the published tables [[2](https://doi.org/10.1007/978-3-031-12409-9)]

$\displaystyle D = \frac{1}{n}\sum_{i=1}^n v_i\thinspace d\Big(\frac{N_i}{v_i}, \frac{\mu_i}{v_i}\Big) \times 100, \qquad d(y,m) = 2\Big(m - y - y\log\frac{m}{y}\Big)$

where $n$ = number of policies; $N_i$ = claims of policy $i$; $v_i$ = its exposure; $\mu_i$ = its predicted claim count; $d(y, m)$ = unit deviance of observed frequency $y$ against predicted $m$

- Twice the log-likelihood gap to a perfect fit: **minimum deviance = maximum likelihood**; lower is better
- Dividing by total exposure instead of $n$ multiplies $D$ by $n / \sum_i v_i \approx 1.89$ here; pipelines report both

</details>

<details>
<summary><b>③ The papers' learn/test split</b></summary>

Fixed split of Wüthrich & Merz [[2](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.2: 610,206 learning and 67,801 test policies. Networks validate on 10% of the learning data; the test set is only used to report results.

</details>

<details>
<summary><b>④ Reconciliation to a named table</b></summary>

**Reconciliation rule**, per benchmarked metric

$\displaystyle \text{PASS} \iff |\text{this repo} - \text{paper}| \le \text{tolerance}$

where the tolerance is rounding for closed-form fits and twice the published seed standard deviation for networks. Benchmarks apply only at the reference setup; smaller runs report `SKIPPED`.

</details>

<details>
<summary><b>⑤ Reproducible, on CPU or GPU</b></summary>

Settings live in config, seeds are fixed, MLflow logs every run. Networks train on a GPU when one is present (`DAS_DEVICE`); `notebooks/reference_run.ipynb` runs everything on a free Colab GPU.

</details>

---

## Run it three ways

```bash
# Colab: open a notebook from the README (GPU runtime); reference_run.ipynb reproduces everything
docker compose up lab mlflow                            # container: JupyterLab :8888, MLflow :5000
kedro run --env smoke                                   # local: all pipelines, 2% data, ~2 min
kedro run --pipeline frequency --namespaces glm,glm3    # chosen models, full data
```

---

## Every model, one leaderboard

![w:640](figures/00-leaderboard.png)

Out-of-sample Poisson deviance per policy on 67,801 test policies. Diamonds: the published value.

---

## Reconciliation: every number has a paper

| Model | Paper, table | Published test | This repo | Status |
| --- | --- | --- | --- | --- |
| Homogeneous | Wüthrich & Merz [[2](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.3 | 25.445 | <!--r:homogeneous.deviance_test-->pending<!--/r--> | <!--r:homogeneous.status-->pending<!--/r--> |
| GLM1 | Wüthrich & Merz [[2](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.3 | 24.146 | <!--r:glm.deviance_test-->pending<!--/r--> | <!--r:glm.status-->pending<!--/r--> |
| GLM3 | Wüthrich & Merz [[2](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.5 | 24.102 | <!--r:glm3.deviance_test-->pending<!--/r--> | <!--r:glm3.status-->pending<!--/r--> |
| FNN (15 seeds) | Brauer [[3](https://doi.org/10.1007/s13385-024-00388-2)], Table 2 (f) | 23.827 | <!--r:fnn.deviance_test-->pending<!--/r--> | <!--r:fnn.status-->pending<!--/r--> |
| CANN (15 seeds) | Brauer [[3](https://doi.org/10.1007/s13385-024-00388-2)], Table 2 (g) | 23.810 | <!--r:cann.deviance_test-->pending<!--/r--> | <!--r:cann.status-->pending<!--/r--> |
| LocalGLMnet | Richman & Wüthrich [[4](https://doi.org/10.1080/03461238.2022.2081816)], Table 2 (c) | 23.945 | <!--r:localglmnet.deviance_test-->pending<!--/r--> | <!--r:localglmnet.status-->pending<!--/r--> |

Mortality (lesson 05) and reserving (lesson 06) reconcile the same way.

---

## Add a paper in four steps

1. Implement the model as in the paper; register it in the domain's `MODELS`
2. Add `<domain>.models.<name>` parameters
3. Add the paper's value, table and tolerance under `benchmarks.<domain>.<name>`
4. `kedro run --pipeline <domain> --namespaces <name>`, or `/build-lesson <paper>` in Claude Code

---

## Learn more

1. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [doi:10.2139/ssrn.5162304](https://doi.org/10.2139/ssrn.5162304)
2. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9) (open access)
3. Brauer, A. (2024). Enhancing actuarial non-life pricing models via transformers. *European Actuarial Journal*, 14(3), 991–1012. [doi:10.1007/s13385-024-00388-2](https://doi.org/10.1007/s13385-024-00388-2)
4. Richman, R., & Wüthrich, M. V. (2023). LocalGLMnet: interpretable deep learning for tabular data. *Scandinavian Actuarial Journal*, 2023(1), 71–95. [doi:10.1080/03461238.2022.2081816](https://doi.org/10.1080/03461238.2022.2081816)
5. [Kedro documentation](https://docs.kedro.org/) · [MLflow documentation](https://mlflow.org/docs/latest/) · [Marp](https://marp.app/)

_Last reviewed: 2026-11-07_
