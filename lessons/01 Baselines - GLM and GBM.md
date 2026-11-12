---
marp: true
theme: default
paginate: true
math: katex
title: "01 · Baselines: GLM and GBM"
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
# 01 · Baselines: GLM and GBM
### What every deep model has to beat

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) [![Open In SageMaker Studio](https://img.shields.io/badge/Open%20in-SageMaker%20Studio-FF9900)](https://console.aws.amazon.com/sagemaker/home#/studio-landing) · `kedro run --pipeline frequency --namespaces homogeneous,glm,glm3,gbm`

**Reproduces:** Wüthrich & Merz (2023) [[4](https://doi.org/10.1007/978-3-031-12409-9)], Tables 5.3 and 5.5

---

<!-- _class: theory -->

## Theory

**Model: Poisson GLM for claim frequency** [[1](https://doi.org/10.2307/2344614)]

```math
N_i \sim \mathrm{Poi}\big(v_i\thinspace\lambda(\boldsymbol{x}_i)\big), \qquad
\log \lambda^{\mathrm{GLM}}(\boldsymbol{x}) = \beta_0 + \langle \boldsymbol{\beta}, \boldsymbol{x}\rangle
```

where $N_i$ = number of claims of policy $i$; $v_i$ = exposure (years on cover); $\boldsymbol{x}_ i$ = its pre-processed covariates; $\lambda(\boldsymbol{x})$ = expected claims per year of cover; $\beta_0$ = intercept; $\boldsymbol{\beta}$ = coefficients, with $\langle \boldsymbol{\beta}, \boldsymbol{x}\rangle = \sum_j \beta_j x_j$

- **GLM1 / GLM3** [[4](https://doi.org/10.1007/978-3-031-12409-9)]: hand-engineered features; GLM3 adds polynomial driver age and bonus-malus interactions
- **GBM** (gradient boosting machine) [[2](https://doi.org/10.1214/aos/1013203451), [3](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)]: trees learn the shapes and interactions themselves

---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① Exposure is an offset</b></summary>

**Poisson GLM with log link**: $\log \mathbb{E}[N_i] = \log v_i + \eta(\boldsymbol{x}_ i)$

where $\eta(\boldsymbol{x}_ i) = \beta_0 + \langle\boldsymbol\beta, \boldsymbol{x}_ i\rangle$ = the linear predictor. $\log v_i$ has coefficient 1, so half a year on cover expects half the claims; $\exp(\beta_j)$ is a rating-table relativity.

</details>

<details>
<summary><b>② GLM1: the book's feature engineering</b></summary>

Wüthrich & Merz [[4](https://doi.org/10.1007/978-3-031-12409-9)], Sect. 5.2.4, $q + 1 = 49$ parameters: `Area` as a linear code 1–6; `VehPower` in 6 classes (≥ 9 merged); `VehAge` in [0, 6), [6, 13), [13, ∞); `DrivAge` in 7 bands, 31–40 the reference; `BonusMalus` linear, capped at 150; log `Density`; `VehBrand`, `Region`, `VehGas` dummy coded.

</details>

<details>
<summary><b>③ GLM3: smooth age, one interaction</b></summary>

**GLM3 driver-age terms** [[4](https://doi.org/10.1007/978-3-031-12409-9)], replacing the age bands ($q + 1 = 50$)

$\displaystyle \beta_1 a + \beta_2 \log a + \sum_{j=2}^{4} \beta_{j+1} a^j + \gamma_1\thinspace b\thinspace a + \gamma_2\thinspace b\thinspace a^2$

where $a$ = driver age; $b$ = bonus-malus level (capped at 150); $\gamma_1, \gamma_2$ = interaction coefficients

</details>

<details>
<summary><b>④ Maximum likelihood and the balance property</b></summary>

**Score equation for the intercept** of a Poisson GLM with log link

$\displaystyle \frac{\partial \ell}{\partial\beta_0} = \sum_i (N_i - \hat\mu_i) = 0$

where $\ell$ = Poisson log-likelihood; $\hat\mu_i$ = fitted claim count. The fitted GLM reproduces the total claim count on the learning data exactly; early-stopped networks do not.

</details>

<details>
<summary><b>⑤ Gradient boosting: many small trees</b></summary>

**GBM, additive on the log scale** [[2](https://doi.org/10.1214/aos/1013203451)]: $\eta_M(\boldsymbol{x}) = \eta_0 + \sum_{m=1}^{M} \nu\thinspace h_m(\boldsymbol{x})$

where $\eta_0$ = log exposure plus log portfolio frequency; $h_m$ = $m$-th regression tree, fitted to the Poisson gradient $N_i - \mu_i$; $\nu = 0.05$ = learning rate; $M$ = trees, chosen by early stopping (50 rounds) on 10% validation data. No published configuration exists on this split, so the GBM is on the leaderboard only.

</details>

<details>
<summary><b>⑥ Ranking vs level</b></summary>

- **Lorenz curve / Gini:** how well the model **ranks** risks
- **Calibration by decile:** whether the **level** is right in each risk band
- Deviance mixes both, so check both

</details>

---

## In code

```python
class GLM(FrequencyModel):
    def _fit(self, learn, params, base):
        d = add_glm_features(learn)        # bands, caps and log(Density) of [4], Sect. 5.2.4
        self.result = smf_glm(params["formula"], data=d, offset=np.log(d.Exposure),
                              family=sm.families.Poisson()).fit()
```

`frequency.models.glm.formula`: `ClaimNb ~ AreaGLM + VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region`

---

## Results: reconciled to Wüthrich & Merz (2023) [[4](https://doi.org/10.1007/978-3-031-12409-9)]

Poisson deviance per policy, $\times 10^{-2}$.

| Model | Paper, table | Published learn / test | This repo learn / test | Gini (test) | Status |
| --- | --- | --- | --- | --- | --- |
| Homogeneous | [[4](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.3 | 25.213 / 25.445 | <!--r:homogeneous.deviance_learn-->25.213<!--/r--> / <!--r:homogeneous.deviance_test-->25.445<!--/r--> | <!--r:homogeneous.gini_test-->0.000<!--/r--> | <!--r:homogeneous.status-->PASS<!--/r--> |
| GLM1 | [[4](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.3 | 24.101 / 24.146 | <!--r:glm.deviance_learn-->24.101<!--/r--> / <!--r:glm.deviance_test-->24.146<!--/r--> | <!--r:glm.gini_test-->0.305<!--/r--> | <!--r:glm.status-->PASS<!--/r--> |
| GLM3 | [[4](https://doi.org/10.1007/978-3-031-12409-9)], Table 5.5 | 24.084 / 24.102 | <!--r:glm3.deviance_learn-->24.084<!--/r--> / <!--r:glm3.deviance_test-->24.102<!--/r--> | <!--r:glm3.gini_test-->0.308<!--/r--> | <!--r:glm3.status-->PASS<!--/r--> |
| GBM | none published | n/a | <!--r:gbm.deviance_learn-->23.323<!--/r--> / <!--r:gbm.deviance_test-->23.550<!--/r--> | <!--r:gbm.gini_test-->0.353<!--/r--> | leaderboard only |

Brauer [[5](https://doi.org/10.1007/s13385-024-00388-2)], Table 2 (b)–(d), reproduces the same GLM numbers in Python.

GLM1 → GLM3: <!--r:gap.glm.glm3-->0.044 (0.019 to 0.069)<!--/r-->, <!--r:verdict.glm.glm3-->real<!--/r-->. The GBM beats every network of lessons 02–04, even the CANN averaged over 15 seeds, by <!--r:gap.cann_x15.gbm-->0.240 (0.169 to 0.311)<!--/r--> (gap in test deviance, 95% interval over test policies: lesson 00, Core ideas ⑥).

---

## Diagnostics

![w:470](figures/01-glm-drivage.png) ![w:470](figures/01-gbm-drivage.png)

GLM1's age bands are step functions; the GBM finds the young-driver shape itself.

---

## Ranking and calibration

![w:430](figures/01-lorenz.png) ![w:470](figures/01-glm-calibration.png)

---

## In production

- **Calibration:** actual vs expected by decile on the test set, not just deviance
- **Balance:** a GLM with intercept satisfies $\sum_i \hat\mu_i = \sum_i N_i$ on the learning data
- **GBM:** tune with early stopping on a validation split; monitor feature drift
- **Baseline first:** a network that cannot beat GLM3 and a tuned GBM is not worth deploying; here every network beats GLM3 and none beats the GBM

---

## Learn more

1. Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models. *JRSS A*, 135(3), 370–384. [doi:10.2307/2344614](https://doi.org/10.2307/2344614)
2. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
3. Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS 30*. [paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
4. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9) (open access)
5. Brauer, A. (2024). Enhancing actuarial non-life pricing models via transformers. *European Actuarial Journal*, 14(3), 991–1012. [doi:10.1007/s13385-024-00388-2](https://doi.org/10.1007/s13385-024-00388-2)

_Last reviewed: 2026-11-10_
