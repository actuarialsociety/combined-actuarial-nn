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
---

<!-- _class: lead -->
# 01 · Baselines: GLM and GBM
### What every deep model has to beat

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/01_glm_gbm_baselines.ipynb) · `kedro run --pipeline frequency --namespaces homogeneous,glm,gbm`

---

## Theory

$$
N_i \sim \mathrm{Poi}\big(v_i\thinspace\lambda(\boldsymbol{x}_i)\big), \qquad
\log \lambda^{\mathrm{GLM}}(\boldsymbol{x}) = \beta_0 + \langle \boldsymbol{\beta}, \boldsymbol{x}\rangle
$$

$$
D = \frac{2}{\sum_i v_i}\sum_i v_i \Big[\tfrac{\mu_i}{v_i} - \tfrac{N_i}{v_i} - \tfrac{N_i}{v_i}\log\tfrac{\mu_i}{N_i}\Big] \times 100
$$

- **GLM** [1]: interpretable, but features are hand-engineered (age bands, log density)
- **GBM** [2, 3]: learns interactions with trees; Poisson loss with $\log v_i$ as offset


---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① Exposure enters as an offset</b></summary>

With a log link, the expected count of policy $i$ is

$$
\log \mathbb{E}[N_i] = \log v_i + \eta(\boldsymbol{x}_i), \qquad \eta = \log\lambda .
$$

$\log v_i$ has a fixed coefficient of 1 (an **offset**), so a policy on cover for half a year expects half the claims. Covariates act **multiplicatively** on frequency: $\exp(\beta_j)$ is a relativity, the familiar form of a rating table.

</details>

<details>
<summary><b>② Fitting = maximum likelihood = minimum deviance</b></summary>

Up to terms that do not depend on the model, the Poisson log-likelihood is

$$
\ell(\boldsymbol\beta) = \sum_i \big(N_i \log\mu_i - \mu_i\big), \qquad \mu_i = v_i\exp(\beta_0 + \langle\boldsymbol\beta, \boldsymbol{x}_i\rangle).
$$

The deviance is $2(\ell_{\text{saturated}} - \ell)$, so the parameters that maximise $\ell$ also minimise the deviance on the learning data. statsmodels solves this with iteratively reweighted least squares.

</details>

<details>
<summary><b>③ The GLM's features are made by hand</b></summary>

`add_glm_features` builds the design the GLM can use:

- `DrivAge` in 7 bands (18–20, …, 71+), with 31–40 as the reference level
- `log(Density)`; `Area` as an ordinal code 1–6; `VehBrand` and `VehGas` as categories
- The reference formula `DrivAgeGLM + VehBrand + VehGas + DensityGLM + AreaGLM` reproduces the published deviance. It leaves out `BonusMalus`, `VehAge` and `VehPower`, which the GBM and the networks do use

Every band and transformation is a modelling decision. Deep models aim to learn these from the data.

</details>

<details>
<summary><b>④ The balance property</b></summary>

For a Poisson GLM with log link and an intercept, the likelihood equation for $\beta_0$ is

$$
\frac{\partial \ell}{\partial\beta_0} = \sum_i (N_i - \hat\mu_i) = 0 .
$$

The fitted GLM therefore reproduces the total observed claim count on the learning data exactly (balance = 1.0000). Models fitted another way, such as networks with early stopping, lose this property and must be checked for it.

</details>

<details>
<summary><b>⑤ Gradient boosting: many small trees</b></summary>

$$
\eta_M(\boldsymbol{x}) = \eta_0 + \sum_{m=1}^{M} \nu\thinspace h_m(\boldsymbol{x})
$$

- Each tree $h_m$ is fitted to the negative gradient of the Poisson loss, which on the log scale is $N_i - \mu_i$: the trees chase the remaining residuals
- $\eta_0 = \log v_i + \log\bar\lambda$: exposure offset plus the portfolio frequency
- Here: learning rate $\nu = 0.05$, 31 leaves, at least 1,000 policies per leaf, 80% of features per tree; stop after 50 rounds without improvement on the validation set
- Trees find thresholds and interactions themselves, which is why the GBM beats the GLM

</details>

<details>
<summary><b>⑥ Ranking vs level</b></summary>

- **Lorenz curve / Gini:** sort policies from highest to lowest predicted frequency and plot the cumulative share of claims against the cumulative share of exposure. A higher Gini means better **ranking** of risks
- **Calibration by decile:** within each predicted-risk decile, compare observed and predicted frequency. This checks the **level**
- A model can rank well and still be mis-priced, or the reverse. Deviance mixes both, so look at both

</details>

---

## In code

```python
class GLM(FrequencyModel):
    def _fit(self, learn, params, base):
        d = add_glm_features(learn)           # bands, log(Density), capped BonusMalus
        self.result = smf_glm(params["formula"], data=d, offset=np.log(d.Exposure),
                              family=sm.families.Poisson()).fit()

class GBM(FrequencyModel):                    # LightGBM, objective="poisson"
    ...  init_score = np.log(exposure) + log(mean frequency)   # offset
```

`frequency.models.glm.formula: "ClaimNb ~ DrivAgeGLM + VehBrand + VehGas + DensityGLM + AreaGLM"`

---

## Results: reconciled to the benchmark

| Model | Learn | Test | Published test | Gini (test) | Status |
| --- | --- | --- | --- | --- | --- |
| Homogeneous | <!--r:homogeneous.deviance_learn-->47.722<!--/r--> | <!--r:homogeneous.deviance_test-->47.967<!--/r--> | 47.967 | <!--r:homogeneous.gini_test-->0.082<!--/r--> | <!--r:homogeneous.status-->PASS<!--/r--> |
| GLM | <!--r:glm.deviance_learn-->46.954<!--/r--> | <!--r:glm.deviance_test-->47.179<!--/r--> | 47.179 | <!--r:glm.gini_test-->0.172<!--/r--> | <!--r:glm.status-->PASS<!--/r--> |
| GBM | <!--r:gbm.deviance_learn-->44.144<!--/r--> | <!--r:gbm.deviance_test-->44.394<!--/r--> | n/a | <!--r:gbm.gini_test-->0.353<!--/r--> | leaderboard only |

Closed-form fits reproduce the published deviances to three decimals [4].

---

## Diagnostics

![w:470](figures/01-glm-drivage.png) ![w:470](figures/01-gbm-drivage.png)

The GLM's age bands are step functions; the GBM finds the young-driver shape itself.

---

## Ranking and calibration

![w:430](figures/01-lorenz.png) ![w:470](figures/01-glm-calibration.png)

---

## In production

- **Calibration:** check actual vs expected by decile on the test set, not just deviance
- **Balance:** the GLM with intercept satisfies $\sum_i \hat\mu_i = \sum_i N_i$ on learn data
- **GBM:** tune with early stopping on a validation split; monitor feature drift
- **Baseline first:** a network that cannot beat a tuned GLM/GBM is not worth deploying

---

## Learn more

1. Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models. *JRSS A*, 135(3), 370–384. [doi:10.2307/2344614](https://doi.org/10.2307/2344614)
2. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
3. Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS 30*. [paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
4. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9)
5. Noll, A., Salzmann, R., & Wüthrich, M. V. (2018). Case study: French motor third-party liability claims. *SSRN 3164764*. [doi:10.2139/ssrn.3164764](https://doi.org/10.2139/ssrn.3164764)

_Last reviewed: 2026-11-06_
