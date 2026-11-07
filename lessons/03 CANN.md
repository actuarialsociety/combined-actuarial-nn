---
marp: true
theme: default
paginate: true
math: katex
title: "03 · CANN"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
  section:has(details[open]) details:not([open]) { display: none; }
  section.theory { font-size: 23px; }
  section.theory p { margin: .45em 0; }
---

<!-- _class: lead -->
# 03 · CANN
### Nest the GLM inside a network: keep the baseline, learn what it misses

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/03_cann.ipynb) · `kedro run --pipeline frequency --namespaces glm,cann`

---

<!-- _class: theory -->

## Theory

**Model: Combined Actuarial Neural Network (CANN)** [[1](https://doi.org/10.1017/asb.2018.42), [2](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3320525)]

```math
\lambda^{\mathrm{CANN}}(\boldsymbol{x}) =
\underbrace{\lambda^{\mathrm{GLM}}(\boldsymbol{x})}_{\text{offset}}\medspace
\exp\big\lbrace w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x})\rangle\big\rbrace
```

where $\lambda^{\mathrm{CANN}}(\boldsymbol{x})$ = expected claims per year of cover for covariates $\boldsymbol{x}$; $\lambda^{\mathrm{GLM}}$ = the fitted GLM of lesson 01, kept fixed; $\boldsymbol{z}^{(d:1)}(\boldsymbol{x})$ = last hidden layer of the lesson 02 network; $w_0, \boldsymbol{w}$ = output bias and weights; $\exp\lbrace\cdot\rbrace$ = the network's multiplicative correction to the GLM

- **Zero initialisation:** $w_0 = 0,\ \boldsymbol{w} = \boldsymbol{0}$ ⇒ training starts *exactly* at the GLM [[1](https://doi.org/10.1017/asb.2018.42), [2](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3320525)]
- The network term is a **multiplicative correction**: it shows where the GLM is wrong
- If training barely moves it, the GLM was adequate

---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① Learn the residual on the log scale</b></summary>

**CANN on the log scale**: fixed GLM plus a learned correction

$\displaystyle \log\lambda^{\mathrm{CANN}}(\boldsymbol{x}) = \underbrace{\log\lambda^{\mathrm{GLM}}(\boldsymbol{x})}_ {\text{fixed}} + \underbrace{w_0 + \langle\boldsymbol{w}, \boldsymbol{z}^{(d)}(\boldsymbol{x})\rangle}_ {\text{learned correction}}$

where $\lambda^{\mathrm{GLM}}$ = frequency of the fitted GLM (lesson 01); $\boldsymbol{z}^{(d)}(\boldsymbol{x})$ = last hidden layer of the network; $w_0, \boldsymbol{w}$ = output bias and weights

The GLM is fitted first and then **frozen**: its expected counts, exposure included, enter the network as the offset. Only the correction term is trained.

</details>

<details>
<summary><b>② Zero initialisation: start exactly at the GLM</b></summary>

With $w_0 = 0$ and $\boldsymbol{w} = \boldsymbol{0}$ the correction is $\exp(0) = 1$, so before any training (**CANN at initialisation**)

$\displaystyle \lambda^{\mathrm{CANN}}(\boldsymbol{x}) = \lambda^{\mathrm{GLM}}(\boldsymbol{x}) .$

Training can therefore only move away from the GLM if that lowers the loss. `test_cann_starts_exactly_at_glm` checks this property on every CI run.

</details>

<details>
<summary><b>③ Same network, different starting point</b></summary>

The CANN reuses lesson 02's network exactly: same inputs and embeddings, layers [20, 15, 10], NAdam, batch 5,000, 200 epochs with early stopping. The only differences are the offset (GLM prediction instead of exposure) and the initial bias (0 instead of $\log\bar\lambda$).

</details>

<details>
<summary><b>④ What the correction can pick up</b></summary>

- **Missing covariates:** the reference GLM leaves out `BonusMalus`, `VehAge` and `VehPower`; the network sees all of them
- **Shape inside bands:** the GLM's driver-age bands are flat; the network can bend within them
- **Interactions** between any covariates, which the GLM formula does not contain

</details>

<details>
<summary><b>⑤ Reading the correction</b></summary>

For each policy, $c(\boldsymbol{x}) = \exp(w_0 + \langle\boldsymbol{w}, \boldsymbol{z}^{(d)}(\boldsymbol{x})\rangle)$ is the factor applied on top of the GLM price.

- $c \approx 1$ everywhere: the GLM was already adequate
- Segments with $c$ far from 1 show where the GLM is wrong, and are candidates for new GLM terms
- Track the distribution of $c$ by segment over time as a monitoring signal

</details>

<details>
<summary><b>⑥ Balance after early stopping</b></summary>

The GLM is balanced, but the correction is not constrained to average to 1, so the CANN's total prediction drifts (here <!--r:cann.balance_learn-->0.9833<!--/r-->× observed on the learning data). Restore it with the same bias shift as lesson 02, $w_0 \leftarrow w_0 + \log(\sum_i N_i / \sum_i \hat\mu_i)$, before using the model for pricing.

</details>

---

## In code

```python
class CANN(FNN):                     # same network as lesson 02 …
    uses_base = True                 # … but the offset is the fitted GLM's expected counts

def _offset(self, df, base):
    return base.predict(df) if self.uses_base else df.Exposure.to_numpy()

nn.init.zeros_(self.out.weight)      # output layer starts at zero
nn.init.constant_(self.out.bias, 0)  # → CANN(x) = GLM(x) before training
```

`tests/test_core.py::test_cann_starts_exactly_at_glm` checks the zero-initialisation property.

---

## Results: reconciled to the benchmark

| Model | Learn | Test | Published test | Status |
| --- | --- | --- | --- | --- |
| GLM | <!--r:glm.deviance_learn-->46.954<!--/r--> | <!--r:glm.deviance_test-->47.179<!--/r--> | 47.179 | <!--r:glm.status-->PASS<!--/r--> |
| CANN | <!--r:cann.deviance_learn-->44.989<!--/r--> | <!--r:cann.deviance_test-->44.976<!--/r--> | 44.976 | <!--r:cann.status-->PASS<!--/r--> |
| FNN | <!--r:fnn.deviance_learn-->44.973<!--/r--> | <!--r:fnn.deviance_test-->44.928<!--/r--> | 44.928 | <!--r:fnn.status-->PASS<!--/r--> |
| GBM (lesson 01) | <!--r:gbm.deviance_learn-->44.144<!--/r--> | <!--r:gbm.deviance_test-->44.394<!--/r--> | n/a | leaderboard only |

![w:520](figures/03-leaderboard.png)

The tuned GBM beats all three networks here: a network has to earn its place.

---

## Where the network corrects the GLM

![w:470](figures/03-cann-drivage.png) ![w:470](figures/03-cann-training.png)

The CANN keeps the GLM's age bands as its starting point and corrects within and across them, using every covariate.

---

## In production

- **Governance:** the GLM stays filed and explainable; the CANN is a documented uplift
- **Calibration:** here the CANN's predictions sum to <!--r:cann.balance_learn-->0.9833<!--/r--> × observed claims on the learning set; restore the balance property before pricing [[3](https://doi.org/10.1080/24754269.2021.1877960)]
- **Extensions:** telematics CANNs put new data in the network term only [[4](https://doi.org/10.1017/asb.2024.4)]
- **Monitoring:** track the size of the correction $\exp\lbrace\cdot\rbrace$ by segment over time

---

## Learn more

1. Wüthrich, M. V., & Merz, M. (2019). Editorial: Yes, we CANN! *ASTIN Bulletin*, 49(1), 1–3. [doi:10.1017/asb.2018.42](https://doi.org/10.1017/asb.2018.42)
2. Schelldorfer, J., & Wüthrich, M. V. (2019). Nesting classical actuarial models into neural networks. *SSRN 3320525*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3320525)
3. Wüthrich, M. V. (2022). The balance property in neural network modelling. *Statistical Theory and Related Fields*, 6(1), 1–9. [doi:10.1080/24754269.2021.1877960](https://doi.org/10.1080/24754269.2021.1877960)
4. Duval, F., Boucher, J.-P., & Pigeon, M. (2024). Telematics combined actuarial neural networks for cross-sectional and longitudinal claim count data. *ASTIN Bulletin*, 54(2), 239–262. [doi:10.1017/asb.2024.4](https://doi.org/10.1017/asb.2024.4)

_Last reviewed: 2026-11-06_
