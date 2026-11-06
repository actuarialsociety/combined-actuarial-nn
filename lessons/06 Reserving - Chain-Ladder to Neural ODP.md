---
marp: true
theme: default
paginate: true
math: katex
title: "06 · Reserving: chain-ladder to neural ODP"
---

<!-- _class: lead -->
# 06 · Reserving: chain-ladder to neural ODP
### A CANN for triangles, back-tested on what was actually paid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/06_reserving.ipynb) · `kedro run --pipeline reserving`

---

## Theory

$$
\mathbb{E}[X_{i,j}] = \mu_{i,j} = \exp(\alpha_i + \beta_j), \quad
\mathrm{Var}(X_{i,j}) = \varphi\,\mu_{i,j}
\qquad \text{(ODP = chain-ladder [1, 2])}
$$

$$
\log \mu^{(c)}_{i,j} = \log \mu^{\mathrm{CL},(c)}_{i,j} + \mathrm{NN}\big(e_c,\, e_{\ell},\, e_j,\, i\big),
\qquad \mathrm{NN}\equiv 0 \text{ at initialisation}
$$

- A CANN in the spirit of [3]: one network across **all companies** $c$ and lines $\ell$
- Payments scaled by company premium; early stopping on the **latest diagonal**

---

## The data: CAS Schedule P, with the answer key

- NAIC Schedule P paid triangles, accident years 1988–1997, <!--r:chain_ladder.n_companies-->50<!--/r--> private passenger auto companies
- The database also holds the **realised lower triangle**: every reserve is back-tested
- Chain-ladder reconciles to the `chainladder` package: max relative difference <!--r:chain_ladder.package_diff-->3.8e-15<!--/r-->

![w:520](figures/06-development.png)

---

## Results: back-test on realised payments

| Model | Total reserve error | Median abs. company error | Companies within ±10% |
| --- | --- | --- | --- |
| Chain-ladder | <!--r:chain_ladder.total_reserve_error_pct-->+10.8<!--/r-->% | <!--r:chain_ladder.median_abs_error_pct-->20.0<!--/r-->% | <!--r:chain_ladder.share_within_10pct-->24%<!--/r--> |
| Neural ODP | <!--r:neural_odp.total_reserve_error_pct-->+11.0<!--/r-->% | <!--r:neural_odp.median_abs_error_pct-->20.3<!--/r-->% | <!--r:neural_odp.share_within_10pct-->22%<!--/r--> |

- Chain-ladder **over-reserves** for most companies; neural ODP stays at the chain-ladder
- Why: early stopping never left the zero initialisation (best epoch <!--r:neural_odp.best_epoch-->0<!--/r-->): with a
  per-company chain-ladder offset there is no shared structure left to learn. Same result on all
  260 companies across six lines (`kedro run --pipeline reserving --env scale_up`)

---

## Company-level view

![w:440](figures/06-scatter.png) ![w:480](figures/06-errors.png)

---

## In production

- **Back-test before trusting:** hold out diagonals; compare to realised development
- **The chain-ladder anchor** keeps results explainable to a reserving committee
- **Uncertainty:** Mack [4] or bootstrap for the chain-ladder; ensembles for the network
- **Give the network something to learn:** use a pooled, line-level chain-ladder as the
  offset so company deviations are learned (closer to [3]); that is this lesson's exercise
- **Fully neural alternatives** (e.g. DeepTriangle [5]) drop the anchor: benchmark both

---

## Learn more

1. Renshaw, A. E., & Verrall, R. J. (1998). A stochastic model underlying the chain-ladder technique. *British Actuarial Journal*, 4(4), 903–923. [doi:10.1017/S1357321700000222](https://doi.org/10.1017/S1357321700000222)
2. Mack, T. (1993). Distribution-free calculation of the standard error of chain ladder reserve estimates. *ASTIN Bulletin*, 23(2), 213–225. [doi:10.2143/AST.23.2.2005092](https://doi.org/10.2143/AST.23.2.2005092)
3. Gabrielli, A., Richman, R., & Wüthrich, M. V. (2020). Neural network embedding of the over-dispersed Poisson reserving model. *Scandinavian Actuarial Journal*, 2020(1), 1–29. [doi:10.1080/03461238.2019.1633394](https://doi.org/10.1080/03461238.2019.1633394)
4. Gabrielli, A. (2020). A neural network boosted double overdispersed Poisson claims reserving model. *ASTIN Bulletin*, 50(1), 25–60. [doi:10.1017/asb.2019.33](https://doi.org/10.1017/asb.2019.33)
5. Kuo, K. (2019). DeepTriangle: a deep learning approach to loss reserving. *Risks*, 7(3), 97. [doi:10.3390/risks7030097](https://doi.org/10.3390/risks7030097)

Data: [CAS Loss Reserving Database](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p).

_Last reviewed: 2026-11-06_
