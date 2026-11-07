---
marp: true
theme: default
paginate: true
math: katex
title: "05 · Mortality: Lee–Carter to deep learning"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
  section:has(details[open]) details:not([open]) { display: none; }
  section.theory { font-size: 23px; }
  section.theory p { margin: .45em 0; }
---

<!-- _class: lead -->
# 05 · Mortality: Lee–Carter to deep learning
### Forecast 22 years of Swiss mortality and compare four network types with Lee–Carter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/05_mortality.ipynb) · `kedro run --pipeline mortality`

---

<!-- _class: theory -->

## Theory

**Model 1: Lee–Carter** [[1](https://doi.org/10.1080/01621459.1992.10475265)]

```math
\log m_{x,t} = a_x + b_x k_t, \qquad k_{t+1} = k_t + \delta + \varepsilon_{t+1}
```

where $m_{x,t}$ = central death rate at age $x$ in calendar year $t$ (deaths ÷ person-years lived); $a_x$ = average log death rate at age $x$; $k_t$ = period index: the general mortality level in year $t$; $b_x$ = how strongly age $x$ follows $k_t$; $\delta$ = drift, the average yearly change in $k_t$; $\varepsilon_{t+1}$ = random error

**Model 2: neural forecaster** [[2](https://doi.org/10.1080/03461238.2020.1867232)]

```math
\log m_{\cdot,t} = f_\theta\big(\log m_{\cdot,t-10},\dots,\log m_{\cdot,t-1}\big)
```

where $\log m_{\cdot,t}$ = the vector of log death rates for all ages 0–89 in year $t$; $f_\theta$ = a network with weights $\theta$ (CNN, CNN with tanh, LSTM or Transformer) that maps the previous 10 years to the next one

- Forecasts are **recursive**: each predicted year feeds the next prediction

---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① The data: log death rates by age and year</b></summary>

$m_{x,t}$ is the central death rate at age $x$ in year $t$: deaths over person-years lived. The models work with $\log m_{x,t}$ for ages 0–89 and years 1900–2021. Rates of exactly zero (young ages, small cells) are floored at $e^{-11}$ before taking logs.

- Learn: 1900–1999 · Test: 2000–2021 (a 22-year forecast)

</details>

<details>
<summary><b>② Lee–Carter: one time index drives all ages</b></summary>

**Lee–Carter model with its identifiability constraints** [[1](https://doi.org/10.1080/01621459.1992.10475265)]

$\displaystyle \log m_{x,t} = a_x + b_x k_t, \qquad \sum_x b_x = 1,\quad \sum_t k_t = 0$

where $m_{x,t}$ = death rate at age $x$, year $t$; $a_x$ = age profile; $b_x$ = age sensitivity; $k_t$ = period index. It is fitted in three steps:

1. $a_x$ = average of $\log m_{x,t}$ over the learning years (the age profile)
2. A singular value decomposition of the centred matrix gives the best rank-1 fit $b_x k_t$
3. The two constraints make the split between $b_x$ and $k_t$ unique

$b_x$ says how fast each age improves when the general level $k_t$ moves.

</details>

<details>
<summary><b>③ Lee–Carter forecast: extend the trend</b></summary>

$k_t$ is treated as a random walk with drift, $k_{t+1} = k_t + \delta + \varepsilon_{t+1}$. The point forecast (**Lee–Carter forecast**) is

$\displaystyle \hat k_{T+h} = k_T + h\thinspace\hat\delta ,$

where $T$ = last learning year (1999); $h$ = years ahead (1 to 22); $\hat\delta$ is the slope of a straight line fitted to $k_t$ over the learning years. Every age then improves at its own fixed rate $b_x\hat\delta$ per year.

</details>

<details>
<summary><b>④ The neural forecasters: next year from the last ten</b></summary>

- Each training sample is a **window**: 10 consecutive years × 90 ages as input, the next year's 90 rates as target [[2](https://doi.org/10.1080/03461238.2020.1867232)]
- $\log m$ is min–max scaled to $[0, 1]$; loss = mean squared error
- Windows are shuffled; 90% train, 10% validation. Adam, learning rate 0.001, batch size 4, up to 1,000 epochs; the rate is cut when validation stalls and the best epoch is kept

</details>

<details>
<summary><b>⑤ Four ways to read a window</b></summary>

- **CNN:** a 1-D convolution over time (3 filters, width 3), with the 90 ages as channels, then a dense layer to 90 outputs
- **CNN (tanh):** first compresses each year's 90 ages to 20 features with tanh, then convolves and adds a tanh dense layer
- **LSTM:** compresses each year to 20 features, runs a 20-unit LSTM through the 10 years and maps its last state to 90 ages
- **Transformer:** compresses each year to 3 features and applies one self-attention block, so each year can weigh every other year, with a skip connection, layer normalisation and a feed-forward layer

</details>

<details>
<summary><b>⑥ Recursive forecasts and how they are scored</b></summary>

From 2000 on, each prediction is appended to the window and fed back in to predict the following year: 22 steps, each built on earlier predictions, so **errors compound** with horizon.

**Root mean squared error (RMSE)** of log death rates

$\displaystyle \mathrm{RMSE} = \sqrt{\frac{1}{|S|}\sum_{(x,t)\in S}\big(\log m_{x,t} - \widehat{\log m}_ {x,t}\big)^2}$

where $\widehat{\log m}_ {x,t}$ = predicted log death rate; $S$ = the set of (age, year) cells scored, $|S|$ = how many. $S$ is every age and year in the test period (out-of-sample) or the learning period (in-sample).

</details>

---

## The data: HMD Switzerland, males, 1900–2021

![w:700](figures/05-observed.png)

Learn 1900–1999, forecast 2000–2021 (dashed line).

---

## Results: reconciled to the benchmark

| Model | RMSE in-sample | RMSE out-of-sample | Published out | Status |
| --- | --- | --- | --- | --- |
| Lee–Carter | <!--r:lee_carter.rmse_in-->0.1555<!--/r--> | <!--r:lee_carter.rmse_out-->0.3425<!--/r--> | 0.3425 | <!--r:lee_carter.status-->PASS<!--/r--> |
| CNN | <!--r:cnn.rmse_in-->0.1808<!--/r--> | <!--r:cnn.rmse_out-->0.4281<!--/r--> | 0.4281 | <!--r:cnn.status-->PASS<!--/r--> |
| CNN (tanh) | <!--r:cnn_tanh.rmse_in-->0.1570<!--/r--> | <!--r:cnn_tanh.rmse_out-->0.3535<!--/r--> | 0.3533 | <!--r:cnn_tanh.status-->PASS<!--/r--> |
| LSTM | <!--r:lstm.rmse_in-->0.1595<!--/r--> | <!--r:lstm.rmse_out-->0.3428<!--/r--> | 0.3428 | <!--r:lstm.status-->PASS<!--/r--> |
| Transformer | <!--r:transformer.rmse_in-->0.1819<!--/r--> | <!--r:transformer.rmse_out-->0.3831<!--/r--> | 0.3831 | <!--r:transformer.status-->PASS<!--/r--> |

RMSE of $\log m_{x,t}$. Lee–Carter is closed-form and reproduces the benchmark exactly [[3](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)].

---

## Where forecasts go wrong

![w:470](figures/05-lc-residuals.png) ![w:470](figures/05-lstm-residuals.png)

Right of the dashed line: the LSTM forecasts too high at ages 15–40 (blue) and too low at the youngest and oldest ages (red). Note the 1918 influenza spike in the training data.

---

## Forecast quality by horizon

![w:470](figures/05-horizon.png) ![w:470](figures/05-age-profile.png)

---

## In production

- **Look-ahead:** the reference scales $\log m$ with all years, test included. Set
  `mortality.training.scaler_fit: learn` for an honest forecast
- **Recursive error growth:** errors compound with horizon; report by horizon, not one number
- **Uncertainty:** point forecasts are not enough for reserving; bootstrap intervals [[4](https://doi.org/10.1017/asb.2021.34)]
- **Many populations:** jointly trained networks with embeddings beat per-country Lee–Carter fits [[5](https://doi.org/10.1017/S1748499519000071), [6](https://doi.org/10.1017/asb.2022.5)]

---

## Learn more

1. Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *JASA*, 87(419), 659–671. [doi:10.1080/01621459.1992.10475265](https://doi.org/10.1080/01621459.1992.10475265)
2. Perla, F., Richman, R., Scognamiglio, S., & Wüthrich, M. V. (2021). Time-series forecasting of mortality rates using deep learning. *Scandinavian Actuarial Journal*, 2021(7), 572–598. [doi:10.1080/03461238.2020.1867232](https://doi.org/10.1080/03461238.2020.1867232)
3. Wüthrich, M. V., Richman, R., et al. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)
4. Schnürch, S., & Korn, R. (2022). Point and interval forecasts of death rates using neural networks. *ASTIN Bulletin*, 52(1), 333–360. [doi:10.1017/asb.2021.34](https://doi.org/10.1017/asb.2021.34)
5. Richman, R., & Wüthrich, M. V. (2021). A neural network extension of the Lee–Carter model to multiple populations. *Annals of Actuarial Science*, 15(2), 346–366. [doi:10.1017/S1748499519000071](https://doi.org/10.1017/S1748499519000071)
6. Scognamiglio, S. (2022). Calibrating the Lee–Carter and the Poisson Lee–Carter models via neural networks. *ASTIN Bulletin*, 52(2), 519–561. [doi:10.1017/asb.2022.5](https://doi.org/10.1017/asb.2022.5)

Data: [Human Mortality Database](https://www.mortality.org/).

_Last reviewed: 2026-11-06_
