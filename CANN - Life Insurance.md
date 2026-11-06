# CANN - Life Insurance
> *In life insurance, the classical model to nest is the Lee–Carter mortality model. This lesson follows the research line from Lee–Carter to neural mortality forecasting, and shows how keeping a classical structure inside the network (the CANN principle) gives forecasts that actuaries can audit [1, 2, 3].*

# Prerequisites
- [CANN - Foundations](./CANN%20-%20Foundations.md)
- Basic mortality modelling: central death rates $m_{x,t}$, exposures, life tables

# Objectives
- Fit the Lee–Carter model by singular value decomposition (SVD) and forecast with a random walk with drift
- Write the Poisson Lee–Carter likelihood and explain why it suits death counts
- Explain how neural networks extend Lee–Carter to many populations (embeddings) and to time-series forecasting (CNN, LSTM)
- Train CNN, LSTM and Transformer mortality forecasters on Swiss data and compare them with Lee–Carter out of sample
- Name other life-insurance uses of neural networks beyond mortality

# Theory

## 1. The classical baseline: Lee–Carter
For age $x$ and calendar year $t$, Lee and Carter [1] model log central death rates as

$$
\log m_{x,t} = a_x + b_x \, k_t + \varepsilon_{x,t}, \qquad \sum_x b_x = 1, \quad \sum_t k_t = 0 .
$$

Here $a_x$ is the average age profile, $k_t$ a period mortality index and $b_x$ the age-specific sensitivity to $k_t$. Estimate $a_x$ as the row mean, and $(b_x, k_t)$ from the first singular vectors of the centred matrix $\log m_{x,t} - a_x$. Forecast $k_t$ as a random walk with drift, $k_{t+1} = k_t + \delta + \eta_{t+1}$.

Brouhns, Denuit and Vermunt [4] replace the Gaussian error with a likelihood for death counts $D_{x,t}$ given exposures $E_{x,t}$:

$$
D_{x,t} \sim \text{Poi}\big(E_{x,t} \, \exp(a_x + b_x k_t)\big).
$$

This is the *Poisson Lee–Carter* model. It has the same Poisson deviance loss as the frequency models in [CANN - Foundations](./CANN%20-%20Foundations.md), so neural extensions carry over directly.

## 2. Neural extensions of Lee–Carter
- **Multi-population via embeddings.** Richman and Wüthrich [2] fit dozens of countries jointly. A network takes learned embeddings for age, country and gender plus calendar year, and outputs $\log m_{x,t}$. The embeddings play the role of $a_x$ and $b_x$. The authors report better out-of-sample forecasts than Lee–Carter models fitted to each population separately.
- **Keeping the Lee–Carter structure.** Scognamiglio [3] keeps the form $a_x + b_x k_t$ for each population, but uses a network to calibrate all populations' $(a_x, b_x, k_t)$ jointly, for both the Gaussian and Poisson variants. This is the CANN principle applied to mortality: the classical model is preserved and interpretable, and the network supplies the flexibility.
- **Time-series forecasting.** Perla, Richman, Scognamiglio and Wüthrich [5] treat the mortality surface as a sequence. Each input is a window of the last $\tau$ years of log rates across all ages, $\big(\log m_{\cdot, t-\tau}, \ldots, \log m_{\cdot, t-1}\big)$, and a convolutional (CNN) or recurrent (LSTM) network predicts $\log m_{\cdot, t}$. Their simple designs keep a Lee–Carter-like low-dimensional structure, and they forecast well across many countries.
- **Uncertainty.** Point forecasts are not enough for reserving or solvency. Schnürch and Korn [6] train a CNN on the age–period mortality surface and use a bootstrap to produce prediction intervals around its forecasts.
- **Earlier work.** Hainaut [7] compresses the log-mortality surface into a few latent factors with a neural generalisation of principal component analysis. He forecasts those factors with an econometric model in place of the $(b_x, k_t)$ structure.

A natural exercise in the spirit of [8] is a *residual CANN* for mortality. Use the Lee–Carter forecast as an offset, and let a network learn a correction initialised at zero:

$$
\log m_{x,t} = \underbrace{\widehat{a}_x + \widehat{b}_x \widehat{k}_t}_{\text{Lee–Carter}} + \text{NN}(x, t, \ldots) .
$$

## 3. Beyond mortality
Neural networks also act as fast approximations (*surrogates*) of expensive life-insurance calculations. Kiermayer and Weiß [9] use neural networks to group term life contracts into model points that preserve the portfolio's risk features, and improve on K-means clustering. The msg life lectures below show networks that learn tariff premium calculations from policy administration systems. In both cases the classical actuarial calculation remains the ground truth that the network must reproduce.

# Content
* [AI Tools for Actuaries, Ch. 8 Mortality forecasting](https://aitools4actuaries.com/) [10]: slides and notes for the follow-along
* [Human Mortality Database](https://www.mortality.org/): the standard source for $D_{x,t}$, $E_{x,t}$ and $m_{x,t}$ (free registration)

### How to apply Deep Neural Networks in Life Insurance - Part 1: Introduction & Overview
[![How to apply Deep Neural Networks in Life Insurance - Part 1: Introduction & Overview](https://img.youtube.com/vi/rrVXnrk7uY0/0.jpg)](https://www.youtube.com/watch?v=rrVXnrk7uY0)
> *msg.Insurance Suite · an overview of deep neural networks in core life-insurance systems, including learning the logic of legacy systems during migration*

---
### Actuarial calculations based on Deep Neural Networks (DNN) – Part 1: Representation Learning
[![Actuarial calculations based on Deep Neural Networks (DNN) – Part 1: Representation Learning](https://img.youtube.com/vi/9AznSYCfde8/0.jpg)](https://www.youtube.com/watch?v=9AznSYCfde8)
> *msg.Insurance Suite · the theory behind using DNNs as surrogates for actuarial calculations (Section 3); Part 2 is the hands-on build*

---
### Actuarial calculations based on Deep Neural Networks (DNN) – Part 2: A Tarification Model
[![Actuarial calculations based on Deep Neural Networks (DNN) – Part 2: A Tarification Model](https://img.youtube.com/vi/WUjmIeFow5g/0.jpg)](https://www.youtube.com/watch?v=WUjmIeFow5g)
> *msg.Insurance Suite · a hands-on build: load and prepare data, train a DNN to approximate tariff premiums, evaluate it*

---

# Follow-along
**Repo:** [wueth/AITools4Actuaries](https://github.com/wueth/AITools4Actuaries) · [Open the mortality notebook in Colab](https://colab.research.google.com/github/wueth/AITools4Actuaries/blob/main/Ch8_11_Mortality.ipynb)

Use the folder layout from [CANN - Foundations](./CANN%20-%20Foundations.md). `Data.zip` includes `CHE.Mx_1x1.txt` (Swiss mortality rates from the Human Mortality Database).

1. Open `Ch8_11_Mortality.ipynb` and run **Exploratory analysis & learn-test split**. Males, learning data up to `obs_year = 1999`, testing on later years.
2. Run **Lee-Carter model**. It estimates $a_x$ as the age mean and $(b_x, k_t)$ by SVD, as in Section 1. Plot $k_t$ and note the drift.
3. Run **Data preprocessing for neural networks**. With `lookback = 10`, each sample is a 10-year window of log rates across all ages: the design of [5].
4. Run **Train models**, then compare the RMSE table for Lee–Carter, CNN, CNN with tanh, LSTM and Transformer on the test years.
5. Now try it yourself: build the residual CANN from Section 2. Add the Lee–Carter forecast to the network output, initialise the last layer to zero (as in `Ch5_07_CANN_PyTorch.ipynb`), and see whether the test RMSE improves on both models.
6. Optional (R): [schnuerch/nn_mortality](https://github.com/schnuerch/nn_mortality) is a Jupyter introduction to neural mortality modelling by the first author of [6].

# References
1. Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659–671. [doi:10.1080/01621459.1992.10475265](https://doi.org/10.1080/01621459.1992.10475265)
2. Richman, R., & Wüthrich, M. V. (2021). A neural network extension of the Lee–Carter model to multiple populations. *Annals of Actuarial Science*, 15(2), 346–366. [doi:10.1017/S1748499519000071](https://doi.org/10.1017/S1748499519000071)
3. Scognamiglio, S. (2022). Calibrating the Lee–Carter and the Poisson Lee–Carter models via neural networks. *ASTIN Bulletin*, 52(2), 519–561. [doi:10.1017/asb.2022.5](https://doi.org/10.1017/asb.2022.5) · [arXiv:2106.12312](https://arxiv.org/abs/2106.12312)
4. Brouhns, N., Denuit, M., & Vermunt, J. K. (2002). A Poisson log-bilinear regression approach to the construction of projected lifetables. *Insurance: Mathematics and Economics*, 31(3), 373–393. [doi:10.1016/S0167-6687(02)00185-3](https://doi.org/10.1016/S0167-6687%2802%2900185-3)
5. Perla, F., Richman, R., Scognamiglio, S., & Wüthrich, M. V. (2021). Time-series forecasting of mortality rates using deep learning. *Scandinavian Actuarial Journal*, 2021(7), 572–598. [doi:10.1080/03461238.2020.1867232](https://doi.org/10.1080/03461238.2020.1867232)
6. Schnürch, S., & Korn, R. (2022). Point and interval forecasts of death rates using neural networks. *ASTIN Bulletin*, 52(1), 333–360. [doi:10.1017/asb.2021.34](https://doi.org/10.1017/asb.2021.34)
7. Hainaut, D. (2018). A neural-network analyzer for mortality forecast. *ASTIN Bulletin*, 48(2), 481–508. [doi:10.1017/asb.2017.45](https://doi.org/10.1017/asb.2017.45)
8. Wüthrich, M. V., & Merz, M. (2019). Editorial: Yes, we CANN! *ASTIN Bulletin*, 49(1), 1–3. [doi:10.1017/asb.2018.42](https://doi.org/10.1017/asb.2018.42)
9. Kiermayer, M., & Weiß, C. (2021). Grouping of contracts in insurance using neural networks. *Scandinavian Actuarial Journal*, 2021(4), 295–322. [doi:10.1080/03461238.2020.1836676](https://doi.org/10.1080/03461238.2020.1836676)
10. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)

---
_Last reviewed: 2026-11-06_
