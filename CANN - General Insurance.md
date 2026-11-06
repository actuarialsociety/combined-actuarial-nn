# CANN - General Insurance
> *How the CANN idea carries over to the main non-life problems: claims frequency pricing with telematics, interpretable networks for rating factors, and claims reserving, where the chain-ladder model becomes the nested baseline [1, 2, 3].*

# Prerequisites
- [CANN - Foundations](./CANN%20-%20Foundations.md)
- Familiarity with the chain-ladder method

# Objectives
- Extend the CANN frequency model with telematics covariates, and explain the choice between cross-sectional and longitudinal designs
- Fit a LocalGLMnet and read its attention weights $\beta_j(\boldsymbol{x})$ as covariate-dependent GLM coefficients
- Use a pure-noise covariate to test variable significance in a LocalGLMnet
- Write the over-dispersed Poisson (ODP) chain-ladder model as a GLM, and embed it in a neural network for multiple lines of business
- Compare nested reserving (CANN) with fully neural reserving (DeepTriangle)

# Theory

## 1. Pricing: CANN with richer covariates
Pricing follows [CANN - Foundations](./CANN%20-%20Foundations.md): a Poisson GLM gives the baseline, and the network adds a multiplicative correction. The network term is most valuable when the data has covariates that a GLM cannot use well. Telematics driving data is the leading example. Duval, Boucher and Pigeon [1] build *telematics CANNs*: a classical GLM uses the traditional rating factors, and the network term uses the telematics summaries. They apply this to both cross-sectional claim counts and longitudinal panel claim counts. Gao, Wang and Wüthrich [4] reach a similar conclusion by boosting a Poisson GLM with telematics features. In both papers the classical model remains the baseline, and the telematics signal is learned as a correction to it.

Severity uses the same construction. Replace the Poisson deviance with the gamma deviance

$$
L = \frac{2}{n}\sum_{i=1}^{n} w_i \left[ \frac{Y_i - \mu(\boldsymbol{x}_i)}{\mu(\boldsymbol{x}_i)} - \log\frac{Y_i}{\mu(\boldsymbol{x}_i)} \right],
$$

and use the gamma GLM log-mean as the offset [5, Ch. 7]. Check the balance property after fitting, as in Foundations Section 4 [6].

## 2. Interpretability: LocalGLMnet
Regulators and pricing committees need to read rating factors. LocalGLMnet [2] keeps the GLM's additive structure but lets each coefficient vary with the covariates:

$$
\log \lambda(\boldsymbol{x}) = \beta_0 + \sum_{j=1}^{q} \beta_j(\boldsymbol{x}) \, x_j, \qquad \big(\beta_1(\boldsymbol{x}), \ldots, \beta_q(\boldsymbol{x})\big)^\top = \boldsymbol{z}^{(d:1)}(\boldsymbol{x}) \in \mathbb{R}^q .
$$

The network outputs one *attention weight* $\beta_j(\boldsymbol{x})$ per covariate. Read them as follows:

- $\beta_j(\boldsymbol{x}) \equiv \beta_j$ constant: covariate $j$ enters as in a GLM.
- $\beta_j(\boldsymbol{x}) \equiv 0$: covariate $j$ can be dropped.
- $\beta_j(\boldsymbol{x})$ depends on $x_j$ only: a non-linear main effect.
- $\beta_j(\boldsymbol{x})$ depends on $x_k$: an interaction. The gradient $\partial \beta_j / \partial x_k$ shows which one.

For variable selection, add a covariate drawn as independent noise. Its fitted weights show how far a weight can spread by chance alone, which gives a confidence band: real covariates whose weights stay inside it are not significant [2]. LocalGLMnet can also be initialised at a fitted GLM, the same starting point that CANN uses.

## 3. Reserving: embedding the chain-ladder model
Let $X_{i,j}$ be incremental payments for accident year $i$ and development year $j$. The ODP model of Renshaw and Verrall [7] reproduces the chain-ladder reserves:

$$
\mathbb{E}[X_{i,j}] = \mu_{i,j} = \exp(\alpha_i + \beta_j), \qquad \text{Var}(X_{i,j}) = \varphi \, \mu_{i,j}.
$$

This is a GLM with log link, so it can be nested inside a network. Gabrielli, Richman and Wüthrich [3] fit the ODP model to several lines of business together. They use it as the network's initial state, and add a network term with embeddings for line of business, accident year and development year:

$$
\log \mu_{i,j}^{(\ell)} = \underbrace{\alpha_i^{(\ell)} + \beta_j^{(\ell)}}_{\text{ODP (chain-ladder)}} + \underbrace{\text{NN}\big(e_\ell, e_i, e_j\big)}_{\text{initialised at } 0} .
$$

The network learns how claims development differs across lines and years, while the chain-ladder result stays as the reference. Gabrielli [8] extends this to a *double* ODP model that also uses claim counts, which helps reserving when payment data is volatile. To test such methods on individual claims, the simulation machine of Gabrielli and Wüthrich [9] generates realistic synthetic claim histories.

**Fully neural alternative.** DeepTriangle [10] drops the nested baseline. It feeds payment and claim-count histories into a recurrent network trained across many companies. It is a useful benchmark, but it loses the chain-ladder anchor that makes CANN results easy to explain to a reserving committee. Richman [11, 12] compares these and other reserving approaches.

# Content
* [AI Tools for Actuaries, Ch. 5 LocalGLMnet](https://aitools4actuaries.com/) [13]: slides and notes for Section 2
* [Telematics CANNs](https://doi.org/10.1017/asb.2024.4) [1]: the most recent CANN pricing paper

### Arthur Charpentier: Machine Learning in Actuarial Science & Insurance
[![Arthur Charpentier: Machine Learning in Actuarial Science & Insurance](https://img.youtube.com/vi/__fZSBREiLk/0.jpg)](https://www.youtube.com/watch?v=__fZSBREiLk)
> *ML portal · a research overview of machine learning in actuarial science and insurance, by a leading actuarial-statistics researcher*

---
### Webinar: Machine Learning in Reserving on 15 July 2025
[![Webinar: Machine Learning in Reserving on 15 July 2025](https://img.youtube.com/vi/rL24D8Z_GlE/0.jpg)](https://www.youtube.com/watch?v=rL24D8Z_GlE)
> *International Actuarial Association · the IFoA Machine Learning in Reserving working party on recent developments, the context for Section 3*

---
### (Second session): Machine learning in general insurance reserving–method comparison & interpretation
[![(Second session): Machine learning in general insurance reserving–method comparison & interpretation](https://img.youtube.com/vi/-Da2IkkySQI/0.jpg)](https://www.youtube.com/watch?v=-Da2IkkySQI)
> *Society of Actuaries in Ireland · compares chain-ladder, GLM and neural reserving methods and how to interpret them*

---

# Follow-along
**Repo:** [wueth/AITools4Actuaries](https://github.com/wueth/AITools4Actuaries) · [Open the LocalGLMnet notebook in Colab](https://colab.research.google.com/github/wueth/AITools4Actuaries/blob/main/Ch5_08_LocalGLMnet_PyTorch.ipynb)

Use the folder layout from [CANN - Foundations](./CANN%20-%20Foundations.md) (`aitools/Data/` next to `aitools/work/AITools4Actuaries/`).

1. Open `Ch5_08_LocalGLMnet_PyTorch.ipynb`. It loads `freMTPL2freqEmb.parquet`, which contains `RandN`, a pure-noise covariate used for significance testing.
2. Run **GLM baseline analysis** and **Initialize the LocalGLMnet to the previously fitted GLM**. As with CANN, the network starts at the GLM deviance.
3. Run **Train the LocalGLMnet model** and **LocalGLMnet results**, then compare out-of-sample Poisson deviance with the GLM.
4. Run **Boxplot of the attention weights**. The spread of the noise covariate's weights is the significance band from Section 2. List the covariates whose weights lie inside it.
5. Run **Plot individual attention weights** (`DrivAge`, `BonusMalus`, `RandN`) and **Gradient of interactions**. Name the covariate that `DrivAge` interacts with most strongly.
6. Optional (R): for Section 3, clone [kasaai/deeptriangle](https://github.com/kasaai/deeptriangle) and follow its README to reproduce the DeepTriangle benchmark [10].

# References
1. Duval, F., Boucher, J.-P., & Pigeon, M. (2024). Telematics combined actuarial neural networks for cross-sectional and longitudinal claim count data. *ASTIN Bulletin*, 54(2), 239–262. [doi:10.1017/asb.2024.4](https://doi.org/10.1017/asb.2024.4)
2. Richman, R., & Wüthrich, M. V. (2023). LocalGLMnet: interpretable deep learning for tabular data. *Scandinavian Actuarial Journal*, 2023(1), 71–95. [doi:10.1080/03461238.2022.2081816](https://doi.org/10.1080/03461238.2022.2081816) · [arXiv:2107.11059](https://arxiv.org/abs/2107.11059)
3. Gabrielli, A., Richman, R., & Wüthrich, M. V. (2020). Neural network embedding of the over-dispersed Poisson reserving model. *Scandinavian Actuarial Journal*, 2020(1), 1–29. [doi:10.1080/03461238.2019.1633394](https://doi.org/10.1080/03461238.2019.1633394)
4. Gao, G., Wang, H., & Wüthrich, M. V. (2022). Boosting Poisson regression models with telematics car driving data. *Machine Learning*, 111(1), 243–272. [doi:10.1007/s10994-021-05957-0](https://doi.org/10.1007/s10994-021-05957-0)
5. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer Actuarial. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9) (open access)
6. Wüthrich, M. V. (2020). Bias regularization in neural network models for general insurance pricing. *European Actuarial Journal*, 10(1), 179–202. [doi:10.1007/s13385-019-00215-z](https://doi.org/10.1007/s13385-019-00215-z)
7. Renshaw, A. E., & Verrall, R. J. (1998). A stochastic model underlying the chain-ladder technique. *British Actuarial Journal*, 4(4), 903–923. [doi:10.1017/S1357321700000222](https://doi.org/10.1017/S1357321700000222)
8. Gabrielli, A. (2020). A neural network boosted double overdispersed Poisson claims reserving model. *ASTIN Bulletin*, 50(1), 25–60. [doi:10.1017/asb.2019.33](https://doi.org/10.1017/asb.2019.33)
9. Gabrielli, A., & Wüthrich, M. V. (2018). An individual claims history simulation machine. *Risks*, 6(2), 29. [doi:10.3390/risks6020029](https://doi.org/10.3390/risks6020029)
10. Kuo, K. (2019). DeepTriangle: a deep learning approach to loss reserving. *Risks*, 7(3), 97. [doi:10.3390/risks7030097](https://doi.org/10.3390/risks7030097)
11. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 1. *Annals of Actuarial Science*, 15(2), 207–229. [doi:10.1017/S1748499520000238](https://doi.org/10.1017/S1748499520000238)
12. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 2. *Annals of Actuarial Science*, 15(2), 230–258. [doi:10.1017/S174849952000024X](https://doi.org/10.1017/S174849952000024X)
13. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)

---
_Last reviewed: 2026-11-06_
