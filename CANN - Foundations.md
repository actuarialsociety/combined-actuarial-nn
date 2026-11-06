# CANN - Foundations
> *Combined Actuarial Neural Networks (CANNs) nest a classical actuarial model, usually a GLM, inside a neural network. The network starts from the GLM and only learns what the GLM misses, so the actuary keeps a transparent baseline while gaining predictive power [1, 2].*

# Prerequisites
- [Python - Basic](https://github.com/actuarialsociety/ds-training/blob/main/Python%20-%20Basic.md) and [PyData Stack](https://github.com/actuarialsociety/ds-training/blob/main/PyData%20Stack.md) from ds-training
- Working knowledge of GLMs and maximum likelihood estimation [3]

# Objectives
- State the Poisson GLM for claim frequency and derive its deviance loss
- Write a feed-forward neural network (FNN) as a regression function, including entity embeddings for categorical covariates
- Explain the CANN construction, and why initialising the network output weights to zero reproduces the GLM exactly
- Fit a GLM and a CANN to the freMTPL2 motor data in PyTorch and compare out-of-sample Poisson deviance
- Explain the balance property, and why early-stopped networks can violate it

# Theory

## 1. The Poisson GLM baseline
Let $N_i$ be the claim count of policy $i$, with exposure $v_i > 0$ and covariates $\boldsymbol{x}_i$. The classical frequency model [3, 4] is

$$
N_i \sim \text{Poi}\big(v_i \, \lambda(\boldsymbol{x}_i)\big), \qquad \log \lambda^{\text{GLM}}(\boldsymbol{x}) = \beta_0 + \langle \boldsymbol{\beta}, \boldsymbol{x} \rangle .
$$

Parameters are estimated by minimising the Poisson deviance loss, which is equivalent to maximum likelihood:

$$
L(\boldsymbol{\beta}) = \frac{2}{n} \sum_{i=1}^{n} \left[ v_i \lambda(\boldsymbol{x}_i) - N_i - N_i \log \frac{v_i \lambda(\boldsymbol{x}_i)}{N_i} \right],
$$

with the convention $0 \log 0 = 0$. GLM covariates need manual feature engineering (binning, log-transforms, chosen interactions). This is exactly where a GLM loses predictive power [5].

## 2. Feed-forward neural networks
An FNN of depth $d$ composes hidden layers $\boldsymbol{z}^{(m)}(\boldsymbol{x}) = \phi\big(W^{(m)} \boldsymbol{x} + \boldsymbol{b}^{(m)}\big)$, where $\phi$ is an activation such as $\tanh$. Write the composition as $\boldsymbol{z}^{(d:1)} = \boldsymbol{z}^{(d)} \circ \cdots \circ \boldsymbol{z}^{(1)}$. The FNN frequency model is

$$
\lambda^{\text{FNN}}(\boldsymbol{x}) = \exp\left\{ w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x}) \rangle \right\}.
$$

The network learns its own features and interactions. Categorical covariates with many levels (vehicle brand, region) are mapped to low-dimensional learned vectors, called *entity embeddings*, instead of being one-hot encoded [4, 6]. Training uses stochastic gradient descent on the same Poisson deviance, with early stopping on a validation set to prevent overfitting.

## 3. The CANN construction
Wüthrich and Merz [1] proposed combining the two regression functions additively on the log scale:

$$
\lambda^{\text{CANN}}(\boldsymbol{x}) = \exp\Big\{ \underbrace{\beta_0 + \langle \boldsymbol{\beta}, \boldsymbol{x} \rangle}_{\text{GLM}} + \underbrace{w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x}) \rangle}_{\text{neural network}} \Big\}.
$$

The network enters as a *skip connection*. In the common implementation the GLM is fitted first and its prediction becomes an offset, so the network models a multiplicative correction:

$$
v_i \, \lambda^{\text{CANN}}(\boldsymbol{x}_i) = \underbrace{v_i \, \widehat{\lambda}^{\text{GLM}}(\boldsymbol{x}_i)}_{\text{offset}} \cdot \exp\left\{ w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x}_i) \rangle \right\}.
$$

Initialising $w_0 = 0$ and $\boldsymbol{w} = \boldsymbol{0}$ makes the CANN start *exactly* at the GLM maximum likelihood estimate. Gradient descent then only moves away from the GLM where the data supports it [1, 2]. Three consequences:

- **Interpretability:** the GLM term remains a readable baseline, and the network term measures how much (and where) the GLM is wrong.
- **Model diagnostics:** if training barely moves the network from zero, the GLM is adequately specified.
- **Flexibility:** the GLM parameters can be frozen (offset) or trained jointly with the network. Schelldorfer and Wüthrich [2] give both variants, applied to the French motor third-party liability (freMTPL2) data of [5].

## 4. Calibration: the balance property
A GLM with canonical link and an intercept satisfies the *balance property* on the learning data:

$$
\sum_{i=1}^{n} v_i \, \widehat{\lambda}(\boldsymbol{x}_i) = \sum_{i=1}^{n} N_i ,
$$

so the portfolio premium matches observed claims. Networks fitted by early-stopped gradient descent generally do **not** satisfy this, and the bias can be material for pricing. Wüthrich [7, 8] proposes a correction: refit the output layer as a GLM on the learned features $\boldsymbol{z}^{(d:1)}(\boldsymbol{x}_i)$, which restores balance. Because individual network fits vary with the random seed, averaging several runs (the *nagging predictor* [9]) is recommended before drawing conclusions.

## 5. Where to go next
CANN is the template for the other lessons in this repo. [CANN - General Insurance](./CANN%20-%20General%20Insurance.md) applies it to pricing, interpretability and claims reserving. [CANN - Life Insurance](./CANN%20-%20Life%20Insurance.md) applies the same idea to mortality models. Richman [10, 11] surveys the wider literature, and Wüthrich and Merz [4] is the open-access textbook reference.

# Content
* [AI Tools for Actuaries](https://aitools4actuaries.com/): lecture notes, slides and notebooks by Wüthrich, Richman, Avanzi, Lindholm, Maggi, Mayer, Schelldorfer and Scognamiglio [12]. Read Chapters 3 and 5 (GLMs, FNNs, CANN)
* [Statistical Foundations of Actuarial Learning and its Applications](https://link.springer.com/book/10.1007/978-3-031-12409-9) [4]: open access. Chapter 7 covers deep learning and the CANN approach

### Combined Actuarial Neural Networks - Part 1: Paper Review
[![Combined Actuarial Neural Networks - Part 1: Paper Review](https://img.youtube.com/vi/G6TWGUrd3cs/0.jpg)](https://www.youtube.com/watch?v=G6TWGUrd3cs)
> *The Practical Actuary · a 22-minute walk-through of the "Yes, we CANN!" editorial [1]. Read the paper alongside it*

---
### Poisson regression - How to calculate likelihood and deviance
[![Poisson regression - How to calculate likelihood and deviance](https://img.youtube.com/vi/5GJR81WgLN0/0.jpg)](https://www.youtube.com/watch?v=5GJR81WgLN0)
> *TileStats · the Poisson likelihood and deviance from Section 1, worked by hand*

---
### Forwardpass (Neural Networks) - Part 1 of Stefan Noertemann's lecture
[![Forwardpass (Neural Networks) - Part 1 of Stefan Noertemann's lecture](https://img.youtube.com/vi/87ct8tgd5y4/0.jpg)](https://www.youtube.com/watch?v=87ct8tgd5y4)
> *msg.Insurance Suite · how an FNN computes $\boldsymbol{z}^{(d:1)}(\boldsymbol{x})$, taught for an actuarial audience*

---
### Backpropagation - Basics of artificial neural networks - Part 2 of Stefan Noertemann's lecture
[![Backpropagation - Basics of artificial neural networks - Part 2 of Stefan Noertemann's lecture](https://img.youtube.com/vi/Qt3BiIlik-A/0.jpg)](https://www.youtube.com/watch?v=Qt3BiIlik-A)
> *msg.Insurance Suite · how gradient descent trains the network weights $\boldsymbol{w}$ and $W^{(m)}$*

---

# Follow-along
**Repo:** [wueth/AITools4Actuaries](https://github.com/wueth/AITools4Actuaries) · [Open the CANN notebook in Colab](https://colab.research.google.com/github/wueth/AITools4Actuaries/blob/main/Ch5_07_CANN_PyTorch.ipynb)

The notebooks read data from `../../Data/`, so set up this folder layout:

```bash
mkdir -p aitools/work && cd aitools
curl -LO https://people.math.ethz.ch/~wueth/AITools/Data.zip && unzip Data.zip   # creates aitools/Data/
cd work && git clone https://github.com/wueth/AITools4Actuaries
cd AITools4Actuaries && pip install pandas pyarrow statsmodels scikit-learn torch matplotlib
```

1. Run `Ch3_02_Poisson_GLM.ipynb`. It fits two Poisson GLMs on `freMTPL2freq.parquet` with `offset=np.log(learn["Exposure"])` and compares in-sample and out-of-sample deviance (Section 1). Note the out-of-sample deviance of GLM 1.
2. Open `Ch5_07_CANN_PyTorch.ipynb`. In `class FNN`, find `init.constant_(self.output_layer.weight, 0.0)` and `return torch.exp(self.output_layer(x)).flatten() * v`. This is the CANN equation from Section 3, with the GLM prediction `v` as the offset.
3. Run the cell **Replicate GLM results within the FNN architecture**. Before training, the CANN deviance equals the GLM deviance, which confirms the zero initialisation.
4. Run **Train the CANN model** and **CANN results**. Compare GLM and CANN out-of-sample Poisson deviance, then look at the driver-age plot to see where the network corrects the GLM.
5. Now try it yourself: check the balance property (Section 4). Compare `learn["ClaimNb"].sum()` with the sum of CANN predictions on `learn`. Then refit with a different `SEED` and average the predictions of a few runs, as in [9].

# References
1. Wüthrich, M. V., & Merz, M. (2019). Editorial: Yes, we CANN! *ASTIN Bulletin*, 49(1), 1–3. [doi:10.1017/asb.2018.42](https://doi.org/10.1017/asb.2018.42)
2. Schelldorfer, J., & Wüthrich, M. V. (2019). Nesting classical actuarial models into neural networks. *SSRN 3320525*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3320525) · [code](https://github.com/actuarial-data-science/Tutorials/tree/master/3%20-%20Nesting%20Classical%20Actuarial%20Models%20into%20Neural%20Networks)
3. Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models. *Journal of the Royal Statistical Society, Series A*, 135(3), 370–384. [doi:10.2307/2344614](https://doi.org/10.2307/2344614)
4. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer Actuarial. [doi:10.1007/978-3-031-12409-9](https://doi.org/10.1007/978-3-031-12409-9) (open access)
5. Noll, A., Salzmann, R., & Wüthrich, M. V. (2018). Case study: French motor third-party liability claims. *SSRN 3164764*. [doi:10.2139/ssrn.3164764](https://doi.org/10.2139/ssrn.3164764)
6. Richman, R., & Wüthrich, M. V. (2021). A neural network extension of the Lee–Carter model to multiple populations. *Annals of Actuarial Science*, 15(2), 346–366. [doi:10.1017/S1748499519000071](https://doi.org/10.1017/S1748499519000071)
7. Wüthrich, M. V. (2020). Bias regularization in neural network models for general insurance pricing. *European Actuarial Journal*, 10(1), 179–202. [doi:10.1007/s13385-019-00215-z](https://doi.org/10.1007/s13385-019-00215-z)
8. Wüthrich, M. V. (2022). The balance property in neural network modelling. *Statistical Theory and Related Fields*, 6(1), 1–9. [doi:10.1080/24754269.2021.1877960](https://doi.org/10.1080/24754269.2021.1877960)
9. Richman, R., & Wüthrich, M. V. (2020). Nagging predictors. *Risks*, 8(3), 83. [doi:10.3390/risks8030083](https://doi.org/10.3390/risks8030083)
10. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 1. *Annals of Actuarial Science*, 15(2), 207–229. [doi:10.1017/S1748499520000238](https://doi.org/10.1017/S1748499520000238)
11. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 2. *Annals of Actuarial Science*, 15(2), 230–258. [doi:10.1017/S174849952000024X](https://doi.org/10.1017/S174849952000024X)
12. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)

---
_Last reviewed: 2026-11-06_
