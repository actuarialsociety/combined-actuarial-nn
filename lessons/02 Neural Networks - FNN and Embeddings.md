---
marp: true
theme: default
paginate: true
math: katex
title: "02 · Neural networks and embeddings"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
  section:has(details[open]) details:not([open]) { display: none; }
---

<!-- _class: lead -->
# 02 · Neural networks and embeddings
### Let the network learn the features the GLM needed by hand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) · `kedro run --pipeline frequency --namespaces glm,fnn`

---

## Theory

```math
\lambda^{\mathrm{FNN}}(\boldsymbol{x}) = \exp\big\lbrace w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x})\rangle\big\rbrace,
\qquad \boldsymbol{z}^{(m)} = \tanh\big(W^{(m)}\boldsymbol{z}^{(m-1)} + \boldsymbol{b}^{(m)}\big)
```

- **Entity embeddings** [1]: `VehBrand` and `Region` map to learned vectors in $\mathbb{R}^2$
- **Start at the homogeneous model:** $\boldsymbol{w}=0$, $w_0=\log\bar\lambda$
- **Early stopping** on a 10% validation split; keep the best epoch [2]


---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① A network is a stack of learned features</b></summary>

$\displaystyle \boldsymbol{z}^{(m)} = \tanh\big(W^{(m)}\boldsymbol{z}^{(m-1)} + \boldsymbol{b}^{(m)}\big),\quad m = 1,\dots,d, \qquad \mu(\boldsymbol{x}) = v\exp\big(w_0 + \langle\boldsymbol{w}, \boldsymbol{z}^{(d)}\rangle\big)$

- Here $d = 3$ hidden layers of 20, 15 and 10 units (<!--r:fnn.n_params-->792<!--/r--> parameters in total)
- The last layer is a Poisson GLM on the learned features $\boldsymbol{z}^{(d)}$, so the network is a GLM whose features are learned rather than hand-made
- Non-linearities and interactions come from composing the layers

</details>

<details>
<summary><b>② Preparing the inputs</b></summary>

`tanh` saturates for large inputs, so the scale of each input matters:

- `VehAge`, `DrivAge`, `BonusMalus`, `VehPower` are capped at 20, 90, 150 and 15, then standardised
- `Area` becomes an ordinal code and `Density` is logged; both are standardised
- `VehGas` becomes a 0/1 diesel indicator
- `VehBrand` and `Region` stay as integer codes for the embedding layers

</details>

<details>
<summary><b>③ Entity embeddings for categories</b></summary>

A categorical covariate with $K$ levels is mapped to a learned vector:

$\displaystyle e: \lbrace 1,\dots,K\rbrace \to \mathbb{R}^{b}, \qquad b = 2 .$

- One-hot encoding needs $K-1$ inputs per covariate; an embedding needs $b$
- The vectors are trained together with the rest of the network, so levels with similar claim behaviour end up close together, which you can plot and interpret
- The embeddings are concatenated with the continuous inputs before the first hidden layer

</details>

<details>
<summary><b>④ Training: Poisson loss, NAdam, mini-batches</b></summary>

$\displaystyle L(\theta) = \sum_i \big(\mu_i(\theta) - N_i\log\mu_i(\theta)\big)$

- This is the negative Poisson log-likelihood (`PoissonNLLLoss`), the same objective as the GLM
- Optimiser NAdam; mini-batches of 5,000 policies; up to 200 epochs; fixed seed
- The output layer starts at $\boldsymbol{w} = \boldsymbol{0}$, $w_0 = \log(\sum N_i / \sum v_i)$, so training starts **at the homogeneous model** and every step has to improve on it

</details>

<details>
<summary><b>⑤ Early stopping as regularisation</b></summary>

- 10% of the learning data is held out for validation
- After each epoch, the validation loss is computed; the weights of the **best** epoch are kept
- Without this, a network keeps fitting noise in the learning data and gets worse out of sample
- Read the training plot: the gap between training and validation loss shows how much of the fit generalises

</details>

<details>
<summary><b>⑥ Why the balance property breaks, and the fix</b></summary>

Gradient descent stopped early has not solved $\sum_i(N_i - \hat\mu_i) = 0$, so the total prediction drifts from the observed total (here <!--r:fnn.balance_learn-->1.0031<!--/r-->×). Because the output is $\exp(w_0 + \cdots)$, shifting the bias rescales every prediction by the same factor:

$\displaystyle w_0 \leftarrow w_0 + \log\frac{\sum_i N_i}{\sum_i \hat\mu_i} .$

`balance_correction: true` applies this after training. The relative ranking of policies is unchanged.

</details>

---

## In code

```python
class EmbeddingFNN(nn.Module):
    def forward(self, x, offset):                       # offset = exposure
        cats = [e(x[:, -2 + i].long()) for i, e in enumerate(self.emb)]
        z = torch.cat([x[:, :-2], *cats], dim=1)
        for layer in self.hidden:                       # [20, 15, 10], tanh
            z = torch.tanh(layer(z))
        return torch.exp(self.out(z)).flatten() * offset
```

Loss: Poisson NLL; optimiser: NAdam; batch 5,000; 200 epochs.

---

## Results: reconciled to the benchmark

| Model | Learn | Test | Published test | Balance (learn) | Status |
| --- | --- | --- | --- | --- | --- |
| GLM | <!--r:glm.deviance_learn-->46.954<!--/r--> | <!--r:glm.deviance_test-->47.179<!--/r--> | 47.179 | <!--r:glm.balance_learn-->1.0000<!--/r--> | <!--r:glm.status-->PASS<!--/r--> |
| FNN | <!--r:fnn.deviance_learn-->44.973<!--/r--> | <!--r:fnn.deviance_test-->44.928<!--/r--> | 44.928 | <!--r:fnn.balance_learn-->1.0031<!--/r--> | <!--r:fnn.status-->PASS<!--/r--> |

<!--r:fnn.n_params-->792<!--/r--> parameters; fitted in <!--r:fnn.fit_seconds-->62<!--/r--> s on a CPU.

---

## Diagnostics

![w:470](figures/02-fnn-training.png) ![w:470](figures/02-fnn-bonusmalus.png)

---

## In production

- **Balance property:** early stopping breaks $\sum_i\hat\mu_i=\sum_i N_i$. Correct the
  output bias (`balance_correction: true`) or refit the last layer as a GLM [2, 3]
- **Seed risk:** networks fitted with different seeds price individual policies differently;
  average several fits (the *nagging predictor* [4]) before pricing
- **Explainability:** embeddings and marginal plots, or move to LocalGLMnet (lesson 04)

---

## Learn more

1. Guo, C., & Berkhahn, F. (2016). Entity embeddings of categorical variables. [arXiv:1604.06737](https://arxiv.org/abs/1604.06737)
2. Wüthrich, M. V. (2020). Bias regularization in neural network models for general insurance pricing. *European Actuarial Journal*, 10(1), 179–202. [doi:10.1007/s13385-019-00215-z](https://doi.org/10.1007/s13385-019-00215-z)
3. Wüthrich, M. V. (2022). The balance property in neural network modelling. *Statistical Theory and Related Fields*, 6(1), 1–9. [doi:10.1080/24754269.2021.1877960](https://doi.org/10.1080/24754269.2021.1877960)
4. Richman, R., & Wüthrich, M. V. (2020). Nagging predictors. *Risks*, 8(3), 83. [doi:10.3390/risks8030083](https://doi.org/10.3390/risks8030083)
5. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 1. *Annals of Actuarial Science*, 15(2), 207–229. [doi:10.1017/S1748499520000238](https://doi.org/10.1017/S1748499520000238)

_Last reviewed: 2026-11-06_
