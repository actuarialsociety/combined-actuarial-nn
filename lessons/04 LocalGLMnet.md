---
marp: true
theme: default
paginate: true
math: katex
title: "04 · LocalGLMnet"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
---

<!-- _class: lead -->
# 04 · LocalGLMnet
### A network that reads like a GLM, with built-in variable selection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) · `kedro run --pipeline frequency --namespaces localglmnet`

---

## Theory

$$
\log \lambda(\boldsymbol{x}) = \beta_0 + \sum_{j=1}^{q} \beta_j(\boldsymbol{x})\thinspace x_j,
\qquad \big(\beta_1(\boldsymbol{x}),\dots,\beta_q(\boldsymbol{x})\big) = \boldsymbol{z}^{(d:1)}(\boldsymbol{x})
$$

- $\beta_j(\boldsymbol{x})\equiv\beta_j$: a GLM term · $\beta_j\equiv0$: drop $x_j$ · $\beta_j$ varies with $x_k$: interaction [1]
- **Initialise at the GLM:** attention bias = GLM coefficients, weights = 0
- **Significance:** add pure noise `RandN`; its $\beta$ spread sets a 99% band


---

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① A GLM whose coefficients depend on the policy</b></summary>

$$
\log\lambda(\boldsymbol{x}) = \beta_0 + \sum_{j=1}^{q}\beta_j(\boldsymbol{x})\thinspace x_j,
\qquad \boldsymbol\beta(\boldsymbol{x}) = W\boldsymbol{z}^{(d)}(\boldsymbol{x}) + \boldsymbol{b} \in \mathbb{R}^q
$$

The network (layers [20, 15, 10], tanh) outputs one coefficient per covariate. These **regression attentions** have the same role as GLM coefficients, but each policy gets its own.

</details>

<details>
<summary><b>② How to read the attentions</b></summary>

- $\beta_j(\boldsymbol{x})$ roughly constant: $x_j$ enters **linearly**, as in a GLM
- $\beta_j(\boldsymbol{x}) \approx 0$: $x_j$ has no effect, so it can be **dropped**
- $\beta_j$ changes with $x_j$ itself: a **non-linear** effect
- $\beta_j$ changes with another covariate $x_k$: an **interaction**, visible in $\partial\beta_j/\partial x_k$

Because every covariate is standardised, the sizes of the $\beta_j$ can be compared with each other.

</details>

<details>
<summary><b>③ Every input must be a number</b></summary>

$\beta_j(\boldsymbol{x})\thinspace x_j$ needs a numeric $x_j$, so categories cannot use learned embeddings directly. `VehBrand` and `Region` enter as 2-dimensional pre-computed embeddings, reduced with PCA and standardised (`VehBrandEmb1/2`, `RegionEmb1/2`). The other covariates are processed as in lesson 02.

</details>

<details>
<summary><b>④ Start at the GLM on the same design</b></summary>

A plain Poisson GLM is first fitted on the same 12 standardised inputs. Then:

- attention-layer weights $W = 0$ and bias $\boldsymbol{b}$ = the GLM coefficients
- intercept $\beta_0$ = the GLM intercept

So $\boldsymbol\beta(\boldsymbol{x}) \equiv \boldsymbol{b}$ before training: the network starts as that GLM, and training lets the coefficients vary by policy.

</details>

<details>
<summary><b>⑤ A noise covariate sets the significance band</b></summary>

`RandN` is pure noise, drawn independently of claims. Its attention $\beta_{\text{RandN}}(\boldsymbol{x})$ therefore shows how much a **useless** covariate's coefficient still varies. With $s$ its standard deviation across policies, the shaded band is

$$
\pm 2.576\thinspace s
$$

(a 99% band if the attentions were normally distributed).

A covariate whose $\beta_j(\boldsymbol{x})$ stays inside the band is not distinguishable from noise.

</details>

<details>
<summary><b>⑥ Variable importance</b></summary>

$$
\mathrm{VI}_j = \frac{1}{n}\sum_{i=1}^{n}\big|\beta_j(\boldsymbol{x}_i)\big|
$$

Compare each covariate's importance with that of `RandN`: one that barely clears it adds little and is a candidate to remove before filing rating factors.

</details>

---

## In code

```python
class LocalGLMnetNet(nn.Module):
    def betas(self, x):                         # one coefficient per covariate
        z = x
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return self.attention(z)                # bias initialised to GLM coefficients

    def forward(self, x, offset):
        return torch.exp(self.intercept + (self.betas(x) * x).sum(1)) * offset
```

---

## Results: reconciled to the benchmark

| Model | Learn | Test | Published test | Status |
| --- | --- | --- | --- | --- |
| LocalGLMnet | <!--r:localglmnet.deviance_learn-->44.995<!--/r--> | <!--r:localglmnet.deviance_test-->45.034<!--/r--> | 45.034 | <!--r:localglmnet.status-->PASS<!--/r--> |
| CANN (lesson 03) | <!--r:cann.deviance_learn-->44.989<!--/r--> | <!--r:cann.deviance_test-->44.976<!--/r--> | 44.976 | <!--r:cann.status-->PASS<!--/r--> |

Close to the black-box networks, while every prediction decomposes into $\beta_j(\boldsymbol{x})\thinspace x_j$.

---

## Attention weights: read them like coefficients

![w:900](figures/04-attention.png)

---

## Variable importance

![w:520](figures/04-importance.png)

Every covariate clears the noise covariate `RandN` (red) here; `BonusMalus` and `DrivAge` dominate. A covariate near `RandN` would be a candidate to drop.

---

## In production

- **Explainability:** $\beta_j(\boldsymbol{x})x_j$ gives each policy's premium decomposition
- **Variable selection:** test with a noise covariate before filing rating factors
- **Regularisation:** add an L1 penalty on $\beta_j(\boldsymbol{x})$ to select variables [1]
- **Same checks as any network:** balance property, seed stability, drift

---

## Learn more

1. Richman, R., & Wüthrich, M. V. (2023). LocalGLMnet: interpretable deep learning for tabular data. *Scandinavian Actuarial Journal*, 2023(1), 71–95. [doi:10.1080/03461238.2022.2081816](https://doi.org/10.1080/03461238.2022.2081816)
2. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)
3. Richman, R. (2021). AI in actuarial science – a review of recent advances – part 2. *Annals of Actuarial Science*, 15(2), 230–258. [doi:10.1017/S174849952000024X](https://doi.org/10.1017/S174849952000024X)

_Last reviewed: 2026-11-06_
