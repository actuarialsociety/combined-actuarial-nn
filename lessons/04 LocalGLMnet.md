---
marp: true
theme: default
paginate: true
math: katex
title: "04 · LocalGLMnet"
---

<!-- _class: lead -->
# 04 · LocalGLMnet
### A network that reads like a GLM, with built-in variable selection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/04_localglmnet.ipynb) · `kedro run --pipeline frequency --namespaces localglmnet`

---

## Theory

$$
\log \lambda(\boldsymbol{x}) = \beta_0 + \sum_{j=1}^{q} \beta_j(\boldsymbol{x})\,x_j,
\qquad \big(\beta_1(\boldsymbol{x}),\dots,\beta_q(\boldsymbol{x})\big) = \boldsymbol{z}^{(d:1)}(\boldsymbol{x})
$$

- $\beta_j(\boldsymbol{x})\equiv\beta_j$: a GLM term · $\beta_j\equiv0$: drop $x_j$ · $\beta_j$ varies with $x_k$: interaction [1]
- **Initialise at the GLM:** attention bias = GLM coefficients, weights = 0
- **Significance:** add pure noise `RandN`; its $\beta$ spread sets a 99% band

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

Close to the black-box networks, while every prediction decomposes into $\beta_j(\boldsymbol{x})\,x_j$.

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
