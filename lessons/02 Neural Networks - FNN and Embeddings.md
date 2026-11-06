---
marp: true
theme: default
paginate: true
math: katex
title: "02 · Neural networks and embeddings"
---

<!-- _class: lead -->
# 02 · Neural networks and embeddings
### Let the network learn the features the GLM needed by hand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/02_fnn_embeddings.ipynb) · `kedro run --pipeline frequency --namespaces glm,fnn`

---

## Theory

$$
\lambda^{\mathrm{FNN}}(\boldsymbol{x}) = \exp\big\{w_0 + \langle \boldsymbol{w}, \boldsymbol{z}^{(d:1)}(\boldsymbol{x})\rangle\big\},
\qquad \boldsymbol{z}^{(m)} = \tanh\big(W^{(m)}\boldsymbol{z}^{(m-1)} + \boldsymbol{b}^{(m)}\big)
$$

- **Entity embeddings** [1]: `VehBrand` and `Region` map to learned vectors in $\mathbb{R}^2$
- **Start at the homogeneous model:** $\boldsymbol{w}=0$, $w_0=\log\bar\lambda$
- **Early stopping** on a 10% validation split; keep the best epoch [2]

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

<!--r:fnn.n_params-->792<!--/r--> parameters; fitted in <!--r:fnn.fit_seconds-->93<!--/r--> s on a CPU.

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
