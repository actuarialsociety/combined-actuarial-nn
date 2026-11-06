---
marp: true
theme: default
paginate: true
math: katex
title: "NN · Model name"
---

<!-- _class: lead -->
# NN · Model name
### One line: what it does and why an actuary cares

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/NN_name.ipynb) · `kedro run --pipeline <domain> --namespaces <name>`

---

## Theory

$$
\text{one key equation}
$$

- What is new versus the previous lesson's model (one bullet)
- The trick that makes it work (one bullet)

---

## In code

```python
# the 5-10 lines of the model that matter, copied from src/deep_actuarial_science/models/
```

Config: `conf/base/parameters_<domain>.yml` → `<domain>.models.<name>`

---

## Results: reconciled to the benchmark

| Model | Learn | Test | Published test | Status |
| --- | --- | --- | --- | --- |
| baseline | … | … | … | PASS |
| **this model** | … | … | … | PASS |

![w:560](figures/NN-leaderboard.png)

---

## Diagnostics

![w:480](figures/NN-diagnostic-1.png) ![w:480](figures/NN-diagnostic-2.png)

---

## In production

- Calibration / balance property
- Stability across seeds and data scale
- Look-ahead, drift, monitoring

---

## Learn more

1. Surname, A. (Year). Title. *Journal*, Vol(Issue), pages. [doi:10.xxxx/yyyy](https://doi.org/10.xxxx/yyyy)
2. …

_Last reviewed: YYYY-MM-DD_
