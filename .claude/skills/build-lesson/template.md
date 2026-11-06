---
marp: true
theme: default
paginate: true
math: katex
title: "NN · Model name"
style: |
  details { border: 1px solid rgba(140,149,159,.45); border-radius: 8px; padding: .25em .8em; margin: .35em 0; font-size: .72em; }
  summary { cursor: pointer; font-size: 1.15em; }
  details[open] { background: rgba(140,149,159,.08); }
  section:has(details[open]) details:not([open]) { display: none; }
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

## Core ideas

_Click a panel to open it; work through them in order._

<details>
<summary><b>① The problem the model solves</b></summary>

The setting, notation and the equation that defines the model.

</details>

<details>
<summary><b>② The key mechanism</b></summary>

What is new versus the previous lesson, with its equation.

</details>

<details>
<summary><b>③ How it is fitted here</b></summary>

Loss, optimiser, initialisation, early stopping: the exact settings in `conf/base/parameters_<domain>.yml`.

</details>

<details>
<summary><b>④ How to read the output</b></summary>

What each diagnostic shows; numbers via `<!--r:model.metric-->` markers only.

</details>

<details>
<summary><b>⑤ What can go wrong</b></summary>

Balance, look-ahead, seed stability, extrapolation.

</details>

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
