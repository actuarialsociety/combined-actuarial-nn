# `pipelines/frequency/`

Claim frequency on freMTPL2 with the fixed learn/test split of Wüthrich & Merz (2023).

```mermaid
flowchart TD
    raw[freMTPL2] --> prep["prepare<br/>split + noise controls"]
    prep --> fit["fit_model"] --> pred["predict<br/>per policy, per seed"]
    pred --> ev["evaluate<br/>deviance, balance, Gini"] --> rec["reconcile<br/>vs Table 5.3/5.5, Brauer"]
    pred --> fig["figures"]
    ev --> cmp["compare_frequency_models<br/>leaderboard, findings, Lorenz"]
    nodes["nodes.py: node functions"] --- prep
    init["__init__.py: pipeline + namespaces"] --- fit
```

| File | Role |
| --- | --- |
| `__init__.py` | builds the pipeline: one namespace per model in `MODELS`, then the comparison |
| `nodes.py` | `prepare`, `fit_model`, `predict`, `evaluate`, `figures`, `reconcile`, `make_compare` |
