# `pipelines/mortality/`

Swiss mortality: learn 1900–1999, forecast 2000–2021, RMSE of log m(x,t) against
Richman, Scognamiglio & Wüthrich (2026), Ch. 8.

```mermaid
flowchart TD
    raw[Swiss HMD] --> prep["prepare<br/>long table, log m"]
    prep --> fit["fit_forecast<br/>15 seeds per network"] --> ev["evaluate<br/>RMSE in / out / ages 15+"]
    ev --> rec["reconcile<br/>book Ch. 8"]
    fit --> fig["figures<br/>residual surface, training"]
    ev --> cmp["compare<br/>seeds vs published run"]
    init["__init__.py: nodes + pipeline"] --- fit
```

| File | Role |
| --- | --- |
| `__init__.py` | nodes (`fit_forecast`, `evaluate`, `figures`, `reconcile`, `compare`) and the pipeline |
