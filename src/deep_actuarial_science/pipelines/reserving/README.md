# `pipelines/reserving/`

Paid-loss reserving on CAS Schedule P, back-tested on what was actually paid, reconciled to
Kuo (2019), Table 1, and cross-checked against the `chainladder` package.

```mermaid
flowchart TD
    raw[Schedule P] --> prep["prepare<br/>companies, triangles"]
    prep --> fit["fit_predict<br/>chain-ladder / neural ODP"] --> ev["evaluate<br/>MAPE, RMSPE, reserve errors"]
    ev --> rec["reconcile<br/>Kuo 2019 + package check"]
    ev --> cmp["compare<br/>findings, scatter, errors"]
    init["__init__.py: nodes + pipeline"] --- fit
```

| File | Role |
| --- | --- |
| `__init__.py` | nodes (`fit_predict`, `evaluate`, `reconcile`, `compare`), the package cross-check and the pipeline |
