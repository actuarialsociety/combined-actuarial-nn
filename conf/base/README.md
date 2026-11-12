# `conf/base/`: the reference setup

Every setting the pipelines read. Benchmarks apply only here (`reference_setup: true`); other
environments override single keys (parameters are soft-merged).

```mermaid
flowchart LR
    p["parameters.yml<br/>sources, reference_setup"] --> run((kedro run))
    pf["parameters_frequency.yml"] --> run
    pm["parameters_mortality.yml"] --> run
    pr["parameters_reserving.yml"] --> run
    pb["parameters_benchmarks.yml<br/>published values"] --> run
    c["catalog.yml<br/>dataset factories"] --> run
    ml["mlflow.yml"] --> run
```

| File | Holds |
| --- | --- |
| `parameters.yml` | data sources and `reference_setup` |
| `parameters_frequency.yml`, `parameters_mortality.yml`, `parameters_reserving.yml` | data scale and each model's settings, with the paper they follow |
| `parameters_benchmarks.yml` | published values, tolerances and their source table |
| `catalog.yml` | dataset factories for every namespace |
| `mlflow.yml` | kedro-mlflow tracking (SQLite `mlflow.db`) |
