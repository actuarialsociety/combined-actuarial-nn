# `tests/`

`pytest -q`: fast checks of the scoring rules, models, reconciliation, findings, notebooks and
that the lessons and README agree with `reference/`. The smoke run (`kedro run --env smoke`)
covers the pipelines end to end.

```mermaid
flowchart LR
    t["test_core.py"] --> m[metrics + findings]
    t --> mod[models: GLM, CL, neural ODP, Keras init]
    t --> rec[reconciliation + tolerances]
    t --> nb[notebooks + fill_results vs reference/]
```

| File | Covers |
| --- | --- |
| `test_core.py` | all unit and consistency tests |
