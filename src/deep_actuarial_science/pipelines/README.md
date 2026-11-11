# `pipelines/`

One Kedro pipeline per domain. Each model is its own namespace (`kedro run --namespaces fnn`),
so the dataset factories in `conf/base/catalog.yml` cover every model without catalog edits.

```mermaid
flowchart TD
    ingest["ingest/<br/>download + cache data"] --> frequency["frequency/<br/>7 models"]
    ingest --> mortality["mortality/<br/>5 models"]
    ingest --> reserving["reserving/<br/>2 models"]
    subgraph each["every model namespace"]
        fit --> evaluate --> reconcile
        fit --> figures
    end
    frequency --> each
    mortality --> each
    reserving --> each
    each --> compare["compare: leaderboard, findings, figures"]
    init["__init__.py"]
```

| Folder | Pipeline |
| --- | --- |
| `__init__.py` | package marker |
| `ingest/` | AI Tools data (freMTPL2, Swiss HMD) and CAS Schedule P, cached in `data/01_raw` |
| `frequency/` | claim frequency on freMTPL2 |
| `mortality/` | Swiss mortality forecasting |
| `reserving/` | paid-loss reserving, back-tested on the realised lower triangle |
