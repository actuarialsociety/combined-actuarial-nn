# `pipelines/ingest/`

Downloads the public data once and caches it in `data/01_raw`; the other pipelines reuse its
nodes (`aitools_node`, `cas_node`) so each runs on its own.

```mermaid
flowchart LR
    url1["AI Tools Data.zip<br/>(ETH)"] --> aitools["fetch_aitools"] --> f["freMTPL2 parquet"]
    aitools --> m["Swiss HMD m(x,t)"]
    url2["CAS Schedule P CSVs"] --> cas["fetch_cas"] --> s["cas_schedule_p parquet"]
    init["__init__.py: nodes + pipeline"] --- aitools
    init --- cas
```

| File | Role |
| --- | --- |
| `__init__.py` | the download nodes and the `ingest` pipeline; sources in `conf/base/parameters.yml` |
