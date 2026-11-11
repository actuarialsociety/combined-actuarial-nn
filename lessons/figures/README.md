# `lessons/figures/`

Figures from pipeline runs, copied by `scripts/export_figures.py` (NN-name.png, NN = the
lesson). Transparent PNGs styled in `plots.py` to read on light and dark pages. Never edit by
hand: re-run and export.

```mermaid
flowchart LR
    pipe[pipeline figures<br/>data/08_reporting] --> exp[export_figures.py] --> png["`*.png`"] --> deck[lessons/*.md]
```

Files: `*.png`.
