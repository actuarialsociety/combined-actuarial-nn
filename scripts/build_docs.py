#!/usr/bin/env python3
"""Build docs/: the one-page visual report and its slide-set version, from reference/.

    python3 scripts/build_docs.py          # writes docs/index.html and docs/slides.md
    python3 scripts/build_docs.py --check  # changes nothing; exit 1 if docs/ is out of date

Run by `freeze_reference.py`, so the docs always show the frozen reference run.

- `docs/index.html`: succinct and visual (interactive plotly charts, hover for detail), with
  detail behind click-to-expand panels. It states the model version and data used, gives the
  commands that reproduce it, and links to the running systems (MLflow, W&B, GitHub, wiki).
  The published value is the ground truth on every chart, so a reviewer sees at a glance
  how far each reproduction is from its paper.
- `docs/slides.md`: the same story as a Marp deck (published with the lesson slides).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REF, OUT = ROOT / "reference", ROOT / "docs"
REPO = "https://github.com/actuarialsociety/deep-actuarial-science"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
# The repo's chart palette (plots.py): blue = PASS / real, red = FAIL / published, green = ours.
BLUE, RED, GREEN, MUTED = "#2563eb", "#dc2626", "#059669", "#8c959f"
DATA = [  # what every model is trained and tested on
    ("freMTPL2 claim frequency", "678,007 French motor policies, fixed learn/test split",
     "Wüthrich & Merz (2023), AI Tools for Actuaries data"),
    ("Swiss mortality (HMD)", "log m(x,t), ages 0-89, 1900-2021; learn ≤ 1999",
     "Human Mortality Database via AI Tools for Actuaries"),
    ("CAS Schedule P", "paid triangles 1988-1997; Meyers' 50 auto companies",
     "Meyers & Shi (2011), CAS Loss Reserving Database"),
]  # fmt: skip
LINKS = [  # the running systems behind the numbers
    ("GitHub repository", REPO),
    ("Wiki (lessons)", f"{REPO}/wiki"),
    ("Slides", "https://actuarialsociety.github.io/deep-actuarial-science/"),
    ("CI runs", f"{REPO}/actions"),
    ("MLflow UI (local: mlflow ui --backend-store-uri sqlite:///mlflow.db)", "http://127.0.0.1:5000"),
    ("Weights & Biases", "https://wandb.ai/home"),
]  # fmt: skip


def load() -> dict:
    return {
        "rec": pd.read_csv(REF / "reconciliation.csv"),
        "env": json.loads((REF / "environment.json").read_text()),
        "freq": pd.read_csv(REF / "frequency_findings.csv"),
        "mort": pd.read_csv(REF / "mortality_findings.csv"),
        "res": pd.read_csv(REF / "reserving_findings.csv"),
        "metrics": pd.read_csv(REF / "metrics.csv"),
    }


# --- charts (plotly.js figure specs: data + layout) -------------------------------------------


def chart_distance(rec: pd.DataFrame) -> dict:
    """Each benchmark as (our value - paper) / tolerance: inside the shaded band = PASS."""
    r = rec.assign(
        label=rec.model + " · " + rec.metric,
        z=(rec.difference / rec.tolerance).clip(-3, 3),
    ).iloc[::-1]
    traces = []
    for status, colour, symbol in (("PASS", BLUE, "circle"), ("FAIL", RED, "x")):
        d = r[r.status == status]
        traces.append({
            "type": "scatter", "mode": "markers", "name": status, "x": d.z.round(3).tolist(),
            "y": d.label.tolist(), "marker": {"color": colour, "symbol": symbol, "size": 10},
            "customdata": d[["expected", "actual", "tolerance", "source"]].astype(str).values.tolist(),
            "hovertemplate": "%{y}<br>paper %{customdata[0]} · ours %{customdata[1]}"
                             "<br>tolerance ±%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        })  # fmt: skip
    layout = {
        "xaxis": {"title": "distance from the paper, in tolerances (clipped at ±3)",
                  "range": [-3.3, 3.3], "zeroline": True},
        "shapes": [{"type": "rect", "x0": -1, "x1": 1, "y0": 0, "y1": 1, "yref": "paper",
                    "fillcolor": BLUE, "opacity": 0.08, "line": {"width": 0}}],
        "height": 26 * len(r) + 90, "margin": {"l": 230},
    }  # fmt: skip
    return {"data": traces, "layout": layout}


def chart_gaps(freq: pd.DataFrame) -> dict:
    """Frequency ranking, step by step: each gap with its 95% interval; real vs noise."""
    sys.path.insert(0, str(ROOT / "src"))
    from deep_actuarial_science.findings import summary

    s = summary(freq).iloc[::-1]
    real = s.verdict == "real"
    return {
        "data": [{
            "type": "scatter", "mode": "markers", "x": s.gap.round(4).tolist(),
            "y": (s.worse + " → " + s.better).tolist(),
            "error_x": {"type": "data", "symmetric": False,
                        "array": (s.ci_high - s.gap).round(4).tolist(),
                        "arrayminus": (s.gap - s.ci_low).round(4).tolist(), "color": MUTED},
            "marker": {"color": [BLUE if x else MUTED for x in real], "size": 10,
                       "symbol": ["circle" if x else "circle-open" for x in real]},
            "customdata": s[["verdict", "p_holm"]].round(4).astype(str).values.tolist(),
            "hovertemplate": "%{y}<br>gap %{x} (95% interval)<br>%{customdata[0]}, "
                             "Holm p %{customdata[1]}<extra></extra>",
        }],
        "layout": {"xaxis": {"title": "gap in test deviance (×10⁻²); filled = real"},
                   "height": 28 * len(s) + 90, "margin": {"l": 230}, "showlegend": False},
    }  # fmt: skip


def chart_seeds(mort: pd.DataFrame) -> dict:
    """Mortality: our 15-seed range and mean against the book's single published run."""
    m = mort[mort.seeds > 1]
    lc = float(mort[mort.model == "lee_carter"].seed_mean.iloc[0])
    return {
        "data": [
            {"type": "scatter", "mode": "markers", "name": "our seeds: min–max, mean",
             "x": m.model.tolist(), "y": m.seed_mean.round(4).tolist(),
             "error_y": {"type": "data", "symmetric": False, "color": MUTED,
                         "array": (m.seed_max - m.seed_mean).round(4).tolist(),
                         "arrayminus": (m.seed_mean - m.seed_min).round(4).tolist()},
             "marker": {"color": GREEN, "size": 10}},
            {"type": "scatter", "mode": "markers", "name": "book's published run",
             "x": m.model.tolist(), "y": m.published.tolist(),
             "marker": {"color": RED, "symbol": "diamond", "size": 12},
             "customdata": m.published_run.tolist(),
             "hovertemplate": "%{x}: %{y} (%{customdata})<extra></extra>"},
        ],
        "layout": {"yaxis": {"title": "out-of-sample RMSE of log m (lower is better)",
                             "range": [0.25, float(m.seed_max.max()) + 0.05]},
                   "shapes": [{"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": lc,
                               "y1": lc, "line": {"color": MUTED, "dash": "dot"}}],
                   "annotations": [{"x": 0, "xref": "paper", "y": lc, "text": "Lee–Carter",
                                    "showarrow": False, "xanchor": "left", "yanchor": "top"}],
                   "height": 400},
    }  # fmt: skip


# --- page ----------------------------------------------------------------------------------------


def table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.4g}")


def html(d: dict) -> str:
    rec, env = d["rec"], d["env"]
    n_pass, n_fail = int((rec.status == "PASS").sum()), int((rec.status == "FAIL").sum())
    fails = rec[rec.status == "FAIL"]
    m = d["metrics"].pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    res = m.loc[["chain_ladder", "neural_odp"], ["total_reserve_error_pct", "mape_ultimate"]]
    charts = {"distance": chart_distance(rec), "gaps": chart_gaps(d["freq"]),
              "seeds": chart_seeds(d["mort"])}  # fmt: skip
    data_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in DATA)
    links = " · ".join(f'<a href="{u}">{t}</a>' for t, u in LINKS)
    fail_list = "".join(
        f"<li><b>{r.model}</b> {r.metric}: ours {r.actual:.4g}, paper {r.expected:.4g} "
        f"(tolerance ±{r.tolerance:.3g})</li>"
        for r in fails.itertuples()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark Report</title>
<script src="{PLOTLY}"></script>
<style>
:root {{ --bg:#ffffff; --ink:#1f2328; --muted:#6e7781; --card:#f6f8fa; --line:#d0d7de; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#0d1117; --ink:#e6edf3; --muted:#8c959f; --card:#161b22; --line:#30363d; }} }}
:root[data-theme="dark"] {{ --bg:#0d1117; --ink:#e6edf3; --muted:#8c959f; --card:#161b22; --line:#30363d; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,sans-serif;
  max-width:980px; margin:0 auto; padding:24px 16px; }}
a {{ color:{BLUE}; }} h1 {{ margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 16px; }}
.kpi b {{ display:block; font-size:28px; }}
details {{ border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin:10px 0; }}
summary {{ cursor:pointer; font-weight:600; }}
.t {{ border-collapse:collapse; font-size:13px; width:100%; overflow-x:auto; display:block; }}
.t td, .t th {{ border-bottom:1px solid var(--line); padding:3px 8px; text-align:left; }}
code {{ background:var(--card); padding:1px 4px; border-radius:4px; }}
</style></head><body>
<h1>Deep actuarial science: benchmark report</h1>
<div class="muted">Model version <b>{env.get("package_version", "?")}</b> · commit
<a href="{REPO}/commit/{env["commit"]}"><code>{env["commit"]}</code></a> · frozen {env["frozen"]}
· device {env["device"]}</div>
<div class="muted">{links}</div>

<div class="kpis">
<div class="kpi"><b>{n_pass}</b>PASS</div>
<div class="kpi"><b style="color:{RED}">{n_fail}</b>FAIL</div>
<div class="kpi"><b>{len(rec.model.unique())}</b>models reconciled</div>
</div>
<ul>{fail_list}</ul>

<h2>How far is each reproduction from its paper?</h2>
<p class="muted">Ground truth = the paper's published value (0). Inside the shaded band = within
tolerance. Hover a point for both values and the table it comes from.</p>
<div id="distance"></div>

<h2>Which gaps between models are real?</h2>
<p class="muted">Claim frequency, step by step down the leaderboard: 95% interval of the paired
gap over test policies; filled = real after Holm adjustment.</p>
<div id="gaps"></div>

<h2>Is the book's single mortality run typical?</h2>
<p class="muted">Our 15 seeds (range and mean) against the published run; below the dotted line beats
Lee–Carter.</p>
<div id="seeds"></div>

<h2>Reserving</h2>
<p>Total reserve error: chain-ladder {res.loc["chain_ladder", "total_reserve_error_pct"]:+.1f}%,
neural ODP {res.loc["neural_odp", "total_reserve_error_pct"]:+.1f}% (positive = over-reserved).
MAPE of ultimate: {res.loc["chain_ladder", "mape_ultimate"]:.4f} vs
{res.loc["neural_odp", "mape_ultimate"]:.4f}, gap
{d["res"].verdict.iloc[0]} (Holm p {d["res"].p_holm.iloc[0]:.2f}).</p>

<h2>Detail</h2>
<details><summary>Data used</summary><table class="t"><tr><th>Dataset</th><th>Scope</th>
<th>Source</th></tr>{data_rows}</table></details>
<details><summary>Every benchmark (paper, ours, tolerance, source)</summary>
{table(rec[["pipeline", "model", "metric", "expected", "actual", "tolerance", "status", "source"]])}
</details>
<details><summary>Every gap between models</summary>
{
        table(
            pd.concat(
                [d["freq"].assign(pipeline="frequency"), d["res"].assign(pipeline="reserving")]
            )[["pipeline", "worse", "better", "gap", "ci_low", "ci_high", "p_holm", "verdict"]]
        )
    }</details>
<details><summary>Mortality seeds vs the published run</summary>{table(d["mort"])}</details>
<details><summary>Reproduce</summary><pre><code>git clone {REPO} && cd deep-actuarial-science
git checkout {env["commit"]}
uv venv && uv pip install -e ".[dev,wandb]"
scripts/full_run.sh reference     # or scripts/full_run.ps1 reference
python3 scripts/build_docs.py</code></pre>
Packages: {", ".join(f"{k} {v}" for k, v in env["packages"].items())}</details>

<script>
const charts = {json.dumps(charts)};
const dark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" &&
   matchMedia("(prefers-color-scheme: dark)").matches);
function draw() {{
  const ink = dark() ? "#e6edf3" : "#1f2328", grid = dark() ? "#30363d" : "#d0d7de";
  for (const [id, f] of Object.entries(charts)) {{
    const ax = a => Object.assign({{gridcolor: grid, zerolinecolor: grid, automargin: true}}, a);
    Plotly.react(id, f.data, Object.assign({{}}, f.layout, {{
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {{color: ink}},
      xaxis: ax(f.layout.xaxis), yaxis: ax(f.layout.yaxis), legend: {{orientation: "h", y: 1.12}},
      margin: Object.assign({{t: 40, r: 10, b: 50}}, f.layout.margin)
    }}), {{responsive: true, displaylogo: false}});
  }}
}}
draw(); matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script>
</body></html>
"""


def slides(d: dict) -> str:
    """The same story as a Marp deck: version, scorecard, findings, links."""
    rec, env = d["rec"], d["env"]
    n_pass, n_fail = int((rec.status == "PASS").sum()), int((rec.status == "FAIL").sum())
    fails = "\n".join(
        f"- **{r.model}** {r.metric}: ours {r.actual:.4g}, paper {r.expected:.4g}"
        for r in rec[rec.status == "FAIL"].itertuples()
    )
    mort = d["mort"][d["mort"].seeds > 1]
    seeds = "\n".join(f"| {r.model} | {r.published} | {r.seed_min:.3f}–{r.seed_max:.3f} | "
                      f"{r.published_run} |" for r in mort.itertuples())  # fmt: skip
    return f"""---
marp: true
paginate: true
title: "Benchmark report: summary"
---

# Deep actuarial science: benchmark summary

Model version **{env.get("package_version", "?")}**, commit `{env["commit"]}`, frozen {env["frozen"]}.
Full interactive report: `docs/index.html`.

---

## Scorecard: {n_pass} PASS, {n_fail} FAIL

{fails}

![h:330](../lessons/figures/00-leaderboard.png)

---

## Is the book's single mortality run typical?

| Network | Published | Our 15 seeds | Published run is |
| --- | --- | --- | --- |
{seeds}

![h:280](../lessons/figures/05-seeds.png)

---

## Which gaps are real?

![h:420](../lessons/figures/00-gaps.png)

---

## Data, reproduce, systems

- Data: freMTPL2; Swiss HMD; CAS Schedule P (Meyers' 50 companies)
- Reproduce: `git checkout {env["commit"]}`, then `scripts/full_run.sh reference`
- [Repository]({REPO}) · [Wiki]({REPO}/wiki) · [CI]({REPO}/actions) · MLflow: `mlflow ui`
"""


def main(check: bool = False) -> int:
    d = load()
    outputs = {OUT / "index.html": html(d), OUT / "slides.md": slides(d)}
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text() != text]
    if check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)} (run scripts/build_docs.py)")
        return 1 if stale else 0
    OUT.mkdir(exist_ok=True)
    for p in stale:
        p.write_text(outputs[p])
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
