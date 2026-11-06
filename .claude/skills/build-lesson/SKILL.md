---
name: build-lesson
description: Take a deep learning model or paper in actuarial science from theory to a benchmarked lesson in this deep-actuarial-science repo. Finds the paper and reference results with Exa, implements the model in the Kedro pipeline, reconciles it to a benchmark, exports figures, and writes a Marp slide deck plus a Colab notebook. Use when asked to add, create or draft a lesson, model or course module.
argument-hint: <model or paper> [domain: frequency|mortality|reserving]
---

# Build a lesson

Goal: new theory → workable, reconciled analysis, fast. Read `AGENTS.md` first.

## 1. Scope (no tools)

- Domain pipeline (`frequency`, `mortality`, `reserving`, or a new one), the baseline
  it must beat, and 3 objectives the learner can *do*.
- If the model is a variant of an existing one, add it as a new namespace, not a new
  pipeline.

## 2. Research: paper, equations and a benchmark

```bash
curl -sS -X POST "https://api.exa.ai/search" \
  -H "Content-Type: application/json" -H "x-api-key: $EXA_API_KEY" \
  -d '{"query": "<model> actuarial neural network paper", "type": "auto", "numResults": 8,
       "includeDomains": ["cambridge.org", "tandfonline.com", "springer.com", "mdpi.com", "ssrn.com", "arxiv.org", "github.com"]}' \
  | jq -r '.results[] | "\(.publishedDate[:10])  \(.title)  \(.url)"'
curl -s "https://api.crossref.org/works/<DOI>" | jq '.message | {author: [.author[].family], year: .issued["date-parts"][0][0], title: .title[0], journal: ."container-title"[0], volume, issue, page}'
```

- Use `exa-contents` with a `summary` query to read abstracts before you describe a paper.
- Find a **reference result** on a public dataset (paper table, or the authors' notebook
  outputs, e.g. wueth/AITools4Actuaries). Record it with its source in
  `conf/base/parameters_benchmarks.yml`. No like-for-like result: say so, and back-test.

## 3. Implement and run

1. Model class in `src/deep_actuarial_science/models/<domain>.py`; register it.
2. Parameters under `<domain>.models.<name>`; add the namespace to the pipeline's `MODELS`.
3. `kedro run --env smoke --pipeline <domain> --namespaces <name>` until clean.
4. Reference run: `kedro run --pipeline <domain> --namespaces <baseline>,<name>`.
   Check `data/08_reporting/<domain>/<name>/reconciliation.csv`. Investigate any FAIL
   before writing the lesson; never loosen a tolerance to make it pass.
5. Add a test in `tests/` for the model's defining property (e.g. initialisation).

## 4. Write the lesson

- Copy `.claude/skills/build-lesson/template.md` to `lessons/NN Title.md`. Theory: one or
  two equations. Results: real numbers from the run and figures exported with
  `python3 scripts/export_figures.py`.
- Add the notebook entry to `scripts/make_notebooks.py` and regenerate.
- Add the lesson to the README table.

## 5. Verify, then report

```bash
pytest -q && kedro run --env smoke && python3 scripts/check_links.py
```

Report: the model, the benchmark and whether it reconciled, the figures, the papers
cited, and anything you could not reproduce. Do not commit unless asked.
