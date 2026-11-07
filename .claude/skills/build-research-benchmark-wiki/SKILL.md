---
name: build-research-benchmark-wiki
description: Reproduce a published actuarial / deep-learning research result in the Kedro benchmark framework and publish it as a GitHub wiki lesson. Finds the paper, its results table and its reference implementation with Exa, implements the model as the paper does, reconciles to the published scores (PASS/FAIL against a named table), and writes the Marp lesson, Colab notebook and wiki page with click-through references. Use when asked to add, reproduce, benchmark or replicate a paper or model, add a research benchmark or wiki lesson, or check a model against published results.
argument-hint: <paper, model or DOI> [domain: frequency|mortality|reserving|new]
---

# Build a research benchmark wiki lesson

**Goal:** paper → reference implementation → Kedro pipeline → reproduced score → PASS/FAIL
against a **named table** → wiki lesson with click-through references. The repo is a
research-paper benchmark toolkit; the lesson is how the reproduction is presented.
Read `AGENTS.md` first: its rules are binding.

## 0. Setup (no reading of secrets)

```bash
set -a && . ./.env && set +a          # loads EXA_API_KEY; never print it, never commit it
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://api.exa.ai/search \
  -H "Content-Type: application/json" -H "x-api-key: $EXA_API_KEY" -d '{"query":"test","numResults":1}'
```

No key: ask the user to `cp .env.example .env` and fill `EXA_API_KEY` themselves. Stop.

## 1. Scope (no tools)

- The paper and the **one table** to reproduce; the domain pipeline (`frequency`,
  `mortality`, `reserving`, or a new one); the baseline it must beat.
- A variant of an existing model is a new namespace, not a new pipeline.

## 2. Research with Exa: the published table and how it was produced

For more than one model or domain, run one research agent per domain in parallel with the
brief in [research-brief.md](research-brief.md); otherwise follow it yourself.

- **Find the table.** Use `exa-search` with `includeDomains` (tandfonline.com,
  cambridge.org, springer.com, sciencedirect.com, mdpi.com, ssrn.com, arxiv.org,
  github.com), then `exa-contents` with `text` to read the results table. Quote the row.
- **Verdict per candidate:** LIKE-FOR-LIKE (same data, split, metric and scale: can be a
  PASS/FAIL benchmark), COMPARABLE (say what differs), or NOT COMPARABLE.
- **Prefer, in order:** the original paper's table → a later peer-reviewed paper that
  reproduces it on the same split (e.g. with seed mean ± sd) → the authors' dated,
  authored companion document → companion notebook outputs. Name which one it is.
- **Reference implementation:** the authors' code, notebook or package (GitHub, the
  paper's listings). Record data cleaning, preprocessing, architecture, loss, optimiser,
  learning rate, batch size, epochs, early stopping and patience, validation scheme,
  seeds and ensembling, stated parameter counts. "Not stated" is a valid answer.
- **Check the metric scale.** Same name, different normaliser is common (deviance per
  policy vs per unit of exposure differ by $n/\sum v_i$; RMSE on rates vs log rates).
  Convert an existing model's numbers to confirm the factor before trusting a table.
- **Verify every DOI** with `curl -s https://api.crossref.org/works/<DOI>` (title,
  authors, year, volume, pages). Never type bibliographic details from memory.
- **Search for newer work** since the lesson's `_Last reviewed:` date that reports
  benchmarkable numbers on the same data; cite it as context or a next lesson.

## 3. Record the benchmark

`conf/base/parameters_benchmarks.yml`, under `benchmarks.<domain>.<model>`:

```yaml
cann:
  source: [*brauer, "Table 2 (g) CANN on GLM3, mean (sd) over 15 seeds: 23.742 (0.061), 23.810 (0.033)"]
  tolerance: {deviance_learn: 0.122, deviance_test: 0.066}   # 2 x published seed sd
  expected: {deviance_learn: 23.742, deviance_test: 23.810}
```

- `source` = [paper with DOI, table and row]. A list of entries adds cross-checks
  (e.g. a software package), each with its own source.
- **Tolerance:** closed-form fits = rounding of the published figure; seeded networks =
  2 × the published seed sd (state where the sd comes from); a single published run =
  an explicit band, stated in the lesson. Never loosen a tolerance to make a run pass.
- No published like-for-like result: `null` with a comment saying why; the lesson says
  "compared, not reconciled" and back-tests against realised outcomes if possible.

## 4. Implement as the paper does

1. Model class in `src/deep_actuarial_science/models/<domain>.py`, registered in `MODELS`;
   the docstring names the paper, section and any deviation with its reason.
2. Parameters under `<domain>.models.<name>` with a comment citing the paper's settings.
3. **Match the paper's framework defaults.** Typical Keras vs PyTorch traps: nadam
   learning rate 0.001 (PyTorch default 0.002); Keras reshuffles each epoch; Keras LSTM
   `activation` (often `linear`) and one bias per gate; `layer_attention(use_scale=TRUE)`
   is a learned scale, not $1/\sqrt d$; `validation_split` takes the last 10%;
   `EarlyStopping(patience, restore_best_weights)`.
4. **Check parameter counts** against the paper's stated counts with
   `models.torch_utils.n_params`; they must match exactly.
5. **Seeds:** if the paper reports mean ± sd over $k$ seeds, set `n_runs: k`; each run gets
   its own seed and validation split, and metrics report the mean, `_sd` and `ensemble`.
6. A test in `tests/` for the model's defining property (e.g. starts exactly at its base
   model at zero epochs; parameter count; a closed form equal to a package).

## 5. Run and reconcile

```bash
kedro run --env smoke --pipeline <domain> --namespaces <name>     # until clean
kedro run --pipeline <domain> --to-outputs <prepared datasets>    # rebuild data after smoke runs
kedro run --pipeline <domain> --namespaces <baseline>,<name>      # reference setup
```

- Closed-form models (GLMs, Lee–Carter, chain-ladder): check them directly first; they
  must hit the table to the published decimals.
- **Long or GPU runs need the user's go-ahead.** Training uses CUDA when present
  (`DAS_DEVICE` overrides). Without a local GPU, point the user to
  `notebooks/reference_run.ipynb`: it opens on a Colab T4 GPU runtime (and refuses a CPU
  one), runs everything, refreshes figures and numbers and zips `lessons/` for a PR.
- Read `data/08_reporting/<domain>/<name>/reconciliation.csv`. Investigate every FAIL
  (scale, split, preprocessing, framework default) before writing it up; report what
  remains honestly.

## 6. Write the lesson (`.claude/skills/build-lesson/template.md` → `lessons/NN Title.md`)

- Title slide: `**Reproduces:** Author (Year) [N], Table X`.
- Theory slide (`<!-- _class: theory -->`): one or two equations, each with a bold label
  naming the model and its citation, and a `where …` line defining every symbol.
- `## Core ideas`: 5–7 click-through `<details>` panels, each one idea with its equation
  and the exact settings the code uses; no math in `<summary>`.
- Results slide titled with the paper and table: columns Paper/table · Published · This
  repo · Status. Run numbers only via `<!--r:model.metric-->` markers (`pending` until the
  reference run, then `python3 scripts/fill_results.py`); figures via
  `python3 scripts/export_figures.py`. Never type a run number by hand.
- "In production" checks and 3–6 references with verified DOIs, describing a paper only
  as far as its abstract and quoted tables support.
- Then `python3 scripts/link_citations.py` (every `[N]` links to the paper) and
  `python3 scripts/fix_math.py` (GitHub-safe LaTeX).

## 7. Notebook, README, wiki

- Notebook entry in `scripts/make_notebooks.py`; run `python3 scripts/make_notebooks.py`.
  Never hand-edit `.ipynb` files. **Colab notebooks must run on a GPU:** the generator
  writes `"accelerator": "GPU"` and `"colab": {"gpuType": "T4"}` into the metadata (Colab
  opens a T4 runtime) and puts `GPU_GUARD` first in the setup cell, which stops the
  notebook on a CPU runtime with instructions (`DAS_ALLOW_CPU=1` to override; CI and
  local runs are unaffected). `test_colab_notebooks_force_a_gpu_runtime` enforces both.
- README benchmark table row: lesson · model · reproduces (paper, table) · Colab badge.
- The wiki is generated by `scripts/build_wiki.py` and published by `wiki-sync.yml` on
  merge: Colab links in the sidebar, nav bars, Notebooks page, plain page links (never
  `[[Text|Page]]` inside tables).

## 8. Verify

```bash
pytest -q                                                          # includes the Colab GPU check
kedro run --env smoke
python3 scripts/check_links.py                                     # 0 broken
python3 scripts/link_citations.py --check
GITHUB_TOKEN=$(gh auth token) python3 scripts/fix_math.py --github  # every math span renders
npx -y @marp-team/marp-cli@4.5.1 --input-dir lessons --output /tmp/site < /dev/null  # 0 katex-error
python3 scripts/build_wiki.py /tmp/wiki
```

Render the theory and results slides to PNG (`--images png`) and look at them: nothing
may overflow.

## 9. Report

A table per model: paper, table, published value, this repo, tolerance, status; the
implementation deviations and why; papers considered and rejected (with the verdict);
what still needs the reference run. Commit, PR and merge per the user's standing
instructions.

## Starting a new repo with this framework

Copy from `actuarialsociety/deep-actuarial-science`: `src/deep_actuarial_science/`
(`benchmarks.py`, `metrics.py`, `notebook.py`, `plots.py`, a domain pipeline),
`conf/base/parameters_benchmarks.yml`, `conf/smoke/`, `scripts/` (`build_wiki.py`,
`link_citations.py`, `fix_math.py`, `check_links.py`, `fill_results.py`,
`export_figures.py`, `make_notebooks.py`), `marp.config.mjs`, `.github/workflows/`
(`ci.yml`, `link-check.yml`, `wiki-sync.yml`, `slides.yml`), `AGENTS.md`, and this skill.
Then create the wiki's first page once in the GitHub UI so `wiki-sync.yml` can push.
