---
name: refresh-lessons
description: Audit and update the deep-actuarial-science lessons. Fixes dead links, re-runs pipelines to confirm results still reconcile to their benchmarks, finds newer papers and better reference results with Exa, and keeps each Marp deck succinct. Use when asked to refresh, update, audit or check the lessons, or when the scheduled lesson-refresh workflow runs.
argument-hint: "[lessons/<file> | all]"
---

# Refresh lessons

Read `AGENTS.md` first. Keep lessons **current, reconciled and succinct**.

## 1. Find what's broken

```bash
python3 scripts/check_links.py            # or pass files; --json for a report
pytest -q && kedro run --env smoke        # code still runs with current dependencies
```

Fix broken links (search Exa for the moved page first; delete only if nothing exists).

## 2. Pick lessons

With an argument, refresh that lesson. With `all` or nothing, take the up to **3**
lessons with the oldest `_Last reviewed:` date. Deep-learning tooling (PyTorch, Kedro,
MLflow APIs) goes stale in ~3 months; actuarial theory in ~18 months.

## 3. Per lesson

- Re-run its models at the reference setup and check every reconciliation table is
  still PASS. Update the results slide and figures (`scripts/export_figures.py`) from
  the new run; never edit numbers by hand.
- Search Exa for papers published after `_Last reviewed:` that improve the model or
  publish a better benchmark. Verify DOIs with Crossref before citing.
- Keep the Theory slide to one or two equations; cut rather than add. Depth belongs in the
  `## Core ideas` panels: check they still match the code and config, and run
  `python3 scripts/fix_math.py --check`. Every equation keeps its label and `where …` line; run `python3 scripts/link_citations.py` after changing references.
- Set `_Last reviewed: <today>` on every lesson you reviewed.

## 4. Report

Per lesson: what changed, benchmark status, papers considered and rejected. In CI,
put this in the PR body.
