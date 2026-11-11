---
name: refactor-check
description: Review the deep-actuarial-science code for refactoring opportunities after it has been extended - duplicated logic that should become a shared atomic object, modules that grew too long, parameters hard-coded in code, comments that drifted from the code - and apply the safe ones without changing any result. Use after adding a model or pipeline, before a release, when asked to refactor, simplify or tidy the code, or on the regular upkeep schedule in MAINTENANCE.md.
argument-hint: "[path | all]"
---

# Refactor check

Read `AGENTS.md` §7 first: modular, minimal code; reusable atomic objects with shared
components; few lines, thorough comments; settings in parameters. Extending code is the moment
to look for better structure. **A refactor never changes a published number.**

## 1. Find candidates (report before changing anything)

Scope: the argument, else `src/`, `scripts/` and `tests/`. For each finding note file:line,
what to change and why.

| Look for | Signal | Usual fix |
| --- | --- | --- |
| Duplicated logic | the same 5+ lines in two places; parallel functions per domain (`frequency`, `mortality`, `reserving`) that differ only in names | one shared function or class in `src/deep_actuarial_science/` (e.g. training loops in `models/torch_utils.py`, scoring in `metrics.py`) |
| Long units | a function over ~50 lines or a module over ~500 | split into named steps; one idea per function |
| Hard-coded settings | numbers or lists in code that a user may change (epochs, seeds, thresholds, years) | move to `conf/base/parameters_*.yml` and pass them in |
| Comment drift | a docstring or comment that no longer says what the code does | rewrite it from the code |
| Dead code | unused functions, parameters, imports | delete (ruff finds imports) |
| Twins out of step | a `.sh` and its `.ps1` doing different steps | make them match |

Useful commands:

```bash
.venv/bin/ruff check src scripts tests                       # unused code, style
python3 scripts/check_readmes.py                             # docs still match the code
wc -l src/deep_actuarial_science/**/*.py | sort -n | tail     # longest modules
```

Show the list to the user, most valuable first, before editing.

## 2. Apply the safe ones

- One refactor per commit: `refactor(<scope>): <what>` (conventional commits), on a
  `chore/refactor-<topic>` branch off `develop`.
- Behaviour must not change. Before and after, run `pytest -q` and `kedro run --env smoke`, and
  compare the smoke metrics: `data/08_reporting/*/*/metrics.json` must be identical to rounding.
  A refactor that moves any reference result is not a refactor: revert it.
- Keep every folder README's diagram true (`python3 scripts/check_readmes.py`), add a
  `CHANGELOG.md` line under **Changed**, and update `MAINTENANCE.md` if a workflow changed.

## 3. Report

What changed, what was left and why, and confirmation that the tests and smoke metrics are
unchanged.
