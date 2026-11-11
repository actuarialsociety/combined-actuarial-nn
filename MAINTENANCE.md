# Maintenance

How this repo stays correct. Agent rules: [AGENTS.md](./AGENTS.md). History: [CHANGELOG.md](./CHANGELOG.md).

## On every commit

1. **Changelog:** add a line under `Unreleased` in [CHANGELOG.md](./CHANGELOG.md) (rules at its top). CI fails a pull request that does not touch it.
2. **This file:** update it in the same commit when a workflow, script, check or known issue below changes.
3. **Documentation travels with the code:** a change to a model, loss, metric or training setting updates its docstring, its `conf/base/parameters_*.yml` comment and its lesson (Core ideas panel) in the same commit. Deviations from the benchmark paper are stated with their reason.
4. **No hand-typed results:** numbers in lessons and the README are `<!--r:key-->` markers filled from `reference/` by `scripts/fill_results.py`.
5. **Checks pass:**

```bash
ruff check . && ruff format --check src tests
pytest -q                                   # includes: lessons and README agree with reference/
kedro run --env smoke                       # every pipeline, ~1-2 minutes
python3 scripts/make_notebooks.py           # if notebooks changed (never edit the .ipynb)
python3 scripts/check_links.py              # 0 broken
python3 scripts/link_citations.py --check   # citations click through
GITHUB_TOKEN=$(gh auth token) python3 scripts/fix_math.py --github   # math renders on GitHub
```

## When results change

A change to a model, metric, data preparation or seed moves published numbers. Then:

```bash
scripts/full_run.sh reference          # every pipeline at the reference setup (hours: 15 seeds per network)
                                       # ... which ends with: python3 scripts/freeze_reference.py
git diff reference/ lessons/ README.md # review every number that moved
```

`scripts/freeze_reference.py` writes `reference/` (metrics, reconciliation, findings, environment), copies the lesson figures and refills the lessons and README. It refuses a run that is not the reference setup. One pipeline is enough when only it changed: `kedro run --pipeline reserving`, then freeze. Say in the changelog which numbers moved and why. A PASS that turns into a FAIL is reported, never tuned away.

| File in `reference/` | Holds |
| --- | --- |
| `metrics.csv` | every metric of every model |
| `reconciliation.csv` | paper value, this repo's value, tolerance, PASS/FAIL, source |
| `<pipeline>_findings.csv` | which gaps between models are real; where a published run falls among our seeds |
| `environment.json` | date, commit, device and package versions of the freeze |

## Routine upkeep

| Task | How |
| --- | --- |
| Refresh lessons (links, newer papers) | `/refresh-lessons`; [`lesson-refresh.yml`](./.github/workflows/lesson-refresh.yml) opens a PR monthly |
| Add a paper | `/build-research-benchmark-wiki <paper>`; steps in [AGENTS.md](./AGENTS.md) §3 |
| EDA and model-analysis reports (local) | `python scripts/eda.py`, `python scripts/model_analysis.py` |
| Research library (local) | `python scripts/research.py fetch`, `index`, `search "..."`, `status` |
| Update vendored agent skills | `scripts/update_*_skills.sh` |
| Wiki and slides | generated on merge to `main` (`wiki-sync.yml`, `slides.yml`); never edit the wiki |

## Known issues

- **macOS, LightGBM:** its OpenMP runtime and PyTorch's clash. The GBM trains single-threaded, and reloading the saved model in a new process can crash (`gbm.freq_model`): set `OMP_NUM_THREADS=1` for that process.
- **Container build looks stuck at `uv pip install`:** it is downloading about 500 MB of wheels; on a slow connection that takes 15 minutes or more. `docker compose --progress plain build lab` shows each download.
- **Mortality on MPS** is slower than on the CPU (small networks); results are the same to rounding.
- **Mortality networks above the book's runs:** every network's out-of-sample RMSE averages above the book's single published run, and for the CNN and LSTM the published run beats every one of our seeds. In-sample errors match. Checked and ruled out: the validation split, the min-max scaling and the recursive forecast (identical to the book's code), and the learning-rate plateau threshold. Open: other PyTorch/Keras differences.
- **Look-ahead in the mortality scaling:** `training.scaler_fit: all` (the book's choice) sets the min-max range from all years, test period included. Run with `--params mortality.training.scaler_fit=learn` to check a conclusion does not depend on it.
