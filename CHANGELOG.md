# Changelog

Rules (enforced for pull requests by CI, see [MAINTENANCE.md](./MAINTENANCE.md)):

- Every commit that changes code, configuration, results or lessons adds one line under `Unreleased`, in the section that fits: **Results** (a published number moved: say which, from what to what, and why), **Added**, **Changed**, **Fixed**.
- One line per change, written for a reader of the lessons, most important first.
- On a release, rename `Unreleased` to the version and date, and start a new `Unreleased`.

## Unreleased

### Results
- Homogeneous model: Gini 0.082 → 0.000. A constant prediction ranks nothing; the old value came from the order of the data (ties in the Lorenz curve).
- Leaderboard figure: GLM1's published diamond was GLM3's value (24.102); it is now its own (24.146), and GLM3 has one.
- No other reference number moved: frequency was re-scored from the saved predictions, mortality and reserving were re-run at the reference setup and reproduced the previous run.

### Added
- MLflow agent skills (`mlflow-docs`, `mlflow-onboarding`, `mlflow-sagemaker`) vendored from mlflow/skills with `scripts/update_mlflow_skills.sh` / `.ps1`. No official skills exist yet for PyTorch, LightGBM, pandas, scikit-learn, statsmodels or plotly.
- `refactor-check` skill: a regular, behaviour-preserving review for shared objects, long functions, hard-coded settings and comment drift.
- PowerShell twins of every bash script (`full_run.ps1`, `update_*_skills.ps1`, `install_hooks.ps1`) and `login.sh` / `login.ps1` for GitHub, W&B, AWS and Azure; `check_readmes.py` fails if a bash script has no twin. The PowerShell versions are not yet run in CI.
- A README.md with a mermaid diagram in every folder, and `scripts/check_readmes.py`, which fails if a README stops matching its code; run by a pre-push hook (`.githooks/`, enabled with `scripts/install_hooks.sh` or `.ps1`) and by CI.
- `scripts/research.py`: a local library of every paper the repo cites, with open-access PDFs from OpenAlex, repository pages and arXiv, and a full-text search index by page (`data/12_research/`, never committed).
- Per-epoch training logs for every trained model (FNN, CANN, LocalGLMnet, GBM, the four mortality networks, neural ODP): training and validation loss for every seed (and the learning rate where it changes), logged to MLflow as step metrics plus `training/<model>.csv`, and to W&B as one curve chart per model and metric (`training_log.py`).
- Mortality: `rmse_out_age15plus`, the out-of-sample RMSE without ages 0-14, whose few Swiss deaths make the observed rates noisy and carry about half of every model's squared error.
- Findings: two-sided p-values and Holm-adjusted p-values for every gap between models.
- `findings.py`: which gaps between models are real (paired 95% interval over test policies or companies) and where a published single run falls among our seeds. Each pipeline writes `findings.csv` and a figure.
- `reference/`: the frozen reference run (metrics, reconciliation, findings, environment), written by `scripts/freeze_reference.py`. Lessons and README are filled from it; a test fails if they disagree.
- Notebooks: `reconciliation()` shows paper, reference run and this run; `findings()` shows the gaps against the reference run.
- README "Findings"; lesson 00 Core ideas ⑥ (is a gap real?) and ⑦ (reference run); lesson 05 Core ideas ⑥ and the slide "Is the published run typical?".
- `MAINTENANCE.md`, this changelog, and a CI check that a pull request updates it.

### Changed
- Lesson scripts and the slide build read only the numbered decks (`lessons/NN *.md`), so `lessons/README.md` is not published as a lesson.
- Git flow: feature branches off `develop` with conventional branch names and commits; CI also runs on pushes to `develop`. AGENTS.md §7 records the maintainer's working principles.
- FNN and CANN are reconciled to Brauer (2024) with a tolerance of 2 standard errors of the gap between two 15-seed means (`2se`), not 2 seed sd, which was about 2.7 times too wide.
- Findings: a gap is called real only if its Holm-adjusted p-value is below 0.05 over all pairs in the table; the 95% intervals stay per pair.
- Mortality networks: a seed now also sets every epoch's mini-batch order (as Keras's `set_random_seed` does), not only the initial weights, so the seed spread behind the `2sd` tolerance is not understated; the learning-rate plateau uses Keras's absolute `min_delta`.
- Mortality comparison figure: per-seed dots with the published run (`seeds.png`) replace the bar chart of means.
- Notebook 01 also fits GLM3; notebook 03 also fits the FNN (the comparison the CANN has to win).
- CI also runs when lessons, scripts, `reference/` or the README change.

### Fixed
- Link check: two lessons had unlinked citations; the synthetic telematics data page (UConn) is offline, so Open Actuarial Data links its paper only.
- Neural ODP was never trained (best epoch 0 in every run): it chose its epochs on the 1997 diagonal using chain-ladder offsets fitted to that same diagonal. Epochs are now chosen out of time (offsets and training from calendar years up to 1996), then the network is refitted on the whole upper triangle.
- Tests and notebooks brought in line with the reconciliation API (`reconcile(..., reference, name)`).
- Reconciliation rows are labelled with the model's name, not its kind: GLM3's rows were labelled `glm`.
- Reserving: the published benchmark applies only to the reference setup; smoke and scale-up runs report SKIPPED instead of a false FAIL.

## 0.1.0

Initial toolkit: Kedro pipelines for claim frequency (GLM, GBM, FNN, CANN, LocalGLMnet), mortality (Lee–Carter, CNNs, LSTM, Transformer) and reserving (chain-ladder, neural ODP), each reconciled to a published table; Marp lessons and wiki; notebooks for Colab, Kaggle, SageMaker and Azure ML; MLflow and optional W&B tracking; container; local EDA and model-analysis reports.
