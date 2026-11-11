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
- `findings.py`: which gaps between models are real (paired 95% interval over test policies or companies) and where a published single run falls among our seeds. Each pipeline writes `findings.csv` and a figure.
- `reference/`: the frozen reference run (metrics, reconciliation, findings, environment), written by `scripts/freeze_reference.py`. Lessons and README are filled from it; a test fails if they disagree.
- Notebooks: `reconciliation()` shows paper, reference run and this run; `findings()` shows the gaps against the reference run.
- README "Findings"; lesson 00 Core ideas ⑥ (is a gap real?) and ⑦ (reference run); lesson 05 Core ideas ⑥ and the slide "Is the published run typical?".
- `MAINTENANCE.md`, this changelog, and a CI check that a pull request updates it.

### Changed
- Mortality comparison figure: per-seed dots with the published run (`seeds.png`) replace the bar chart of means.
- Notebook 01 also fits GLM3; notebook 03 also fits the FNN (the comparison the CANN has to win).
- CI also runs when lessons, scripts, `reference/` or the README change.

### Fixed
- Reconciliation rows are labelled with the model's name, not its kind: GLM3's rows were labelled `glm`.
- Reserving: the published benchmark applies only to the reference setup; smoke and scale-up runs report SKIPPED instead of a false FAIL.

## 0.1.0

Initial toolkit: Kedro pipelines for claim frequency (GLM, GBM, FNN, CANN, LocalGLMnet), mortality (Lee–Carter, CNNs, LSTM, Transformer) and reserving (chain-ladder, neural ODP), each reconciled to a published table; Marp lessons and wiki; notebooks for Colab, Kaggle, SageMaker and Azure ML; MLflow and optional W&B tracking; container; local EDA and model-analysis reports.
