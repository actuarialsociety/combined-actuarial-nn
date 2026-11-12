# Open Actuarial Data
> *Public datasets for actuarial modelling, chosen to be as close to real portfolios as possible. ✅ = already wired into the pipelines (downloaded and cached by the `ingest` nodes).*

Licences and terms vary: check each source before redistributing. Never commit data to this repo; add a download node instead (see [Adding a dataset](#adding-a-dataset)).

## General insurance: pricing

| Dataset | What | Scale | Access |
| --- | --- | --- | --- |
| ✅ **freMTPL2freq / freMTPL2sev** | French motor third-party liability: claim counts, exposure, 9 rating factors; claim amounts | 678,013 policies; 26,639 claims | [AI Tools for Actuaries `Data.zip`](https://aitools4actuaries.com/) (cleaned, fixed learn/test split, used here) · [OpenML 41214](https://www.openml.org/d/41214) · [OpenML 41215](https://www.openml.org/d/41215) · [CASdatasets](https://dutangc.github.io/CASdatasets/reference/freMTPL.html) |
| **CASdatasets** | A large collection of insurance datasets (motor, property, health, reinsurance, life) from many countries, collected for Charpentier (2014) | 1k to 1M+ rows | [R package](https://dutangc.github.io/CASdatasets/) · [GitHub](https://github.com/dutangc/CASdatasets) |
| **insuranceData** | Classic risk-classification datasets (e.g. car insurance from de Jong & Heller) | 1k to 70k rows | [CRAN](https://cran.r-project.org/package=insuranceData) |
| **Synthetic telematics** | 100,000 policies with traditional rating factors plus 39 telematics features and claims, synthesised from a real portfolio [1] | 100k rows | [paper](https://doi.org/10.3390/risks9040058) (the authors' UConn data page is offline) |
| **Loss Data Analytics data** | Wisconsin Local Government Property Insurance Fund (LGPIF) and other teaching datasets of the open textbook | Thousands of rows | [Data resources](https://openacttexts.github.io/LDAVer2/DataResources.html) |
| **MEPS** | US Medical Expenditure Panel Survey: health expenditure and utilisation by person | ~30k persons a year | [AHRQ downloads](https://meps.ahrq.gov/data_stats/download_data_files.jsp) |

## General insurance: reserving

| Dataset | What | Scale | Access |
| --- | --- | --- | --- |
| ✅ **CAS Loss Reserving Database** | NAIC Schedule P paid and incurred triangles for six US lines, 1988–1997, **including the realised lower triangle** for back-testing | ~200 companies × 6 lines × 100 cells | [CAS](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) |
| **chainladder samples** | Classic triangles (Taylor & Ashe, Mack, Clark, CAS) ready to load in Python | Single triangles to multi-company | [`chainladder.load_sample`](https://chainladder-python.readthedocs.io/stable/library/sample_data.html) · R: [ChainLadder](https://cran.r-project.org/package=ChainLadder) |
| **Individual claims simulation machine** | Simulates individual claims histories calibrated to a real Swiss portfolio [2] | Any size | [paper](https://doi.org/10.3390/risks6020029) |
| **SynthETIC** | Individual claims simulator with controllable features (notification, settlement, partial payments, inflation) [3] | Any size | [CRAN](https://cran.r-project.org/package=SynthETIC) · [GitHub](https://github.com/agi-lab/SynthETIC) |

## Life, health and mortality

| Dataset | What | Scale | Access |
| --- | --- | --- | --- |
| ✅ **Human Mortality Database (HMD)** | Deaths, exposures and death rates by age, sex and year for 40+ countries | Long histories, ages 0–110+ | [mortality.org](https://www.mortality.org/) (free registration; Switzerland ships in `Data.zip`) |
| **HMD Short-Term Mortality Fluctuations** | Weekly deaths and rates: excess mortality, pandemics | Many countries, weekly | [STMF](https://mortality.org/Data/STMF) |
| **United States Mortality Database** | HMD-style life tables for US states and counties | 50 states | [usa.mortality.org](https://usa.mortality.org/) |
| **Human Life-Table Database** | Official life tables from national statistics offices, including historical tables | Many countries | [lifetable.de](https://www.lifetable.de/) |
| **SOA Mortality and Other Rate Tables** | Thousands of published mortality, lapse and morbidity tables in a common format | Thousands of tables | [mort.soa.org](https://mort.soa.org/) |
| **SOA ILEC experience studies** | US individual life insurance mortality experience by underwriting, duration and face amount | Industry-wide | [Reports and data](https://www.soa.org/research/topics/indiv-mort-exp-study-list/) · [R tools](https://github.com/Society-of-actuaries-research-institute/RILEC) |

## Collections and case studies

- [AI Tools for Actuaries](https://aitools4actuaries.com/) [4]: `Data.zip` with freMTPL2, HMD Switzerland, Swiss accident (SUVA) and text data, plus R and Python notebooks
- [Actuarial Data Science tutorials](https://github.com/actuarial-data-science/Tutorials): SAV tutorials with data and code
- [IAA Actuarial AI case studies](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies): curated case studies with code
- [freMTPL2 on Hugging Face](https://huggingface.co/datasets/mabilton/fremtpl2): one-line `datasets.load_dataset` access

## Adding a dataset

1. Add a download-and-cache function to `src/deep_actuarial_science/pipelines/ingest/__init__.py` (see `fetch_cas`), with its URL in `conf/base/parameters.yml` under `ingest`.
2. Register the raw output in `conf/base/catalog.yml`, then add a `prepare` node with a `scale` (or subset) parameter so experiments can run small.
3. If a published result exists on the dataset, add it to `conf/base/parameters_benchmarks.yml`.

## References
1. So, B., Boucher, J.-P., & Valdez, E. A. (2021). Synthetic dataset generation of driver telematics. *Risks*, 9(4), 58. [doi:10.3390/risks9040058](https://doi.org/10.3390/risks9040058)
2. Gabrielli, A., & Wüthrich, M. V. (2018). An individual claims history simulation machine. *Risks*, 6(2), 29. [doi:10.3390/risks6020029](https://doi.org/10.3390/risks6020029)
3. Avanzi, B., Taylor, G., Wang, M., & Wong, B. (2021). SynthETIC: an individual insurance claim simulator with feature control. *Insurance: Mathematics and Economics*, 100, 296–308. [doi:10.1016/j.insmatheco.2021.06.004](https://doi.org/10.1016/j.insmatheco.2021.06.004)
4. Wüthrich, M. V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., & Scognamiglio, S. (2025). AI tools for actuaries. *SSRN 5162304*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)

---
_Last reviewed: 2026-11-06_
