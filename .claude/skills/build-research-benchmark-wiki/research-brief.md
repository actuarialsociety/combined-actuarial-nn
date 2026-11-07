# Research brief for a benchmark agent

Fill the angle brackets and give this to one agent per domain (run them in parallel).
The agent must not edit repo files; it returns a report.

---

You are researching published benchmarks for a research-paper benchmark toolkit
(repo: <path>). Do NOT edit any repo files. Return a research report only.

## Tools
- Exa via curl. Load the key without printing it: `set -a && . ./.env && set +a`. Never
  echo or write EXA_API_KEY. Request formats: `.claude/skills/exa-search/SKILL.md` and
  `.claude/skills/exa-contents/SKILL.md`. Use `contents` with `text` to read result tables
  (arXiv, SSRN, author PDFs and open-access versions are fine).
- Verify every DOI with `curl -s https://api.crossref.org/works/<DOI>` (title, authors,
  year must match). Without a Crossref DOI, give the URL and say so.

## Our setup
- Data: <dataset, version, cleaning, split sizes>
- Models and settings: <architecture, preprocessing, training, seeds>
- Metric and its exact normalisation: <e.g. Poisson deviance x 100, divided by n or by total exposure>
- Our current values: <model: learn / test>
- Current benchmark source: <paper table | notebook | package | none>

## What to find
1. For each model: papers whose **published tables** report the metric on the same or a
   clearly comparable setup. Per candidate: citation, verified DOI, table/figure, quoted
   row, units, exact setup, verdict LIKE-FOR-LIKE / COMPARABLE (how it differs) /
   NOT COMPARABLE. Key candidates: <papers>.
2. For each recommended benchmark: the implementation as stated (data cleaning,
   preprocessing, architecture, parameter counts, loss, optimiser, learning rate, batch,
   epochs, early stopping and patience, validation scheme, seeds/ensembling) and the
   reference code (repo, notebook, listing). Quote the line. "Not stated" when absent.
   Then list the concrete differences from our setup that would need to change.
3. Whether the metric scale differs from ours, and the exact conversion factor if so.
4. Newer work (last ~3 years) reporting benchmarkable numbers on the same data, best first.
5. Citation checks: does each current reference's abstract support what it is cited for?

## Rules
- Never invent numbers, tables, company lists or DOIs. Only report what you saw in text
  Exa returned; quote it. If a table was not accessible, say so.
- Save fetched texts in the scratchpad so the main session can verify quotes.
- Be concise: candidate benchmarks per model (verdict first), implementation details and
  differences, newer papers, citation checks, then a one-paragraph recommendation of
  which paper/table each model should reconcile to.
