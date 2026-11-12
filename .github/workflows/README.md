# `.github/workflows/`

```mermaid
flowchart LR
    pr[pull request] --> ci["ci.yml<br/>ruff, pytest, smoke, READMEs, CHANGELOG"]
    pr --> lc["link-check.yml<br/>links, citations, math"]
    main[push to main] --> wiki["wiki-sync.yml"]
    main --> slides["slides.yml"]
    cron[monthly] --> refresh["lesson-refresh.yml"]
```

| Workflow | Runs |
| --- | --- |
| `ci.yml` | tests, smoke run, README and CHANGELOG checks on pull requests and pushes to `main`/`develop` |
| `link-check.yml` | links, citations and math rendering when Markdown changes; weekly |
| `wiki-sync.yml` | rebuilds the wiki on push to `main` |
| `slides.yml` | publishes the Marp slides to Pages on push to `main` |
| `lesson-refresh.yml` | monthly lesson refresh PR |
