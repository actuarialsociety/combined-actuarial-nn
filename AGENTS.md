# Agent Guidelines

`combined-actuarial-nn` is a set of short, **video-driven, code-centric** lessons
that teach actuaries Combined Actuarial Neural Networks (CANNs): GLMs, neural
networks, and nesting a GLM inside a network for pricing and reserving. Each
lesson is one Markdown file in the repo root, listed in [README.md](./README.md).
General tooling (Python, VS Code, Git, Claude Code) is covered in
[ds-training](https://github.com/actuarialsociety/ds-training); link to those
lessons as prerequisites with absolute GitHub URLs instead of rebuilding them. Coding agents (Claude Code, Copilot,
Cursor) maintain the lessons: they find good new material, check that links
still work and keep each lesson short.

## Project rules (read first)

### 1. A lesson = research-referenced theory + videos + a repo to code along in

- Lessons are **research-focused**. A short `# Theory` section states the
  models in LaTeX (`$...$` inline, `$$...$$` on their own lines; GitHub and
  the wiki render both) and cites sources as numbered references `[n]`.
- Cite **academic sources**: peer-reviewed journals (ASTIN Bulletin,
  Scandinavian Actuarial Journal, Annals of Actuarial Science, European
  Actuarial Journal, Insurance: Mathematics and Economics, Risks), textbooks,
  or SSRN/arXiv preprints by recognised authors. Every lesson ends with a
  `# References` list in author–year style, with a `https://doi.org/...` link
  wherever a DOI exists. Verify each DOI's authors, year, volume and pages with
  `curl -s https://api.crossref.org/works/<DOI>` before you write it. Never
  cite from memory, and never describe a paper's findings beyond its abstract.
- Keep the prose tight: each theory subsection is a few sentences plus the key
  equation. Prerequisites, Objectives and Follow-along steps stay short.
- Every new lesson has a **Follow-along** section that points at a real GitHub
  repo or notebook (Colab link where possible). It gives 3 to 6 concrete steps
  the learner types and runs.
- Up to 6 videos per lesson, when good ones exist (CANN talks are scarce; do
  not pad with weak videos). Choose official channels first (Anthropic, Hugging
  Face, Google, NVIDIA, Microsoft/VS Code, the library's own maintainers),
  then actuarial bodies and researchers (SAV/Actuarial Data Science, IFoA,
  SOA, ASSA, Wüthrich, Richman), then trusted educators (StatQuest and similar). Pick recent, short videos over
  long or old ones.
- Follow [`.claude/skills/build-lesson/template.md`](.claude/skills/build-lesson/template.md)
  for the layout and the video block format. The existing lessons use the same
  format, so keep new and edited lessons consistent with them.

### 2. Never invent a link

- Find material with Exa (the `exa-search` and `exa-contents` skills, which
  need `EXA_API_KEY`), or with web search if Exa is not available.
- Check **every** YouTube ID with oEmbed before you write it into a lesson, and
  use the exact title that oEmbed returns:
  `curl -s "https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v=<ID>"`
- Check every GitHub repo with `gh repo view <owner/name> --json isArchived,pushedAt`.
  Do not use an archived repo, or one with no pushes in the last 2 years,
  unless it is the canonical source.
- Run `python3 scripts/check_links.py <files>` before you finish. It must
  report **0 broken**.

### 3. Repo conventions

- File names are `Topic - Level.md` with spaces, e.g. `Claude Code - Basic.md`.
  In README links, encode spaces as `%20` and `&` as `%26`.
- Every new lesson gets a line in the matching README section:
  `- [Name](./File%20Name.md): one-line description`. Mark optional lessons
  `(Optional)`.
- Each new or refreshed lesson ends with `_Last reviewed: YYYY-MM-DD_`. The
  refresh skill uses that date to look for newer material.
- Do not delete a learner-facing lesson without asking. Move superseded videos
  to an `## Older content` section at the end of the lesson, as
  `Devcontainers in VS Code.md` does.

- The [GitHub wiki](https://github.com/actuarialsociety/combined-actuarial-nn/wiki) is
  **generated** from the root `.md` files. Edit lessons in the repo, never in
  the wiki. `AGENTS.md` and `CLAUDE.md` are not published.

### 4. Workflows

| Task | How |
| --- | --- |
| Add a lesson | `/build-lesson <topic> [level]` ([skill](.claude/skills/build-lesson/SKILL.md)) |
| Refresh lessons | `/refresh-lessons [file]` ([skill](.claude/skills/refresh-lessons/SKILL.md)) |
| Check links | `python3 scripts/check_links.py` (CI runs it weekly and on every PR) |
| Update Exa skills | `scripts/update_exa_skills.sh` |
| Publish wiki | Automatic on push to `main` ([wiki-sync.yml](.github/workflows/wiki-sync.yml)); preview locally with `python3 scripts/build_wiki.py <dir>` |

### 5. Secrets

- `EXA_API_KEY` lives in your shell or in `.env`, which git ignores. See
  `.env.example`. Never write a key into a lesson, skill or commit.
- CI needs the `ANTHROPIC_API_KEY` and `EXA_API_KEY` repo secrets only for the
  scheduled refresh workflow.
