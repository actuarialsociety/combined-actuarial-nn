# combined-actuarial-nn
> *Code centric lessons about Combined Actuarial Neural Networks (CANNs)*
> * *Every lesson is **video-driven** with a **repo to code along in**. Lessons are kept current by Claude Code (see [Maintaining these lessons](#maintaining-these-lessons))*
> * *New to Python, VS Code or Git? Start with [ds-training](https://github.com/actuarialsociety/ds-training)*

## Lessons
_No lessons yet. Add one with `/build-lesson <topic> [level]`._

## Maintaining these lessons
Browse the lessons as a [wiki](https://github.com/actuarialsociety/combined-actuarial-nn/wiki). It is regenerated from this repo on every merge to `main`.

This repo is set up for [Claude Code](https://code.claude.com/docs/en/overview) to find and maintain lessons, using [Exa](https://exa.ai/docs/get-started/agent-skills/overview) agent skills for search. Rules are in [AGENTS.md](./AGENTS.md).
1. `cp .env.example .env` and add an [Exa API key](https://dashboard.exa.ai/api-keys), or `export EXA_API_KEY=...`
2. In Claude Code:
   - `/build-lesson <topic> [level]`: find videos and a follow-along repo, verify every link, write the lesson and add it here
   - `/refresh-lessons [file|all]`: fix dead links and swap in newer or better material
3. `python3 scripts/check_links.py` checks every link. It runs in CI on each PR and weekly, and [`lesson-refresh.yml`](./.github/workflows/lesson-refresh.yml) opens a refresh PR monthly
