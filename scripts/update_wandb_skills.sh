#!/usr/bin/env bash
# Re-vendor the W&B agent skill (Apache-2.0, https://github.com/wandb/skills, see
# https://wandb.ai/site/skills/) into .claude/skills/, pinned in git so every learner and CI
# run gets the same skill text. Used with the optional W&B logging (WANDB_PROJECT, see README):
#   wandb-primary   W&B runs, metrics, artifacts, sweeps and reports: query and analyse runs
# wandb/skills also has wandb-autoresearch (W&B Launch queues), wandb-eval-tables,
# weave-instrument and agentlens-tracing-instrument (LLM-agent tracing); add them to `skills`
# if you use those. To install all of them globally instead:
#   npx skills add wandb/skills --agent claude-code --skill '*' --yes --global
# Review the diff before committing: skills are instructions your agent follows.
set -euo pipefail
cd "$(dirname "$0")/.."
skills=(wandb-primary)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL https://codeload.github.com/wandb/skills/tar.gz/refs/heads/main \
  | tar -xz -C "$tmp" --strip-components=1
for skill in "${skills[@]}"; do
  rm -rf ".claude/skills/$skill"
  cp -R "$tmp/skills/$skill" ".claude/skills/$skill"
  cp "$tmp/LICENSE" ".claude/skills/$skill/LICENSE"
done
git status --short -- "${skills[@]/#/.claude/skills/}"
