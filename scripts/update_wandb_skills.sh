#!/usr/bin/env bash
# Re-vendor the W&B agent skills (Apache-2.0, https://github.com/wandb/skills, see
# https://wandb.ai/site/skills/) into .claude/skills/, pinned in git so every learner and CI
# run gets the same skill text. Used with the optional W&B logging (WANDB_PROJECT, see README):
#   wandb-primary                 W&B runs, metrics, artifacts, sweeps and reports
#   wandb-autoresearch            experiment loops on W&B Launch queues
#   wandb-eval-tables             turn W&B table artifacts into evaluation tables
#   weave-instrument              Weave tracing for LLM apps
#   agentlens-tracing-instrument  Agent Lens tracing for LLM agents
# Review the diff before committing: skills are instructions your agent follows.
set -euo pipefail
cd "$(dirname "$0")/.."
skills=(wandb-primary wandb-autoresearch wandb-eval-tables weave-instrument agentlens-tracing-instrument)
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
