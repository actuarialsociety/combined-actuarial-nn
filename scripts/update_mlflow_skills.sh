#!/usr/bin/env bash
# Re-vendor the MLflow agent skills (Apache-2.0, https://github.com/mlflow/skills) into
# .claude/skills/, pinned in git. The ones this project uses (classic ML tracking, MLflow on
# SageMaker); mlflow/skills also has GenAI tracing and evaluation skills (mlflow-trace-*,
# mlflow-evaluation, mlflow-agent*): add them to `skills` if you need them.
#   mlflow-docs        look up MLflow documentation from the official docs
#   mlflow-onboarding  set up experiment tracking
#   mlflow-sagemaker   connect to SageMaker Managed MLflow
# Review the diff before committing: skills are instructions your agent follows.
# PowerShell version: scripts/update_mlflow_skills.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
skills=(mlflow-docs mlflow-onboarding mlflow-sagemaker)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL https://codeload.github.com/mlflow/skills/tar.gz/refs/heads/main \
  | tar -xz -C "$tmp" --strip-components=1
for skill in "${skills[@]}"; do
  rm -rf ".claude/skills/$skill"
  cp -R "$tmp/$skill" ".claude/skills/$skill"
  cp "$tmp/LICENSE" ".claude/skills/$skill/LICENSE"
done
git status --short -- "${skills[@]/#/.claude/skills/}"
