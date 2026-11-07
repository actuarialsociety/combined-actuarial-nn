#!/usr/bin/env bash
# Re-vendor Microsoft's official Azure agent skills (MIT) into .claude/skills/, pinned in
# git so every learner and CI run gets the same skill text. Used when running the
# notebooks on Azure Machine Learning (the Azure counterparts of update_aws_skills.sh):
#   azure-ai-ml-py   Azure ML SDK v2: workspaces, compute, jobs, pipelines (microsoft/skills)
#   azure-quotas     GPU / vCPU quota checks and increases          (microsoft/azure-skills)
#   cost-estimation  price compute and forecast spend               (microsoft/azure-skills)
# Review the diff before committing: skills are instructions your agent follows.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() {  # repo -> $tmp/<repo name>
  mkdir -p "$tmp/${1#*/}"
  curl -fsSL "https://codeload.github.com/$1/tar.gz/refs/heads/main" \
    | tar -xz -C "$tmp/${1#*/}" --strip-components=1
}
fetch microsoft/skills
fetch microsoft/azure-skills
vendor() {  # repo-name path/to/skill
  local name; name=$(basename "$2")
  rm -rf ".claude/skills/$name"
  cp -R "$tmp/$1/$2" ".claude/skills/$name"
  cp "$tmp/$1/LICENSE" ".claude/skills/$name/LICENSE"
  echo "vendored $name"
}
vendor skills .github/plugins/azure-sdk-python/skills/azure-ai-ml-py
vendor azure-skills skills/azure-quotas
vendor azure-skills .github/plugins/azure-cost/skills/cost-estimation
git status --short -- .claude/skills/azure-ai-ml-py .claude/skills/azure-quotas .claude/skills/cost-estimation
