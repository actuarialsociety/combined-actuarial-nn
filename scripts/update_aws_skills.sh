#!/usr/bin/env bash
# Re-vendor AWS agent skills (Apache-2.0, https://github.com/aws/agent-toolkit-for-aws,
# part of the Agent Toolkit for AWS) into .claude/skills/, pinned in git so every learner
# and CI run gets the same skill text. Used when running the notebooks on SageMaker:
#   aws-ai-ml                        SageMaker training/processing jobs, managed MLflow
#   signing-in-to-aws                AWS credentials for the CLI/SDK (`aws login`)
#   aws-billing-and-cost-management  instance/GPU pricing and cost checks
# For the AWS MCP server as well, install the full toolkit in Claude Code:
#   /plugin install aws-core@claude-plugins-official
# Review the diff before committing: skills are instructions your agent follows.
set -euo pipefail
cd "$(dirname "$0")/.."
skills=(aws-ai-ml signing-in-to-aws aws-billing-and-cost-management)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL https://codeload.github.com/aws/agent-toolkit-for-aws/tar.gz/refs/heads/main \
  | tar -xz -C "$tmp" --strip-components=1
for skill in "${skills[@]}"; do
  rm -rf ".claude/skills/$skill"
  cp -R "$tmp/skills/core-skills/$skill" ".claude/skills/$skill"
  cp "$tmp/LICENSE" ".claude/skills/$skill/LICENSE"
done
git status --short -- "${skills[@]/#/.claude/skills/}"
