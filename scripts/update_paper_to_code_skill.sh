#!/usr/bin/env bash
# Re-vendor the paper-to-code skill (Apache-2.0, LeCoder, from
# https://github.com/aryateja2106/nested-learning/tree/main/.claude/skills/paper-to-code;
# listed at https://awesomeskill.ai/skill/nested-learning-paper-to-code) into .claude/skills/.
# It is kept verbatim; how it applies in this repo (inside the Kedro pipelines, not a new
# project) is in AGENTS.md §3 and the build-research-benchmark-wiki skill.
# Review the diff before committing: skills are instructions your agent follows.
# PowerShell version: scripts/update_paper_to_code_skill.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL https://codeload.github.com/aryateja2106/nested-learning/tar.gz/refs/heads/main \
  | tar -xz -C "$tmp" --strip-components=1
rm -rf .claude/skills/paper-to-code
cp -R "$tmp/.claude/skills/paper-to-code" .claude/skills/paper-to-code
cp "$tmp/LICENSE" .claude/skills/paper-to-code/LICENSE
git status --short -- .claude/skills/paper-to-code
