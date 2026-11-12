#!/usr/bin/env bash
# Point git at the repo's hooks in .githooks/ (pre-push: README and CHANGELOG checks).
# Run once per clone. PowerShell version: scripts/install_hooks.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true
echo "git hooks enabled: $(ls .githooks | grep -v README | tr '\n' ' ')"
