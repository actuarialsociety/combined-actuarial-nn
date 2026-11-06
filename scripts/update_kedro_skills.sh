#!/usr/bin/env bash
# Update the Kedro agent skills (Apache-2.0, https://github.com/kedro-org/kedro-skills).
# - catalog-config is installed by the `kedro-skills` package (managed files, AGENTS.md block):
#     kedro skills update
# - parameters-and-config, explain-project and kedro-build-demo are not in a PyPI release yet,
#   so they are vendored from GitHub at a pinned ref. Once released, switch them to
#   `kedro skills install <id> --ide claude` and delete them from this script.
# Review the diff before committing: skills are instructions your agent follows.
set -euo pipefail
cd "$(dirname "$0")/.."
ref="${1:-main}"
sha=$(curl -fsSL "https://api.github.com/repos/kedro-org/kedro-skills/commits/$ref" | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
base="https://raw.githubusercontent.com/kedro-org/kedro-skills/$sha"
for skill in parameters-and-config explain-project kedro-build-demo; do
  mkdir -p ".claude/skills/$skill"
  curl -fsSL "$base/skills/$skill/SKILL.md" -o ".claude/skills/$skill/SKILL.md"
  curl -fsSL "$base/LICENSE" -o ".claude/skills/$skill/LICENSE"
  printf 'Vendored from https://github.com/kedro-org/kedro-skills/tree/%s/skills/%s (Apache-2.0).\nNot yet in a kedro-skills PyPI release; re-vendor with scripts/update_kedro_skills.sh.\n' "$sha" "$skill" > ".claude/skills/$skill/SOURCE"
done
DO_NOT_TRACK=1 kedro skills update
git --no-pager diff --stat -- .claude/skills .agents AGENTS.md
