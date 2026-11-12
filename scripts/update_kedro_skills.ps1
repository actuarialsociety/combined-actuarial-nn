# PowerShell twin of update_kedro_skills.sh: re-vendor the Kedro skills pinned to one commit,
# record the source, then let `kedro skills update` refresh the packaged ones.
param([string]$Ref = "main")
. "$PSScriptRoot/_skills.ps1"
$sha = (Invoke-RestMethod "https://api.github.com/repos/kedro-org/kedro-skills/commits/$Ref").sha
$repo = Get-Repo "kedro-org/kedro-skills" $sha
foreach ($s in "parameters-and-config", "explain-project", "kedro-build-demo") {
    Copy-SkillFile $repo $s
    "Vendored from https://github.com/kedro-org/kedro-skills/tree/$sha/skills/$s (Apache-2.0).`nNot yet in a kedro-skills PyPI release; re-vendor with scripts/update_kedro_skills.sh.`n" |
        Set-Content -NoNewline ".claude/skills/$s/SOURCE"
}
Remove-Item -Recurse -Force $repo
$env:DO_NOT_TRACK = "1"
kedro skills update
git --no-pager diff --stat -- .claude/skills .agents AGENTS.md
