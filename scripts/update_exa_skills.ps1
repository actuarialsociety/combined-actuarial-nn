# PowerShell twin of update_exa_skills.sh: re-vendor the Exa search skills into .claude/skills/.
. "$PSScriptRoot/_skills.ps1"
$repo = Get-Repo "exa-labs/agent-skills"
foreach ($s in "exa-search", "exa-contents") { Copy-SkillFile $repo $s }
Remove-Item -Recurse -Force $repo
git --no-pager diff --stat -- .claude/skills/exa-search .claude/skills/exa-contents
