# PowerShell twin of update_paper_to_code_skill.sh: re-vendor the paper-to-code skill verbatim.
. "$PSScriptRoot/_skills.ps1"
$repo = Get-Repo "aryateja2106/nested-learning"
Copy-Skill $repo ".claude/skills/paper-to-code"
Remove-Item -Recurse -Force $repo
git status --short -- .claude/skills/paper-to-code
