# PowerShell twin of update_wandb_skills.sh: re-vendor the W&B agent skills into .claude/skills/.
. "$PSScriptRoot/_skills.ps1"
$repo = Get-Repo "wandb/skills"
foreach ($s in "wandb-primary", "wandb-autoresearch", "wandb-eval-tables", "weave-instrument", "agentlens-tracing-instrument") { Copy-Skill $repo "skills/$s" }
Remove-Item -Recurse -Force $repo
git status --short -- .claude/skills
