# PowerShell twin of update_mlflow_skills.sh: re-vendor the MLflow agent skills into .claude/skills/.
. "$PSScriptRoot/_skills.ps1"
$repo = Get-Repo "mlflow/skills"
foreach ($s in "mlflow-docs", "mlflow-onboarding", "mlflow-sagemaker") { Copy-Skill $repo $s }
Remove-Item -Recurse -Force $repo
git status --short -- .claude/skills
