# PowerShell twin of update_azure_skills.sh: re-vendor the Azure agent skills into .claude/skills/.
. "$PSScriptRoot/_skills.ps1"
$sdk = Get-Repo "microsoft/skills"
$azure = Get-Repo "microsoft/azure-skills"
Copy-Skill $sdk ".github/plugins/azure-sdk-python/skills/azure-ai-ml-py"
Copy-Skill $azure "skills/azure-quotas"
Copy-Skill $azure ".github/plugins/azure-cost/skills/cost-estimation"
Remove-Item -Recurse -Force $sdk, $azure
git status --short -- .claude/skills
