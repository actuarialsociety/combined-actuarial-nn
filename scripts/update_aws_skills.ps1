# PowerShell twin of update_aws_skills.sh: re-vendor the AWS agent skills into .claude/skills/.
. "$PSScriptRoot/_skills.ps1"
$repo = Get-Repo "aws/agent-toolkit-for-aws"
foreach ($s in "aws-ai-ml", "signing-in-to-aws", "aws-billing-and-cost-management") {
    Copy-Skill $repo "skills/core-skills/$s"
}
Remove-Item -Recurse -Force $repo
git status --short -- .claude/skills
