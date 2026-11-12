# Point git at the repo's hooks in .githooks/ (pre-push: README and CHANGELOG checks).
# Run once per clone. Bash version: scripts/install_hooks.sh.
# Git for Windows runs the hook with its bundled bash, so the hook itself stays a bash script.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
git config core.hooksPath .githooks
$hooks = Get-ChildItem .githooks | Where-Object Name -ne "README.md" | ForEach-Object Name
Write-Host "git hooks enabled: $($hooks -join ' ')"
