# Load KEY=value lines from the repo's .env (git-ignored) into this PowerShell session,
# like `set -a; . ./.env` in the bash scripts. Dot-source it: . "$PSScriptRoot/_env.ps1"
$envFile = Join-Path $PSScriptRoot "../.env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"', "'"))
        }
    }
}
