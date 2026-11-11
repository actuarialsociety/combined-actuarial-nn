# Sign in to every service the project uses, skipping any that is already signed in.
#   scripts/login.ps1                       # github wandb (the defaults)
#   scripts/login.ps1 github wandb aws azure
# Keys never go in a file the repo tracks: W&B reads WANDB_API_KEY from .env (git-ignored)
# or its own saved login. Bash version: scripts/login.sh.
param([string[]]$Services = @("github", "wandb"))
Set-Location (Join-Path $PSScriptRoot "..")
. "$PSScriptRoot/_env.ps1"   # loads .env into this session
$py = if (Test-Path .venv/Scripts/python.exe) { ".venv/Scripts/python.exe" }
      elseif (Test-Path .venv/bin/python) { ".venv/bin/python" } else { "python" }
foreach ($s in $Services) {
    switch ($s) {
        "github" { gh auth status *> $null; if ($LASTEXITCODE) { gh auth login } else { "github: signed in" } }
        "wandb" {
            & $py -c "import wandb, sys; sys.exit(not wandb.login(anonymous='never', verify=True))" *> $null
            if ($LASTEXITCODE) { & $py -m wandb login --relogin } else { "wandb: signed in" }
        }
        "aws"   { aws sts get-caller-identity *> $null; if ($LASTEXITCODE) { aws login } else { "aws: signed in" } }
        "azure" { az account show *> $null; if ($LASTEXITCODE) { az login } else { "azure: signed in" } }
        default { throw "unknown service: $s (github, wandb, aws, azure)" }
    }
}
