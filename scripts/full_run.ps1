# PowerShell twin of full_run.sh: every model, dataset and benchmark, logged to W&B.
#   scripts/full_run.ps1                      # reference pass (then freeze), then scale_up
#   scripts/full_run.ps1 reference            # one pass
# Settings as environment variables: DAS_DEVICE (mps, cuda or cpu; default cuda on Windows),
# REF_ENV, UP_ENV, WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_MODE. Logs and a copy of every report
# go to data/09_full_run/<group>/.
param([string[]]$Passes = @("reference", "scale_up"))
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. "$PSScriptRoot/_env.ps1"

$py = if (Test-Path .venv/Scripts/python.exe) { ".venv/Scripts/python.exe" }
      elseif (Test-Path .venv/bin/python) { ".venv/bin/python" } else { "python" }
function Default([string]$name, [string]$value) {   # keep a variable the caller already set
    if (-not [Environment]::GetEnvironmentVariable($name)) { Set-Item "env:$name" $value }
}
Default DAS_DEVICE ($(if ($IsMacOS) { "mps" } else { "cuda" }))
Default REF_ENV "local"; Default UP_ENV "scale_up"
Default WANDB_PROJECT "deep-actuarial-science"
Default WANDB_RUN_GROUP "full-$env:DAS_DEVICE-$(Get-Date -Format yyyyMMdd-HHmm)"
$env:DO_NOT_TRACK = "1"; $env:MLFLOW_DISABLE_AGENT_HINT = "1"

& $py -c "import wandb" 2>$null
if ($LASTEXITCODE) { throw "W&B not installed: uv pip install -e '.[wandb]'" }
if (-not $env:WANDB_MODE) {   # no key: log offline, upload later with `wandb sync`
    & $py -c "import wandb, sys; sys.exit(not wandb.api.api_key)" 2>$null
    if ($LASTEXITCODE) { "No W&B key: logging offline; upload with wandb sync"; $env:WANDB_MODE = "offline" }
}
$out = "data/09_full_run/$env:WANDB_RUN_GROUP"
New-Item -ItemType Directory -Force $out | Out-Null
"device=$env:DAS_DEVICE  W&B project=$env:WANDB_PROJECT group=$env:WANDB_RUN_GROUP"
"logs and reports: $out"

function Invoke-Pipeline([string]$envName, [string]$pipeline) {
    # One `kedro run`, its log in $out; stop the whole run on the first failure.
    $start = Get-Date
    "== $envName / $pipeline  $(Get-Date -Format HH:mm:ss)"
    & $py -m kedro run --env $envName --pipeline $pipeline *> "$out/$envName-$pipeline.log"
    if ($LASTEXITCODE) { Get-Content "$out/$envName-$pipeline.log" -Tail 20; throw "FAILED: see $out/$envName-$pipeline.log" }
    "   done in $([int]((Get-Date) - $start).TotalMinutes) min"
}
function Save-Reports([string]$dest) {
    # Copy data/08_reporting into the run archive, without the .gitkeep placeholders.
    Copy-Item -Recurse data/08_reporting $dest
    Get-ChildItem -Recurse -Force $dest -Filter .gitkeep | Remove-Item
}

foreach ($pass in $Passes) {
    switch ($pass) {
        "reference" {
            foreach ($p in "frequency", "mortality", "reserving") { Invoke-Pipeline $env:REF_ENV $p }
            Save-Reports "$out/reference"
            & $py -c @"
import glob, sys
import pandas as pd
d = sys.argv[1]
rec = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f'{d}/*/*/reconciliation.csv')))
rec.to_csv(f'{d}/reconciliation_all.csv', index=False)
print(rec[['model', 'metric', 'expected', 'actual', 'tolerance', 'status']].to_string(index=False))
print(rec.status.value_counts().to_string())
"@ "$out/reference"
            & $py scripts/freeze_reference.py   # reference/, lesson figures and numbers
        }
        "scale_up" {
            foreach ($p in "mortality", "reserving") { Invoke-Pipeline $env:UP_ENV $p }
            Save-Reports "$out/scale_up"
        }
        default { throw "unknown pass: $pass (reference or scale_up)" }
    }
}
"W&B group: $env:WANDB_RUN_GROUP (project $env:WANDB_PROJECT); MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db"
