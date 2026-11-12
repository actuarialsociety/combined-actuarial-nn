#!/usr/bin/env bash
# Sign in to every service the project uses, skipping any that is already signed in.
#   scripts/login.sh                  # github wandb (the defaults)
#   scripts/login.sh github wandb aws azure
# Keys never go in a file the repo tracks: W&B reads WANDB_API_KEY from .env (git-ignored)
# or its own saved login. PowerShell version: scripts/login.ps1.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
WANDB=$(dirname "$PY")/wandb; [ -x "$WANDB" ] || WANDB=wandb
services=("${@:-github wandb}")
for s in ${services[*]}; do
  case $s in
    github) gh auth status >/dev/null 2>&1 && echo "github: signed in" || gh auth login ;;
    wandb)  # a key in the environment, or a saved login that the W&B API accepts
      if "$PY" -c "import wandb, sys; sys.exit(not wandb.login(anonymous='never', verify=True))" 2>/dev/null
      then echo "wandb: signed in"; else "$WANDB" login --relogin; fi ;;
    aws)    aws sts get-caller-identity >/dev/null 2>&1 && echo "aws: signed in" || aws login ;;
    azure)  az account show >/dev/null 2>&1 && echo "azure: signed in" || az login ;;
    *) echo "unknown service: $s (github, wandb, aws, azure)"; exit 1 ;;
  esac
done
