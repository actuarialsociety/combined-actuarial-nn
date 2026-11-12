#!/usr/bin/env bash
# Full run: every model, every dataset, every published benchmark, on Apple silicon (MPS)
# with Weights & Biases logging (and MLflow, as always).
#
#   1. reference setup (conf/base): frequency, mortality and reserving, each model
#      reconciled PASS/FAIL to its paper's table. Frequency fits 15 seeds per network: hours.
#   2. every dataset (conf/scale_up): mortality for both sexes; reserving on all six
#      Schedule P lines and every company that passes the data checks. Frequency is not
#      repeated: freMTPL2 is already used in full by the reference setup.
#
# Each pipeline is one W&B run in the group $WANDB_RUN_GROUP. Reports are archived under
# data/09_full_run/<group>/ (pass 2 overwrites data/08_reporting).
#
#   scripts/full_run.sh                  # both passes
#   scripts/full_run.sh reference        # pass 1 only
#   scripts/full_run.sh scale_up         # pass 2 only
#   DAS_DEVICE=cpu scripts/full_run.sh   # same run on the CPU, to compare
#   REF_ENV=smoke UP_ENV=smoke WANDB_MODE=offline scripts/full_run.sh   # 2-minute check
#
# Needs: pip install -e ".[wandb]", and WANDB_API_KEY in .env (or `wandb login`). Without a
# key it logs offline; upload later with `wandb sync wandb/offline-run-*`.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
KEDRO=$(dirname "$PY")/kedro; [ -x "$KEDRO" ] || KEDRO=kedro

export DAS_DEVICE=${DAS_DEVICE:-mps}
REF_ENV=${REF_ENV:-local} UP_ENV=${UP_ENV:-scale_up}
export WANDB_PROJECT=${WANDB_PROJECT:-deep-actuarial-science}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-full-$DAS_DEVICE-$(date +%Y%m%d-%H%M)}
export DO_NOT_TRACK=1 MLFLOW_DISABLE_AGENT_HINT=1

"$PY" -c "import wandb" 2>/dev/null || { echo "W&B not installed: uv pip install -e '.[wandb]'"; exit 1; }
if [ "$DAS_DEVICE" = mps ]; then
  "$PY" -c "import torch, sys; sys.exit(not torch.backends.mps.is_available())" \
    || { echo "No MPS on this machine: run with DAS_DEVICE=cpu"; exit 1; }
fi
if [ -z "${WANDB_MODE:-}" ] && ! "$PY" -c "import wandb, sys; sys.exit(not wandb.api.api_key)"; then
  echo "No W&B key (WANDB_API_KEY or wandb login): logging offline; upload with wandb sync"
  export WANDB_MODE=offline
fi

OUT=data/09_full_run/$WANDB_RUN_GROUP
mkdir -p "$OUT"
echo "device=$DAS_DEVICE  W&B project=$WANDB_PROJECT group=$WANDB_RUN_GROUP mode=${WANDB_MODE:-online}"
echo "logs and reports: $OUT"

# Keep the Mac awake while it runs.
STAY_AWAKE=(); command -v caffeinate >/dev/null && STAY_AWAKE=(caffeinate -i)

run() {  # env pipeline
  local start=$SECONDS
  echo "== $1 / $2  $(date +%H:%M:%S)"
  "${STAY_AWAKE[@]}" "$KEDRO" run --env "$1" --pipeline "$2" >"$OUT/$1-$2.log" 2>&1 \
    || { echo "FAILED: see $OUT/$1-$2.log"; tail -20 "$OUT/$1-$2.log"; exit 1; }
  echo "   done in $(((SECONDS - start) / 60)) min"
}

passes=("${@:-reference scale_up}")
for pass in ${passes[*]}; do
  case $pass in
    reference)
      for p in frequency mortality reserving; do run "$REF_ENV" "$p"; done
      cp -R data/08_reporting "$OUT/reference" && find "$OUT/reference" -name .gitkeep -delete
      "$PY" - "$OUT/reference" <<'EOF'
import glob, sys
import pandas as pd
rec = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{sys.argv[1]}/*/*/reconciliation.csv")))
rec.to_csv(f"{sys.argv[1]}/reconciliation_all.csv", index=False)
print(rec[["model", "metric", "expected", "actual", "tolerance", "status"]].to_string(index=False))
print(rec.status.value_counts().to_string())
EOF
      "$PY" scripts/freeze_reference.py   # reference/, lesson figures and numbers
      ;;
    scale_up)
      for p in mortality reserving; do run "$UP_ENV" "$p"; done
      cp -R data/08_reporting "$OUT/scale_up" && find "$OUT/scale_up" -name .gitkeep -delete
      ;;
    *) echo "unknown pass: $pass (reference or scale_up)"; exit 1 ;;
  esac
done
echo "W&B group: $WANDB_RUN_GROUP (project $WANDB_PROJECT); MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db"
