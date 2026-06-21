#!/bin/bash
# Fixed-trajectory cross-model probe: roll out ONE policy (1.7B, violation-rich),
# then judge those EXACT trajectories with 1.7B / 4B / 8B critics. Capped so it
# coexists with the running Exp1 matrix / a miniF2F run. Gates on free GPU.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLVP_MEM_FRAC=${RLVP_MEM_FRAC:-0.35}
POLICY=${POLICY:-Qwen/Qwen3-1.7B}
PSLUG=$(echo "$POLICY" | sed 's#.*/##; s#\.#_#g')
TRAJ="results/exp_selfcritic/traj/${PSLUG}.json"
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
log () { echo "[$(date +%H:%M:%S)] $*"; }

while [ "$(free_gb)" -lt 30 ]; do sleep 60; done
if [ ! -f "$TRAJ" ]; then
  log "rollout policy $POLICY"
  python3 scripts/sc_rollout.py "$POLICY" 24 || { log "rollout FAILED"; exit 1; }
fi
for CRITIC in Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  CS=$(echo "$CRITIC" | sed 's#.*/##; s#\.#_#g')
  [ -f "results/exp_selfcritic/probe/${PSLUG}__${CS}.json" ] && { log "skip critic $CRITIC (done)"; continue; }
  while [ "$(free_gb)" -lt 30 ]; do sleep 60; done
  log "critique with $CRITIC"
  python3 scripts/sc_critique.py "$TRAJ" "$CRITIC" || log "critique $CRITIC FAILED"
done
python3 scripts/sc_probe_aggregate.py > results/sc_probe_summary.txt 2>&1 || true
log "PROBE DONE"
