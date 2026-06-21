#!/bin/bash
# Master: run the four remaining experiments sequentially, each gated on a GPU
# window so miniF2F is never starved. Order: reliable HF training first, the
# heavier tau2 (server + venv) stack last so it can't block the rest.
set -u
cd "$(dirname "$0")/.."
log () { echo "[$(date +%m-%d_%H:%M:%S)] MASTER: $*"; }

log "=== Driver A: multi-seed + frozen-critic (csops, 1.7B) ==="
bash scripts/drive_multiseed.sh >> results/drive_A.log 2>&1

log "=== Driver B: Exp1 matrix at 4B ==="
bash scripts/drive_exp1_4b.sh >> results/drive_B.log 2>&1

log "=== Driver C+D: tau2 cell-C offline x3 + training ==="
bash scripts/drive_tau2_all.sh >> results/drive_tau2.log 2>&1

log "=== ALL FOUR DONE ==="
