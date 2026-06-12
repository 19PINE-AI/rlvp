#!/bin/bash
# Tier 1.5: the "why RL" discriminator — imperfect (compliant-but-failing)
# demonstrations. BC must clone the failure; RL group advantage filters it out
# while the discharge channel keeps the compliant scaffolding.
# Waits for the main Tier 1-3 driver to finish before touching the GPU.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/tier123_status.log
mark() { echo "$(date '+%H:%M:%S') $1" >> "$S"; }

while pgrep -f run_tier123 >/dev/null; do sleep 60; done
mark "=== TIER 1.5 (imperfect scripts) START ==="

python3 scripts/sft_bc.py --imperfect > results/sftbc_imp.log 2>&1 \
  && mark "TRAIN sftbc_imp OK" || mark "TRAIN sftbc_imp FAILED"
python3 scripts/eval_checkpoint.py results/run_sftbc_imp/final sftbc_imp_norules \
  >> results/evals_tier123.log 2>&1 && mark "EVAL sftbc_imp OK" || mark "EVAL sftbc_imp FAILED"

python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
cfg = TrainConfig(model_name='Qwen/Qwen3-4B', credit='c3', lam=0.25, beta=0.25,
                  mix_scripted=True, imperfect_scripts=True, anneal_at=40,
                  gen_batch=48, out_dir='results/run_c3mix_imp')
train(cfg)
" > results/c3mix_imp.log 2>&1 && mark "TRAIN c3mix_imp OK" || mark "TRAIN c3mix_imp FAILED"
python3 scripts/eval_checkpoint.py results/run_c3mix_imp/final c3mix_imp_norules \
  >> results/evals_tier123.log 2>&1 && mark "EVAL c3mix_imp OK" || mark "EVAL c3mix_imp FAILED"

mark "=== TIER 1.5 DONE ==="
