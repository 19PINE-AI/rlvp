#!/bin/bash
# T2 ceiling (chain6) + fairness control. RECIPE CORRECTED per the full E4
# ablation: clean RLVP (penalty+discharge+anneal, NO mixing, NO step_cost) is
# the proven best on chain4 (final 1.0). step_cost HURTS (sc3 final 0.84) ->
# dropped. chain6 (~2% base succ) is the decisive ceiling + mixing test.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec 8>/tmp/rlvp_t2.lock; flock -n 8 || { echo "t2 already running"; exit 0; }
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
tr() { local name=$1 kwargs=$2
  [ -d "results/run_${name}/final" ] && { mark "SKIP ${name} (done)"; return; }
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }

while pgrep -f "run_flagship.sh" >/dev/null; do sleep 120; done

C4="domains=('chain4',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"
C6="domains=('chain6',), tasks_per_iter=8, gen_batch=24, max_episode_tokens=15000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"
CLEAN="credit='c3', lam=0.25, beta=0.25, anneal_at=40"   # proven recipe, no mixing
MIX="credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=40"

# fairness control on chain4: outcome + demos, no process channel
mark "=== T2-P1: fairness control (chain4) ==="
tr ctrl_outmix "credit='outcome', mix_scripted=True, script_scalar=False, ${C4}"

# decisive ceiling + mixing test on the hard chain6 regime
mark "=== T2-P2: chain6 ceiling + clean-vs-mixed ==="
tr t2_outcome    "credit='outcome', ${C6}"
tr t2_dapo       "credit='outcome', dynamic_sampling=True, ${C6}"
tr t2_rlvp_clean "${CLEAN}, ${C6}"
tr t2_rlvp_mix   "${MIX}, ${C6}"
tr t2_outmix     "credit='outcome', mix_scripted=True, script_scalar=False, ${C6}"
mark "=== T2 DONE ==="
