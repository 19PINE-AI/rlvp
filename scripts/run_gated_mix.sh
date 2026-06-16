#!/bin/bash
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 4>/tmp/rlvp_gatedmix.lock; flock -n 4 || exit 0
S=results/paper_status.log; mark(){ echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
tr(){ local n=$1 k=$2; [ -d "results/run_${n}/final" ] && { mark "SKIP $n"; return; }
  python3 -c "
import sys; sys.path.insert(0,'.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${n}', ${k}))
" > "results/${n}.log" 2>&1 && mark "TRAIN $n OK" || mark "TRAIN $n FAILED"; }
while pgrep -f "run_gated.sh|run_autorule.sh|run_t2.sh" >/dev/null; do sleep 120; done
mark "=== GATED-MIX: does mixing break the exploration wall? ==="
G="domains=('gated',), tasks_per_iter=8, group_size=8, gen_batch=48, max_episode_tokens=3500, eval_tasks=24, eval_k=2, iters=80, eval_every=8"
tr gated_rlvp_mix "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=60, ${G}"
mark "=== GATED-MIX DONE ==="
