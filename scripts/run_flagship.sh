#!/bin/bash
# FLAGSHIP (re-aimed thesis): process rewards make OUTCOME learning faster and
# its ceiling higher than outcome-only GRPO, on hard sparse-outcome tasks.
# 1) calibrate chain difficulty, 2) train 4 variants on the hard chain domain,
# 3) seeds for the headline pair, 4) tau2 head-to-head (outcome vs RLVP).
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }

# wait until nothing else trains
while pgrep -f "run_paper_abc|run_paper_def|scripts/train.py" >/dev/null; do sleep 60; done

# calibration already done (results/horizon_calib.json):
#   chain2 succ .29 -> only ~6% all-fail groups (GRPO rarely blind: too easy)
#   chain4 succ .083 -> ~50% all-fail groups (GRPO blind half the time: ideal)
#   chain6 succ .021 -> ~85% all-fail (very hard, base barely solves any)
# Pick the regime where the all-fail-group mechanism is strongest while live
# rollouts still produce some success: smallest N with base success <= 0.15.
if [ ! -f results/horizon_calib.json ]; then
  mark "=== FLAGSHIP: calibration ==="
  python3 scripts/eval_horizon.py Qwen/Qwen3-4B calib --stages 2,4,6,8 --k 2 --n-tasks 24 \
    >> results/evals_paper.log 2>&1 && mark "CALIB OK" || mark "CALIB FAILED"
fi
N=$(python3 -c "
import json
d = json.load(open('results/horizon_calib.json'))
for n in (2, 4, 6, 8):
    if d[f'stages_{n}']['success'] <= 0.15:
        print(n); break
else:
    print(4)")
mark "FLAGSHIP domain=chain${N} (all-fail-rich regime)"

FLAG="domains=('chain${N}',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"

tr() { local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }

mark "=== FLAGSHIP: main 4-way (chain${N}) ==="
tr flag_outcome  "credit='outcome', ${FLAG}"
tr flag_rlvp     "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=40, ${FLAG}"
tr flag_gigpo    "credit='gigpo', lam=0.25, beta=0.25, ${FLAG}"
tr flag_steptool "credit='steptool', ${FLAG}"

mark "=== FLAGSHIP chains main DONE ==="

# ---- tau2 head-to-head: outcome-only GRPO vs RLVP on real benchmark reward ----
mark "=== FLAGSHIP: tau2 head-to-head ==="
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8011 --gpu-memory-utilization 0.30 \
  --max-model-len 16384 --reasoning-parser qwen3 > results/tau2/vllm_g.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { mark "G vllm FAILED"; exit 1; }

.venv-tau2/bin/python scripts/tau2_eval.py Qwen/Qwen3-4B tau2_base_nopolicy \
  > results/tau2_eval_base_nopolicy.log 2>&1 && mark "T2 EVAL base OK" || mark "T2 EVAL base FAILED"
.venv-tau2/bin/python scripts/tau2_train.py 30 --credit outcome --out run_tau2_outcome \
  > results/tau2_outcome.log 2>&1 && mark "T2 TRAIN outcome OK" || mark "T2 TRAIN outcome FAILED"
.venv-tau2/bin/python scripts/tau2_train.py 30 --credit c3 --out run_tau2_rlvp \
  > results/tau2_rlvp.log 2>&1 && mark "T2 TRAIN rlvp OK" || mark "T2 TRAIN rlvp FAILED"
for v in outcome rlvp; do
  if [ -d "results/run_tau2_${v}/final" ]; then
    .venv-tau2/bin/python scripts/tau2_eval.py "results/run_tau2_${v}/final" "tau2_${v}_nopolicy" \
      > "results/tau2_eval_${v}.log" 2>&1 && mark "T2 EVAL ${v} OK" || mark "T2 EVAL ${v} FAILED"
  fi
done
mark "=== FLAGSHIP DONE ==="

mark "=== E2 seeds for headline pair ==="
for seed in 11 12; do
  tr "flag_outcome_s${seed}" "credit='outcome', data_seed=${seed}, ${FLAG}"
  tr "flag_rlvp_s${seed}"    "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=40, data_seed=${seed}, ${FLAG}"
done

mark "=== E4 component attribution ==="
tr abl_nomix      "credit='c3', lam=0.25, beta=0.25, anneal_at=40, ${FLAG}"
tr abl_nodisch    "credit='c3', lam=0.25, beta=0.0, mix_scripted=True, script_scalar=False, anneal_at=40, ${FLAG}"
tr abl_noanneal   "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, ${FLAG}"
tr abl_scalaronly "credit='c1', lam=0.25, beta=0.25, mix_scripted=True, ${FLAG}"
tr abl_naivemix   "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, anneal_at=40, ${FLAG}"
mark "=== FLAGSHIP ALL DONE ==="
