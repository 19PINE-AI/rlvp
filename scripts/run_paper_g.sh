#!/bin/bash
# Stage G: tau2 RLVP training. Waits for D-F driver. vLLM serves the user sim.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }

while pgrep -f run_paper_def >/dev/null; do sleep 120; done
mark "=== STAGE G (tau2 training) ==="

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8011 --gpu-memory-utilization 0.30 \
  --max-model-len 16384 --reasoning-parser qwen3 > results/tau2/vllm_g.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { mark "G vllm FAILED"; exit 1; }

# baseline evals (untrained policy), then train, then trained evals
.venv-tau2/bin/python scripts/tau2_eval.py Qwen/Qwen3-4B tau2_base_nopolicy \
  > results/tau2_eval_base_nopolicy.log 2>&1 && mark "G EVAL base_nopolicy OK" || mark "G EVAL base_nopolicy FAILED"
.venv-tau2/bin/python scripts/tau2_eval.py Qwen/Qwen3-4B tau2_base_policy --with-policy \
  > results/tau2_eval_base_policy.log 2>&1 && mark "G EVAL base_policy OK" || mark "G EVAL base_policy FAILED"

.venv-tau2/bin/python scripts/tau2_train.py 30 > results/tau2_train.log 2>&1 \
  && mark "G TRAIN OK" || mark "G TRAIN FAILED"

if [ -d results/run_tau2/final ]; then
  .venv-tau2/bin/python scripts/tau2_eval.py results/run_tau2/final tau2_rlvp_nopolicy \
    > results/tau2_eval_rlvp_nopolicy.log 2>&1 && mark "G EVAL rlvp_nopolicy OK" || mark "G EVAL rlvp_nopolicy FAILED"
  .venv-tau2/bin/python scripts/tau2_eval.py results/run_tau2/final tau2_rlvp_policy --with-policy \
    > results/tau2_eval_rlvp_policy.log 2>&1 && mark "G EVAL rlvp_policy OK" || mark "G EVAL rlvp_policy FAILED"
fi
mark "=== STAGE G DONE ==="
