#!/bin/bash
# E3: tau2-bench head-to-head (outcome-only GRPO vs RLVP) on real benchmark
# reward, trained WITHOUT the policy doc in the prompt. Standalone (the inline
# flagship attempt failed on a user-sim crash, now fixed in tau2_adapter).
# Waits for the chain campaigns to free the GPU.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec 7>/tmp/rlvp_tau2.lock; flock -n 7 || { echo "tau2 h2h already running"; exit 0; }
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }

while pgrep -f "run_flagship.sh|run_t2.sh" >/dev/null; do sleep 180; done
mark "=== E3: tau2 head-to-head (fixed adapter) ==="

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8011 --gpu-memory-utilization 0.30 \
  --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 > results/tau2/vllm_h2h.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { mark "E3 vllm FAILED"; exit 1; }

# base eval (untrained, no policy prompt) — the headroom
.venv-tau2/bin/python scripts/tau2_eval.py Qwen/Qwen3-4B tau2_base_nopolicy --k 2 \
  > results/tau2_eval_base.log 2>&1 && mark "E3 EVAL base OK" || mark "E3 EVAL base FAILED"

# head-to-head training (no policy doc in prompt), 25 iters each
.venv-tau2/bin/python scripts/tau2_train.py 25 --credit outcome --out run_tau2_outcome \
  > results/tau2_outcome.log 2>&1 && mark "E3 TRAIN outcome OK" || mark "E3 TRAIN outcome FAILED"
.venv-tau2/bin/python scripts/tau2_train.py 25 --credit c3 --out run_tau2_rlvp \
  > results/tau2_rlvp.log 2>&1 && mark "E3 TRAIN rlvp OK" || mark "E3 TRAIN rlvp FAILED"

for v in outcome rlvp; do
  if [ -d "results/run_tau2_${v}/final" ]; then
    .venv-tau2/bin/python scripts/tau2_eval.py "results/run_tau2_${v}/final" "tau2_${v}_nopolicy" --k 2 \
      > "results/tau2_eval_${v}.log" 2>&1 && mark "E3 EVAL ${v} OK" || mark "E3 EVAL ${v} FAILED"
  fi
done
mark "=== E3 tau2 DONE ==="
