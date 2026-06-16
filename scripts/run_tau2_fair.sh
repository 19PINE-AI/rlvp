#!/bin/bash
set -uo pipefail; cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 7>/tmp/rlvp_tau2.lock; flock -n 7 || exit 0
S=results/paper_status.log; mark(){ echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --port 8011 \
  --gpu-memory-utilization 0.14 --max-model-len 8192 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 > results/tau2/vllm_fair.log 2>&1 &
V=$!; trap 'kill $V 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { mark "FAIR vllm FAILED"; exit 1; }
mark "=== tau2 FAIR: RLVP + anneal (vs outcome 0.5) ==="
.venv-tau2/bin/python scripts/tau2_train.py 25 --credit c3 --out run_tau2_rlvp_anneal --anneal 12 \
  > results/tau2_rlvp_anneal.log 2>&1 && mark "FAIR TRAIN rlvp_anneal OK" || mark "FAIR TRAIN rlvp_anneal FAILED"
[ -d results/run_tau2_rlvp_anneal/final ] && .venv-tau2/bin/python scripts/tau2_eval.py results/run_tau2_rlvp_anneal/final tau2_rlvp_anneal_nopolicy --k 4 > results/tau2_eval_rlvp_anneal.log 2>&1 && mark "FAIR EVAL rlvp_anneal OK"
mark "=== tau2 FAIR DONE ==="
