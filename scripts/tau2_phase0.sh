#!/bin/bash
# Arm 7: tau2-bench Phase-0 measurement, fully local.
# Uses the py3.12 venv (.venv-tau2), TAU2_DATA_DIR from the cloned repo,
# vLLM-served Qwen3-8B as both agent and user simulator.
set -uo pipefail
cd /home/ubuntu/rlvp
mkdir -p results/tau2
export TAU2_DATA_DIR=/tmp/tau2-bench/data

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8011 --gpu-memory-utilization 0.45 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 > results/tau2/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 90); do
  curl -s localhost:8011/v1/models >/dev/null 2>&1 && break
  sleep 5
done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { echo "vllm never came up"; exit 1; }

export OPENAI_API_KEY=local
export OPENAI_API_BASE=http://localhost:8011/v1
export OPENAI_BASE_URL=http://localhost:8011/v1

.venv-tau2/bin/tau2 run \
  --domain airline \
  --agent-llm openai/Qwen/Qwen3-8B \
  --user-llm openai/Qwen/Qwen3-8B \
  --num-tasks 20 --num-trials 1 --max-concurrency 4 \
  --save-to /home/ubuntu/rlvp/results/tau2/airline_qwen8b \
  2>&1 | tail -25

# results land in TAU2_DATA_DIR/../simulations or local data dir; collect any json
find /tmp/tau2-bench/data -name "*.json" -newer results/tau2/vllm.log -exec cp {} results/tau2/ \; 2>/dev/null
find ~/.tau2 -name "*.json" -newer results/tau2/vllm.log -exec cp {} results/tau2/ \; 2>/dev/null || true
echo "tau2 run finished"
