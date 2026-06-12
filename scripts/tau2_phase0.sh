#!/bin/bash
# Arm 7: tau2-bench Phase-0 measurement with a fully local stack.
# Installs tau2-bench, serves Qwen3-8B via vLLM (OpenAI-compatible), runs the
# airline domain with local agent+user-sim, saves trajectories for post-hoc
# rule-compliance analysis. Everything guarded — failure must not kill the driver.
set -uo pipefail
cd /home/ubuntu/rlvp
mkdir -p results/tau2

pip install --user --quiet "git+https://github.com/sierra-research/tau2-bench.git" || {
  echo "tau2 install failed"; exit 1; }

# serve Qwen3-8B (agent + user simulator share one endpoint)
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8011 --gpu-memory-utilization 0.45 \
  --max-model-len 16384 --disable-log-requests > results/tau2/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 60); do
  curl -s localhost:8011/v1/models >/dev/null 2>&1 && break
  sleep 5
done
curl -s localhost:8011/v1/models >/dev/null 2>&1 || { echo "vllm never came up"; exit 1; }

export OPENAI_API_KEY=local
export OPENAI_API_BASE=http://localhost:8011/v1
export OPENAI_BASE_URL=http://localhost:8011/v1

python3 -m tau2 run \
  --domain airline \
  --agent-llm openai/Qwen/Qwen3-8B \
  --user-llm openai/Qwen/Qwen3-8B \
  --num-trials 1 --max-concurrency 4 \
  --save-to results/tau2/airline_qwen8b \
  2>&1 | tail -20

echo "tau2 run finished; trajectories under results/tau2/"
