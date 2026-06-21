"""Experiment 1 -- train with on-policy self-critique as the dense reward.

Runs ONE variant so arms can be launched/compared separately:
  outcome   -- sparse outcome only (baseline)
  c3        -- RLVP: rule-derived per-turn penalties (the verifiable channel)
  llmcritic -- per-turn penalties from the model's OWN blind self-critique
               (same weights = policy = critic -> on-policy, not distillation)

The rule oracle keeps running under all arms purely as a reward-hacking monitor:
the per-iter log records critic_precision / critic_recall vs the oracle and the
TRUE violation rate, so we can see whether self-critique reward actually lowers
real violations or just games its own signal.

Usage:
  python3 scripts/exp_selfcritic_train.py <outcome|c3|llmcritic> [iters] [domain] [model]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rlvp.grpo import TrainConfig, train

credit = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 24
domain = sys.argv[3] if len(sys.argv) > 3 else "csops"
model = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3-1.7B"

frac = float(os.environ.get("RLVP_MEM_FRAC", "0"))
if frac:
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"GPU memory cap: {frac:.2f} of device", flush=True)

# optional overrides for ablation cells (e.g. penalty-only rules = c2 + beta=0)
beta = float(os.environ.get("RLVP_BETA", "0.5"))
suffix = os.environ.get("RLVP_OUT_SUFFIX", "")
seed = int(os.environ.get("RLVP_SEED", "7"))                  # multi-seed: vary data_seed
frozen = os.environ.get("RLVP_FROZEN_CRITIC", "0") == "1"     # frozen vs live critic

cfg = TrainConfig(
    model_name=model,
    credit=credit,
    critic_mode="blind",
    domains=(domain,),
    iters=iters,
    tasks_per_iter=8,
    group_size=8,
    lr=1e-5,
    lora_r=16,                 # fit alongside any co-resident run
    beta=beta,
    data_seed=seed,
    frozen_critic=frozen,
    inner_epochs=2,
    micro_token_budget=1280,   # keep the per-microbatch logits spike small enough
    max_episode_tokens=1300,   # to fit a coexistence-safe (~15GB) GPU slice
    gen_batch=4,
    temp=1.0,
    eval_every=max(iters, 1),  # eval once at the end (saves memory mid-run)
    eval_tasks=16,
    eval_k=2,
    out_dir=f"results/exp_sc_train_{credit}{suffix}_{domain}",
)
print(f"training credit={credit} domain={domain} model={model} iters={iters}", flush=True)
train(cfg)
print("DONE", flush=True)
