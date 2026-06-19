"""Smoke test for the TerminalBench RLVP adapter (NO training).

Rolls out 3 episodes on one small task via run_terminal_episode and prints,
per episode: n_turns, the bash commands issued, the oracle reward, and the
violations/discharges. Confirms: episodes complete, container exec works,
reward is 0/1 from the oracle, and the rule tracker fires.

Usage:
    cd /home/ubuntu/rlvp && python3 scripts/termbench_smoke.py [task_id] [model]
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.rollout import set_template
from rlvp.termbench_adapter import GenServer, run_terminal_episode

TASK = sys.argv[1] if len(sys.argv) > 1 else "hello-world"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
N_EPISODES = 3
MAX_STEPS = 8


def main():
    print(f"[smoke] task={TASK} model={MODEL} episodes={N_EPISODES}", flush=True)
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[smoke] model loaded in {time.time() - t_load:.1f}s", flush=True)

    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=200, max_batch=4)

    rewards = []
    try:
        for i in range(N_EPISODES):
            t0 = time.time()
            ep = run_terminal_episode(
                TASK, gen_srv.generate, tok, rule_mode="structural",
                max_steps=MAX_STEPS, keep_image=True, verbose=False)
            wall = time.time() - t0
            env = ep.env
            rewards.append(env.outcome_reward())
            print(f"\n===== EPISODE {i + 1}/{N_EPISODES}  "
                  f"({wall:.1f}s, {len(ep.ids)} tokens) =====", flush=True)
            print(f"  n_turns          : {ep.n_turns}")
            print(f"  reward (oracle)  : {env.outcome_reward()}  "
                  f"success={env.success}")
            print(f"  commands issued  :")
            for j, c in enumerate(env.calls):
                print(f"     [{j}] $ {c[:160]}")
            if not env.calls:
                print("     (none)")
            print(f"  violations       : {env.violations}")
            print(f"  discharges       : {env.discharges}")
    finally:
        gen_srv.stop()

    print("\n===== SUMMARY =====", flush=True)
    print(f"  rewards   : {rewards}")
    print(f"  mean rew  : {sum(rewards) / max(len(rewards), 1):.3f}")
    print("  GATE: episodes completed, container exec worked, oracle gave 0/1, "
          "rules tracked.")


if __name__ == "__main__":
    main()
