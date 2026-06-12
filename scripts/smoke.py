"""Smoke test: template sanity + a few live episodes with Qwen3-1.7B."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import ASSISTANT_PREFIX, episode_stats, run_episodes, start_episode

MODEL = "Qwen/Qwen3-1.7B"

tok = AutoTokenizer.from_pretrained(MODEL)

# --- template sanity: our manual format must equal apply_chat_template ---
env = make_env("fileops", 0)
msgs = [{"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": env.initial_user_msg()}]
ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ours = (f"<|im_start|>system\n{env.system_prompt()}<|im_end|>\n"
        f"<|im_start|>user\n{env.initial_user_msg()}<|im_end|>\n" + ASSISTANT_PREFIX)
assert ref == ours, f"template mismatch:\nREF:\n{ref!r}\nOURS:\n{ours!r}"
print("template sanity: OK")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()

for domain in ("fileops", "csops"):
    eps = [start_episode(tok, make_env(domain, 1000 + s)) for s in range(8)]
    run_episodes(model, tok, eps, temperature=0.7, top_p=0.95)
    print(f"\n=== {domain} ===")
    print(episode_stats(eps))
    e = eps[0]
    print("--- transcript (episode 0) ---")
    print(tok.decode(e.ids))
    print("violations:", e.env.violations, "| success:", e.env.success)
