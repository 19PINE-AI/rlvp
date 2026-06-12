"""Phase 0: base-model violation rates and success rates (no training).

Grid: model x domain x rules-in-prompt. Writes results/phase0.json.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import episode_stats, run_episodes, start_episode

MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
N_TASKS = 30
K = 2  # rollouts per task
EVAL_SEED0 = 1000
TEMP = 0.7

results = {}
for model_name in MODELS:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for domain in ("fileops", "csops"):
        for include_rules in (False, True):
            t0 = time.time()
            eps = []
            for s in range(N_TASKS):
                for _ in range(K):
                    eps.append(start_episode(tok, make_env(domain, EVAL_SEED0 + s), include_rules))
            run_episodes(model, tok, eps, temperature=TEMP, top_p=0.95, gen_batch=60)
            st = episode_stats(eps)
            st["wall_s"] = round(time.time() - t0, 1)
            key = f"{model_name.split('/')[-1]}|{domain}|rules={'on' if include_rules else 'off'}"
            results[key] = st
            print(key, json.dumps(st), flush=True)
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results/phase0.json").write_text(json.dumps(results, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
print("DONE")
