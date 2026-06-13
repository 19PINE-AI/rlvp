"""Stage D: clean-episode probability vs horizon length (the decay figure).

Chained FileOps tasks with 1/2/4 stages (~3/6/12 rule-relevant decisions),
evaluated zero-shot (models were trained on single-stage tasks only).

Usage: python3 scripts/eval_horizon.py <model> <tag> [--rules] [--k 4]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs.fileops import make_chain_env
from rlvp.rollout import run_episodes, start_episode

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("tag")
ap.add_argument("--rules", action="store_true")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--n-tasks", type=int, default=30)
ap.add_argument("--temp", type=float, default=0.7)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()

out = {"model": args.model, "rules_in_prompt": args.rules, "k": args.k}
for n_stages in (1, 2, 4):
    eps = []
    for s in range(args.n_tasks):
        for _ in range(args.k):
            eps.append(start_episode(tok, make_chain_env(2000 + s, n_stages), args.rules))
    run_episodes(model, tok, eps, temperature=args.temp, top_p=0.95,
                 gen_batch=32, max_episode_tokens=9000)
    n_calls = sum(len(e.env.calls) for e in eps)
    n_viol = sum(len(e.env.violations) for e in eps)
    per_rule = {}
    for e in eps:
        for _, r in e.env.violations:
            per_rule[r] = per_rule.get(r, 0) + 1
    res = {
        "success": sum(e.env.success for e in eps) / len(eps),
        "clean": sum(not e.env.violations for e in eps) / len(eps),
        "success_and_clean": sum(e.env.success and not e.env.violations for e in eps) / len(eps),
        "viol_per_100_calls": round(100 * n_viol / max(n_calls, 1), 1),
        "calls_per_ep": round(n_calls / len(eps), 1),
        "truncated": sum(e.truncated for e in eps) / len(eps),
        "per_rule": per_rule,
    }
    out[f"stages_{n_stages}"] = res
    print(f"stages={n_stages}", json.dumps(res), flush=True)

(ROOT / f"results/horizon_{args.tag}.json").write_text(json.dumps(out, indent=2))
print("wrote", f"results/horizon_{args.tag}.json")
