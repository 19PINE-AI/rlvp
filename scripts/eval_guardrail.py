"""Arm 4: runtime-guardrail baseline. Same k=8 protocol as eval_checkpoint,
but the environment rejects rule-violating actions instead of recording them.
The agent pays in turns; we measure success, blocked-count, turns, and
whether ordering rules can be satisfied at all by masking.

Usage: python3 scripts/eval_guardrail.py <model> <tag> [--k 8] [--temp 0.7]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import run_episodes, start_episode

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("tag")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--n-tasks", type=int, default=30)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--gen-batch", type=int, default=48)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()

out = {"model": args.model, "guardrail": True, "k": args.k, "temp": args.temp}
for domain in ("fileops", "csops"):
    by_task, eps = {}, []
    for s in range(args.n_tasks):
        for _ in range(args.k):
            e = start_episode(tok, make_env(domain, args.seed0 + s, guardrail=True))
            by_task.setdefault(s, []).append(e)
            eps.append(e)
    run_episodes(model, tok, eps, temperature=args.temp, top_p=0.95, gen_batch=args.gen_batch)
    tasks = list(by_task.values())
    dom = {
        "pass@1": sum(e.env.success for e in eps) / len(eps),
        "pass^k": sum(all(e.env.success for e in g) for g in tasks) / len(tasks),
        "blocked_per_ep": sum(e.env.blocked for e in eps) / len(eps),
        "eps_with_blocks": sum(e.env.blocked > 0 for e in eps) / len(eps),
        "residual_viol": sum(len(e.env.violations) for e in eps),
        "turns_per_ep": sum(e.env.turn for e in eps) / len(eps),
        "calls_per_ep": sum(len(e.env.calls) for e in eps) / len(eps),
        "timeout_rate": sum(1 for e in eps if e.env.turn >= e.env.max_turns) / len(eps),
    }
    out[domain] = dom
    print(domain, json.dumps(dom), flush=True)

(ROOT / f"results/eval_{args.tag}.json").write_text(json.dumps(out, indent=2))
print("wrote", f"results/eval_{args.tag}.json")
