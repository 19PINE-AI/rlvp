"""Stage E1: best-of-n inference baseline — the strongest practical
alternative to training. Sample n episodes per task with the rule checker as
a DEPLOYMENT-TIME selector (compliance + completion are observable at deploy
time; task success is not). Deploy the first clean finished episode, else the
fewest-violations one. Report deployed-episode metrics + compute cost.

Usage: python3 scripts/eval_bestofn.py <model> <tag> [--n 4] [--rules]
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
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--rules", action="store_true")
ap.add_argument("--n-tasks", type=int, default=30)
ap.add_argument("--k", type=int, default=8, help="independent best-of-n deployments per task")
ap.add_argument("--temp", type=float, default=0.7)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()

out = {"model": args.model, "n": args.n, "rules_in_prompt": args.rules, "k": args.k}
for domain in ("fileops", "csops"):
    groups, eps = [], []
    for s in range(args.n_tasks):
        for _ in range(args.k):
            g = [start_episode(tok, make_env(domain, 1000 + s), args.rules)
                 for _ in range(args.n)]
            groups.append((s, g))
            eps.extend(g)
    run_episodes(model, tok, eps, temperature=args.temp, top_p=0.95, gen_batch=48)
    deployed, by_task = [], {}
    for s, g in groups:
        clean_done = [e for e in g if not e.env.violations and e.env.success is not None and e.env.done]
        clean = [e for e in clean_done if not e.env.violations]
        pick = (clean[0] if clean else min(g, key=lambda e: len(e.env.violations)))
        deployed.append(pick)
        by_task.setdefault(s, []).append(pick)
    tasks = list(by_task.values())
    res = {
        "pass@1_deployed": sum(e.env.success for e in deployed) / len(deployed),
        "clean_deployed": sum(not e.env.violations for e in deployed) / len(deployed),
        "perfect^k": sum(all(e.env.success and not e.env.violations for e in g)
                         for g in tasks) / len(tasks),
        "episodes_generated_per_deploy": args.n,
        "calls_generated_per_deploy": sum(len(e.env.calls) for e in eps) / len(deployed),
        "calls_in_deployed_ep": sum(len(e.env.calls) for e in deployed) / len(deployed),
    }
    out[domain] = res
    print(domain, json.dumps(res), flush=True)

(ROOT / f"results/eval_{args.tag}.json").write_text(json.dumps(out, indent=2))
print("wrote", f"results/eval_{args.tag}.json")
