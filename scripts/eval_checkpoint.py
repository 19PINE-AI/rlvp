"""Full RLVP evaluation of a model (base or trained checkpoint).

Usage: python3 scripts/eval_checkpoint.py <model_path_or_name> <tag> [--rules] [--k 8] [--temp 0.7]
Writes results/eval_<tag>.json

Metrics per domain over EVAL tasks (seeds 1000..1000+N-1), k rollouts each:
  pass@1            mean per-rollout success
  pass^k            fraction of tasks where ALL k rollouts succeed
  clean@1           mean per-rollout zero-violation rate
  clean^k           fraction of tasks where ALL k rollouts are violation-free
  perfect^k         all k rollouts succeed AND are clean (the RLVP target)
  viol_per_100      violations per 100 tool calls, plus per-rule breakdown
  tactic_diversity  mean over tasks of (#unique tool-call name sequences)/k
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
from rlvp.rollout import run_episodes, set_template, start_episode

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("tag")
ap.add_argument("--rules", action="store_true", help="include rules in system prompt")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--n-tasks", type=int, default=30)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--gen-batch", type=int, default=48)
args = ap.parse_args()

set_template(args.model)
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.unk_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()

out = {"model": args.model, "rules_in_prompt": args.rules, "k": args.k,
       "temp": args.temp, "n_tasks": args.n_tasks}
for domain in ("fileops", "csops"):
    by_task = {}
    eps = []
    for s in range(args.n_tasks):
        for _ in range(args.k):
            e = start_episode(tok, make_env(domain, args.seed0 + s), args.rules)
            by_task.setdefault(s, []).append(e)
            eps.append(e)
    run_episodes(model, tok, eps, temperature=args.temp, top_p=0.95, gen_batch=args.gen_batch)

    n_calls = sum(len(e.env.calls) for e in eps)
    n_viol = sum(len(e.env.violations) for e in eps)
    per_rule = {}
    for e in eps:
        for _, r in e.env.violations:
            per_rule[r] = per_rule.get(r, 0) + 1
    task_vals = list(by_task.values())
    div = []
    for grp in task_vals:
        seqs = {tuple(c.name for c in e.env.calls) for e in grp}
        div.append(len(seqs) / len(grp))
    dom = {
        "pass@1": sum(e.env.success for e in eps) / len(eps),
        "pass^k": sum(all(e.env.success for e in g) for g in task_vals) / len(task_vals),
        "clean@1": sum(not e.env.violations for e in eps) / len(eps),
        "clean^k": sum(all(not e.env.violations for e in g) for g in task_vals) / len(task_vals),
        "perfect^k": sum(all(e.env.success and not e.env.violations for e in g)
                         for g in task_vals) / len(task_vals),
        "viol_per_100_calls": 100.0 * n_viol / max(n_calls, 1),
        "per_rule": per_rule,
        "tactic_diversity": sum(div) / len(div),
        "format_errors_per_ep": sum(e.env.format_errors for e in eps) / len(eps),
        "calls_per_ep": n_calls / len(eps),
    }
    out[domain] = dom
    print(domain, json.dumps(dom), flush=True)

path = ROOT / f"results/eval_{args.tag}.json"
path.write_text(json.dumps(out, indent=2))
print("wrote", path)
