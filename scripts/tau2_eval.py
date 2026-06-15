"""Stage G eval: tau2 airline eval tasks, k sims each, with/without the
policy document in the agent prompt.

Usage: .venv-tau2/bin/python scripts/tau2_eval.py <model> <tag> [--with-policy] [--k 2]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TAU2_DATA_DIR", "/tmp/tau2-bench/data")
os.environ.setdefault("OPENAI_API_KEY", "local")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8011/v1")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8011/v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.rollout import set_template
from rlvp.tau2_adapter import GenServer, run_one_sim

USER_LLM = 'openai/Qwen/Qwen3-4B'

MODEL = sys.argv[1]
TAG = sys.argv[2]
WITH_POLICY = "--with-policy" in sys.argv
K = 2
for i, a in enumerate(sys.argv):
    if a == "--k":
        K = int(sys.argv[i + 1])

from tau2.domains.airline.environment import get_tasks

set_template("Qwen/Qwen3-4B")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()
gen_srv = GenServer(model, tok, temperature=0.7, max_batch=12)

eval_tasks = get_tasks()[:10]
eps = []
with ThreadPoolExecutor(max_workers=10) as ex:
    futs = [ex.submit(run_one_sim, t, gen_srv.generate, tok, WITH_POLICY, USER_LLM)
            for t in eval_tasks for _ in range(K)]
    for f in futs:
        e = f.result()
        if e is not None:
            eps.append(e)
gen_srv.stop()

n = len(eps)
viol = sum(len(e.env.violations) for e in eps)
calls = sum(e.n_turns for e in eps)
out = {
    "model": MODEL, "with_policy": WITH_POLICY, "n_sims": n,
    "mean_reward": sum(e.env.outcome_reward() for e in eps) / max(n, 1),
    "success": sum(e.env.success for e in eps) / max(n, 1),
    "viol_per_ep": viol / max(n, 1),
    "clean_eps": sum(1 for e in eps if not e.env.violations) / max(n, 1),
    "per_rule": {},
}
for e in eps:
    for _, r in e.env.violations:
        out["per_rule"][r] = out["per_rule"].get(r, 0) + 1
print(json.dumps(out, indent=2))
(ROOT / f"results/tau2_eval_{TAG}.json").write_text(json.dumps(out, indent=2))
