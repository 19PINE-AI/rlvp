"""Calibration: is the gated task non-saturating for the base model?
Rolls out base Qwen3-4B on gated, reports success + how often it discovers the
gate (read /acl -> request_access -> write). If base success is low AND the
gate is rarely discovered, the task is a valid ceiling testbed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import run_episodes, set_template, start_episode

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
RULES = "--rules" in sys.argv
set_template(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()

eps = []
for s in range(40):
    for _ in range(2):
        eps.append(start_episode(tok, make_env("gated", 1000 + s), RULES))
run_episodes(model, tok, eps, temperature=0.7, top_p=0.95, gen_batch=48)

n = len(eps)
gate_found = sum(any(c.name == "request_access" for c in e.env.calls) for e in eps)
acl_read = sum(e.env.acl_read for e in eps)
out = {
    "model": MODEL, "rules_in_prompt": RULES, "n": n,
    "success": sum(e.env.success for e in eps) / n,
    "read_acl_rate": acl_read / n,
    "tried_request_access_rate": gate_found / n,
    "viol_per_ep": sum(len(e.env.violations) for e in eps) / n,
}
print(json.dumps(out, indent=2))
(ROOT / f"results/calib_gated{'_rules' if RULES else ''}.json").write_text(json.dumps(out, indent=2))
