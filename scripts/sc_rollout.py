"""Stage 1 of the fixed-trajectory cross-model probe: roll out ONE policy and
serialize its trajectories, so any-size critic can later judge the EXACT same
trajectories. This removes the confound in the naive scale sweep (each model
produced different trajectories, so per-rule recall wasn't apples-to-apples).

Usage: python3 scripts/sc_rollout.py [policy_model] [n_tasks_per_domain]
Writes results/exp_selfcritic/traj/<policy_slug>.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import run_episodes, set_template, start_episode
from rlvp.self_critic import episode_to_record

POLICY = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DOMAINS = ("fileops", "csops", "gated")
ROLL_TEMP, GEN_BATCH, MAX_EP_TOK = 1.0, 6, 1800

frac = float(os.environ.get("RLVP_MEM_FRAC", "0"))
if frac > 0:
    torch.cuda.set_per_process_memory_fraction(frac, 0)

slug = POLICY.split("/")[-1].replace(".", "_")
out = Path("results/exp_selfcritic/traj"); out.mkdir(parents=True, exist_ok=True)


def main():
    set_template(POLICY)
    tok = AutoTokenizer.from_pretrained(POLICY)
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    print(f"loading policy {POLICY} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(POLICY, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    records = []
    for domain in DOMAINS:
        eps = [start_episode(tok, make_env(domain, 2000 + s), include_rules=False)
               for s in range(N)]
        run_episodes(model, tok, eps, temperature=ROLL_TEMP, top_p=1.0,
                     gen_batch=GEN_BATCH, max_episode_tokens=MAX_EP_TOK)
        for e in eps:
            r = episode_to_record(e, tok)
            r["domain_name"] = domain
            records.append(r)
        nv = sum(len(e.turn_violations) for e in eps)
        ns = sum(e.env.success for e in eps)
        print(f"  {domain}: success={ns}/{N}, violating_turns={nv}", flush=True)

    path = out / f"{slug}.json"
    path.write_text(json.dumps({"policy": POLICY, "n": N, "records": records}))
    print(f"saved {len(records)} trajectories -> {path}", flush=True)


if __name__ == "__main__":
    main()
