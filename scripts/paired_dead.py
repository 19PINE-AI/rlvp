"""E2b: paired dead-iteration experiment. Roll out IDENTICAL batches, then
apply both outcome-only and RLVP credit to each — count which batches yield a
zero-gradient (dead) update under each scheme. Makes the dead-iteration claim
paired rather than across-run.

Usage: python3 scripts/paired_dead.py [model] [n_batches]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import ENVS, make_env
from rlvp.grpo import TrainConfig, build_advantages
from rlvp.rollout import run_episodes, scripted_episode, set_template, start_episode

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
DOMAIN, G, TASKS = "chain4", 8, 8

set_template(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()

cfg_out = TrainConfig(credit="outcome", domains=(DOMAIN,))
cfg_rlvp = TrainConfig(credit="c3", lam=0.25, beta=0.25, mix_scripted=True,
                       script_scalar=False, domains=(DOMAIN,))


def dead(adv_eps):
    """True if no episode carries any nonzero advantage (zero-gradient update)."""
    for e, a_seq, per_turn in adv_eps:
        if abs(a_seq) > 1e-9 or any(abs(v) > 1e-9 for v in per_turn.values()):
            return False
    return True


import random
rng = random.Random(0)
out_dead = rlvp_dead = 0
seeds = list(range(500))
for b in range(N_BATCHES):
    # one batch = TASKS groups of G live episodes (+1 scripted for the RLVP view)
    groups_live, groups_full, eps = [], [], []
    for _ in range(TASKS):
        s = rng.choice(seeds)
        live = [start_episode(tok, make_env(DOMAIN, s)) for _ in range(G)]
        env_s = make_env(DOMAIN, s)
        scr = scripted_episode(tok, env_s, ENVS[DOMAIN].compliant_script(env_s.task))
        groups_live.append(live)
        groups_full.append(live + [scr])
        eps.extend(live)
    run_episodes(model, tok, eps, temperature=1.0, top_p=1.0, gen_batch=32,
                 max_episode_tokens=9000)
    # IDENTICAL rollouts -> two credit schemes
    od = dead(build_advantages(groups_live, cfg_out))
    rd = dead(build_advantages(groups_full, cfg_rlvp))
    out_dead += od
    rlvp_dead += rd
    print(f"batch {b}: outcome_dead={od} rlvp_dead={rd}", flush=True)

res = {"model": MODEL, "n_batches": N_BATCHES, "domain": DOMAIN,
       "outcome_dead_frac": out_dead / N_BATCHES,
       "rlvp_dead_frac": rlvp_dead / N_BATCHES}
print(json.dumps(res, indent=2))
(ROOT / "results/paired_dead.json").write_text(json.dumps(res, indent=2))
