"""E-A (granularity) + E-B (sparsity) on the synthetic chain: a CONTROLLED test of
the central claim -- RLVP helps iff a verifiable potential Phi strictly finer than the
terminal outcome exists. Phi = #satisfied stages (verifiable). Two knobs:
  granularity (coarse=outcome / mid=1 milestone / fine=every -dPhi)  -- fineness of Phi
  n_stages                                                            -- outcome sparsity
All arms use credit=c3 (potential-based shaping, no penalties); coarse emits no
discharge so c3 reduces to outcome. Prediction: dead-iter elimination + success scale
with granularity, and the benefit appears as n_stages grows (outcome blinds).

Usage: chain_potential_exp.py <granularity> <n_stages> [iters] [out] [seed]
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlvp.grpo import TrainConfig, build_advantages, update_policy   # noqa: E402
from rlvp.rollout import run_episodes, set_template, start_episode    # noqa: E402
from rlvp.envs.fileops import make_chain_potential_env                # noqa: E402

GRAN = sys.argv[1] if len(sys.argv) > 1 else "fine"
NSTAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUT = sys.argv[4] if len(sys.argv) > 4 else f"chainpot_{GRAN}_n{NSTAGES}"
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
LR = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-5
OPT = sys.argv[7] if len(sys.argv) > 7 else "adamw"   # adamw | muon (bounded updates)
MODEL = sys.argv[8] if len(sys.argv) > 8 else "Qwen/Qwen3-4B"
G, TASKS_PER_ITER = 8, 4
OUTD = ROOT / "results" / f"run_{OUT}"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if OPT == "muon":
        from rlvp.muon import Muon
        opt = Muon(trainable, lr=LR, momentum=0.95)
    else:
        opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"optimizer={OPT} lr={LR}", flush=True)
    # all arms: potential-based shaping (no penalties). coarse emits no discharge -> outcome.
    cfg = TrainConfig(credit="c3", beta=0.5, lam=0.0, max_episode_tokens=3500)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)
    print(f"granularity={GRAN} n_stages={NSTAGES} iters={ITERS} seed={SEED}", flush=True)
    log = open(OUTD / "train_log.jsonl", "a")
    import random, math
    rng = random.Random(SEED)
    EVAL_EVERY, EVAL_K, EVAL_SEED0 = 5, 24, 900001   # fixed held-out task set (Fix B)

    def held_out_eval():
        """Success on a FIXED set of held-out task instances -> low-variance
        efficiency curve (the per-iter training succ on 4 random tasks is too noisy)."""
        model.config.use_cache = True
        ev = [start_episode(tok, make_chain_potential_env(EVAL_SEED0 + i, NSTAGES, granularity=GRAN))
              for i in range(EVAL_K)]
        run_episodes(model, tok, ev, temperature=1.0, top_p=1.0, gen_batch=32,
                     max_new_tokens=160, max_episode_tokens=cfg.max_episode_tokens)
        model.config.use_cache = False
        return round(sum(e.env.success for e in ev) / len(ev), 3)

    collapse = explode = 0     # divergence counters (Fix A)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        groups, eps = [], []
        for t in range(TASKS_PER_ITER):
            s = rng.randint(0, 10 ** 6)
            grp = [start_episode(tok, make_chain_potential_env(s, NSTAGES, granularity=GRAN))
                   for _ in range(G)]
            groups.append(grp)
            eps += grp
        model.config.use_cache = True
        run_episodes(model, tok, eps, temperature=1.0, top_p=1.0, gen_batch=32,
                     max_new_tokens=160, max_episode_tokens=cfg.max_episode_tokens)
        succ = sum(e.env.success for e in eps) / len(eps)
        disc = sum(len(e.env.discharges) for e in eps) / len(eps)
        # dead-group fraction: a group whose scalar reward (outcome + beta*discharge -
        # lam*viol) has ZERO within-group spread produces no GRPO gradient. This is the
        # sample-efficiency mechanism: the fine potential keeps groups alive where the
        # binary outcome is uniformly zero.
        n_dead = 0
        for grp in groups:
            rw = [e.env.outcome_reward() + cfg.beta * len(e.env.discharges)
                  - cfg.lam * len(e.env.violations) for e in grp]
            if max(rw) - min(rw) < 1e-9:
                n_dead += 1
        dead_frac = round(n_dead / len(groups), 3)
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        ent = float(m.get("entropy", 0.0) or 0.0)
        gnorm = float(m.get("grad_norm", 0.0) or 0.0)
        rec = {"iter": it, "succ": round(succ, 3), "disch_per_ep": round(disc, 2),
               "dead_frac": dead_frac, "n_eps": len(eps), **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        if it == 1 or it % EVAL_EVERY == 0 or it == ITERS:
            rec["eval_succ"] = held_out_eval()
        # divergence guards: entropy->0 mode collapse, or grad explosion
        collapse = collapse + 1 if ent < 1e-4 else 0
        explode = explode + 1 if (gnorm > 50 or not math.isfinite(gnorm)) else 0
        rec["collapse"], rec["explode"] = collapse, explode
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if collapse >= 3 or explode >= 2:
            reason = "entropy_collapse" if collapse >= 3 else "grad_explosion"
            log.write(json.dumps({"iter": it, "DIVERGED": True, "reason": reason}) + "\n")
            log.flush()
            print(f"DIVERGED at iter {it}: {reason} (optimizer artifact, aborting)", flush=True)
            break
    print("CHAINPOT DONE", flush=True)


if __name__ == "__main__":
    main()
