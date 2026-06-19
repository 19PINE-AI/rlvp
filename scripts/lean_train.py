"""RLVP training on Lean theorem proving (mirrors scripts/tau2_train.py).

Policy = Qwen3-4B + LoRA (r=32), our HF policy with exact token bookkeeping.
Environment = the Lean kernel via the per-tactic oracle (leanprove/lean_repl.py).
Reward = 1.0 iff the proof reaches `done`. Process signal = LeanRuleTracker
(errored_tactic penalty, goal_progress discharge, no_progress penalty).

Credit: c3 (process terms folded into the group-centered scalar advantage) by
default — the two-channel c2/c3 split from grpo.build_advantages is reused
unchanged.

Usage:
  cd /home/ubuntu/rlvp && python3 scripts/lean_train.py [iters] \
      [--credit c3] [--rule-mode structural] [--anneal N] [--out run_lean]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Self-contained elan toolchain on PATH for the REPL (as in lean_repl.py).
ELAN_HOME = ROOT / "leanprove" / ".elan"
os.environ["ELAN_HOME"] = str(ELAN_HOME)
os.environ["PATH"] = f"{ELAN_HOME / 'bin'}:{os.environ.get('PATH', '')}"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.lean_adapter import run_lean_episode
from rlvp.tau2_adapter import GenServer

sys.path.insert(0, str(ROOT / "leanprove"))
from theorems_synth import load_synth_theorems  # noqa: E402
from lean_repl import LeanREPL  # noqa: E402

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
CREDIT = "c3"
ANNEAL = 0
RULE_MODE = "structural"
OUT_NAME = "run_lean"
SEED = 7
for _i, _a in enumerate(sys.argv):
    if _a == "--credit":
        CREDIT = sys.argv[_i + 1]
    if _a == "--anneal":
        ANNEAL = int(sys.argv[_i + 1])
    if _a == "--rule-mode":
        RULE_MODE = sys.argv[_i + 1]
    if _a == "--out":
        OUT_NAME = sys.argv[_i + 1]
    if _a == "--seed":
        SEED = int(sys.argv[_i + 1])

POLICY_MODEL = "Qwen/Qwen3-4B"
G = 6                       # group size
TASKS_PER_ITER = 4          # theorems per iteration
MAX_STEPS = 10
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=ANNEAL,
                  inner_epochs=2, lr=2e-5, micro_token_budget=1024,
                  clip_eps=0.2, grad_clip=1.0, warmup=3)

# One persistent REPL per worker thread (reused across that worker's tactics).
_TLS = __import__("threading").local()


_ALL_REPLS = []  # registry so every REPL ever created is closed at shutdown


def _repl():
    r = getattr(_TLS, "repl", None)
    if r is None or r.proc.poll() is not None:
        r = LeanREPL()
        _TLS.repl = r
        _ALL_REPLS.append(r)
    return r


def _rollout(thm, gen, tok):
    # repl=None -> the adapter creates a FRESH REPL per episode and closes it
    # (killpg) in its finally. A reused REPL accumulates Lean env/proofState
    # globally and never frees it: RSS balloons ~0.2GB/episode (166GB->6GB in 9
    # iters) even with the process COUNT bounded. Per-episode keeps RSS at the
    # ~860MB baseline with no growth; the ~0.2s spin-up is negligible.
    return run_lean_episode(thm, gen, tok, rule_mode=RULE_MODE,
                            max_steps=MAX_STEPS, repl=None)


def main():
    set_template(POLICY_MODEL)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    theorems = load_synth_theorems()
    print(f"{len(theorems)} theorems; credit={CREDIT} rule_mode={RULE_MODE} "
          f"G={G} iters={ITERS}", flush=True)

    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=120, max_batch=12)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    torch.manual_seed(SEED)  # vary generation sampling across seeds
    rng = random.Random(SEED)
    # ONE persistent pool for the whole run: a per-iter pool spawns fresh threads
    # each iteration, and the thread-local REPLs of the retired threads are never
    # closed -> 32 REPLs/iter leak (576 live REPLs, 189GB). Reusing the pool keeps
    # the thread-locals (and their REPLs) alive and reused across iters.
    ex = ThreadPoolExecutor(max_workers=G * 2)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch = rng.sample(theorems, min(TASKS_PER_ITER, len(theorems)))
        groups = []
        model.config.use_cache = True
        futs = {thm["name"]: [ex.submit(_rollout, thm, gen_srv.generate, tok)
                              for _ in range(G)]
                for thm in batch}
        for name, fs in futs.items():
            grp = [f.result() for f in fs]
            grp = [e for e in grp if e is not None and e.n_turns > 0]
            if len(grp) >= 2:
                groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print(f"iter {it}: no episodes", flush=True)
            continue
        succ = sum(e.env.success for e in eps) / len(eps)
        rew = sum(e.env.outcome_reward() for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        disc = sum(len(e.env.discharges) for e in eps) / len(eps)
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": round(succ, 3), "reward": round(rew, 3),
               "viol_per_ep": round(viol, 2), "disc_per_ep": round(disc, 2),
               "n_eps": len(eps), **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    ex.shutdown(wait=True)
    for r in _ALL_REPLS:  # kill every REPL (process group) so none linger
        try:
            r.close()
        except Exception:
            pass
    gen_srv.stop()
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT / "final")
    tok.save_pretrained(OUT / "final")
    print("LEAN TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
