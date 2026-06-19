"""RLVP training on SWE-Gym (dask): policy edits a REAL repo to fix REAL bugs,
with the hidden FAIL_TO_PASS/PASS_TO_PASS pytest suite as the terminal oracle
and verifiable per-step process signals (reproduce-before-patch, ran_tests,
untested_submit, edited_test_file).

Mirrors scripts/tau2_train.py: Qwen3-4B + LoRA r=32, small group size G (SWE
rollouts are heavy: real worktree + pytest per episode), GenServer for batched
generation, build_advantages (default credit c3) -> update_policy. The trainer
(grpo.py) and Episode shape are unchanged from the tau2 path.

NOTE: this is the TRAINING harness; the smoke gate is scripts/swe_smoke.py.
Run only when GPU budget allows -- SWE rollouts are minutes-scale each.

Usage:
  cd /home/ubuntu/rlvp && python3 scripts/swe_train.py [iters] \
      [--credit c3] [--rule-mode structural] [--anneal N] [--out run_swe] \
      [--n-insts 8] [--G 4]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.swe_adapter import GenServer, load_clean_instances, run_swe_episode

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
CREDIT = "c3"
ANNEAL = 0
RULE_MODE = "structural"
OUT_NAME = "run_swe"
N_INSTS = 8           # clean dask instances in the training pool
G = 4                 # SWE rollouts are heavy -> keep group size small
INSTS_PER_ITER = 2    # tasks (instances) sampled per iteration
MAX_STEPS = 12
for _i, _a in enumerate(sys.argv):
    if _a == "--credit":
        CREDIT = sys.argv[_i + 1]
    if _a == "--anneal":
        ANNEAL = int(sys.argv[_i + 1])
    if _a == "--rule-mode":
        RULE_MODE = sys.argv[_i + 1]
    if _a == "--out":
        OUT_NAME = sys.argv[_i + 1]
    if _a == "--n-insts":
        N_INSTS = int(sys.argv[_i + 1])
    if _a == "--G":
        G = int(sys.argv[_i + 1])

POLICY_MODEL = "Qwen/Qwen3-4B"
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=ANNEAL,
                  inner_epochs=2, lr=2e-5, micro_token_budget=1024,
                  clip_eps=0.2, grad_clip=1.0, warmup=3,
                  max_episode_tokens=5000)


def main():
    set_template(POLICY_MODEL)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL, dtype=torch.bfloat16, device_map="cuda")
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    instances = load_clean_instances(n=N_INSTS)
    print(f"{len(instances)} clean dask instances; credit={CREDIT} "
          f"rule_mode={RULE_MODE} G={G}", flush=True)

    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=400,
                        max_batch=G)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    rng = random.Random(7)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch = rng.sample(instances, min(INSTS_PER_ITER, len(instances)))
        groups = []
        # one worktree per episode; rollouts run concurrently (each in its own
        # worktree dir). The shared venv is reused -- run_swe_episode never
        # builds a venv. Concurrency is bounded by G*len(batch).
        with ThreadPoolExecutor(max_workers=G * len(batch)) as ex:
            futs = {inst["instance_id"]:
                    [ex.submit(run_swe_episode, inst, gen_srv.generate, tok,
                               RULE_MODE, MAX_STEPS) for _ in range(G)]
                    for inst in batch}
            for iid, fs in futs.items():
                grp = [f.result() for f in fs]
                grp = [e for e in grp if e is not None]
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

        cfg_eff = cfg
        if cfg.anneal_at and it > cfg.anneal_at:
            from dataclasses import replace as _replace
            cfg_eff = _replace(cfg, lam=0.0, beta=0.0)
        adv = build_advantages(groups, cfg_eff)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": succ, "reward": rew, "viol_per_ep": viol,
               "disc_per_ep": disc, "n_eps": len(eps), "n_groups": len(groups),
               **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    gen_srv.stop()
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT / "final")
    tok.save_pretrained(OUT / "final")
    print("SWE TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
