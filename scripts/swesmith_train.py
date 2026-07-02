"""RLVP training on SWE-smith (mirrors scripts/endless_train.py).

Policy = Qwen3 + LoRA. Each agent turn is a bash command in the repo's Docker
container; reward = F2P(+sampled P2P) pytest oracle. Process rules = the
discipline tracker (edited_test_file, untested_edit, blind_destructive,
repeat_error + ran_tests/made_progress discharges) via rlvp/swesmith_adapter.

Usage:
    cd /home/ubuntu/rlvp && python3 scripts/swesmith_train.py [iters] \
        [--model Qwen/Qwen3-8B] [--credit c3|outcome] [--rule-mode structural] \
        [--seed S] [--n-tasks 60] [--out NAME]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.swesmith_adapter import GenServer, run_swesmith_episode, load_slice

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
CREDIT, RULE_MODE, OUT_NAME, SEED = "c3", "structural", "run_swesmith", 7
MODEL, N_TASKS = "Qwen/Qwen3-8B", 60
# easiest strategies only (func_basic / lm_modify / procedural func_pm_*)
EASY = {"func_basic", "lm_modify"}
EASY_PREFIX = "func_pm_"
for _i, _a in enumerate(sys.argv):
    if _a == "--credit": CREDIT = sys.argv[_i + 1]
    if _a == "--rule-mode": RULE_MODE = sys.argv[_i + 1]
    if _a == "--out": OUT_NAME = sys.argv[_i + 1]
    if _a == "--seed": SEED = int(sys.argv[_i + 1])
    if _a == "--model": MODEL = sys.argv[_i + 1]
    if _a == "--n-tasks": N_TASKS = int(sys.argv[_i + 1])

G, TASKS_PER_ITER, MAX_STEPS = 6, 3, 20
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

import re as _re
def _strat(iid):
    tail = iid.split(".")[-1]
    return _re.sub(r"__[a-z0-9]+$", "", tail)
POOL = [d for d in load_slice()
        if _strat(d["instance_id"]) in EASY or _strat(d["instance_id"]).startswith(EASY_PREFIX)]
POOL = POOL[:N_TASKS]

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=0,
                  inner_epochs=2, lr=2e-5, micro_token_budget=1024,
                  clip_eps=0.2, grad_clip=1.0, warmup=3, max_episode_tokens=4096)


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    json.dump({"model": MODEL, "credit": CREDIT, "rule_mode": RULE_MODE, "seed": SEED,
               "n_tasks": len(POOL), "iters": ITERS, "G": G}, open(OUT / "config.json", "w"), indent=1)
    print(f"{len(POOL)} SWE-smith tasks; model={MODEL} credit={CREDIT}", flush=True)
    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=320, max_batch=12)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch = rng.sample(POOL, min(TASKS_PER_ITER, len(POOL)))
        groups = []
        with ThreadPoolExecutor(max_workers=G) as ex:
            futs = {d["instance_id"]: [ex.submit(run_swesmith_episode, d, gen_srv.generate,
                                                  tok, RULE_MODE, MAX_STEPS, False)
                                       for _ in range(G)] for d in batch}
            for iid, fs in futs.items():
                grp = []
                for f in fs:
                    try:
                        grp.append(f.result())
                    except Exception as exc:
                        print(f"episode error ({iid}):", str(exc)[:120], flush=True)
                grp = [e for e in grp if e is not None]
                if len(grp) >= 2:
                    groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print("no episodes this iter", flush=True); continue
        succ = sum(e.env.success for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        disch = sum(len(e.env.discharges) for e in eps) / len(eps)
        # partial-progress Phi (fraction of F2P passing) -- the trajectory-quality
        # signal that matters in the all-fail regime (succ==0 for both arms).
        phi = sum(getattr(e.env, "phi", 0.0) for e in eps) / len(eps)
        # per-rule-type breakdown: are bad-practice violations falling and
        # productive actions rising? (the "process reward improves the trajectory
        # even without outcome success" argument).
        from collections import Counter
        vc, dc = Counter(), Counter()
        steps = 0
        for e in eps:
            for _, r in e.env.violations:
                vc[r] += 1
            for _, r in e.env.discharges:
                dc[r] += 1
            steps += len(e.env.calls)
        n = len(eps)
        brk = {f"v_{k}": round(v / n, 2) for k, v in vc.items()}
        brk.update({f"d_{k}": round(v / n, 2) for k, v in dc.items()})
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": succ, "reward": succ, "phi": round(phi, 3),
               "viol_per_ep": viol, "disch_per_ep": disch,
               "steps_per_ep": round(steps / n, 1), "n_eps": n, **brk, **m,
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    gen_srv.stop()
    print("SWESMITH TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
