"""RLVP training on ST-WebAgentBench (mirrors scripts/endless_train.py).

The stretch experiment: RL against Completion-under-Policy. Outcome reward =
task oracle; verifiable PATH penalty = ST-WebAgentBench per-step policy
violations (info['safety_report']) via rlvp/webarena_adapter.

PREREQUISITES (not auto-satisfied; see benchmarks/webarena/PILOT_REPORT.md):
  1. This process needs BOTH torch/transformers AND browsergym.stwebagentbench +
     playwright chromium installed in the SAME environment (the policy runs
     in-process with the Playwright env). If a dependency conflict makes that
     impossible, split into a browsergym rollout server + a torch policy client
     (the adapter's env calls are already isolated in run_webarena_episode).
  2. ST-WebAgentBench sites booted (GitLab / shopping_admin / SuiteCRM) with
     GITLAB / SHOPPING_ADMIN / WA_SUITECRM env vars set.
  3. Validate first: `python3 benchmarks/webarena/validate_env.py <task_id>`.

Usage:
    python3 scripts/webarena_train.py [iters] [--model ...] [--credit c3|outcome]
        [--seed S] [--task-ids 235,236,...] [--out NAME]

Task selection: pass --task-ids explicitly (DOM-only, non-fuzzy-oracle tasks).
Exclude the vision-advantage range 295-334 and the 31 fuzzy-match-oracle tasks.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.tau2_adapter import GenServer
from rlvp.webarena_adapter import run_webarena_episode

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
CREDIT, OUT_NAME, SEED, MODEL = "c3", "run_webarena", 7, "Qwen/Qwen3-8B"
TASK_IDS = None
for _i, _a in enumerate(sys.argv):
    if _a == "--credit": CREDIT = sys.argv[_i + 1]
    if _a == "--out": OUT_NAME = sys.argv[_i + 1]
    if _a == "--seed": SEED = int(sys.argv[_i + 1])
    if _a == "--model": MODEL = sys.argv[_i + 1]
    if _a == "--task-ids": TASK_IDS = [int(x) for x in sys.argv[_i + 1].split(",")]

if not TASK_IDS:
    sys.exit("provide --task-ids (comma-separated DOM-only, non-fuzzy ST-WAB task ids)")

G, TASKS_PER_ITER, MAX_STEPS = 4, 2, 15
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=0,
                  inner_epochs=2, lr=2e-5, micro_token_budget=1024,
                  clip_eps=0.2, grad_clip=1.0, warmup=3, max_episode_tokens=6144)


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

    json.dump({"model": MODEL, "credit": CREDIT, "seed": SEED, "task_ids": TASK_IDS,
               "iters": ITERS, "G": G}, open(OUT / "config.json", "w"), indent=1)
    print(f"{len(TASK_IDS)} ST-WAB tasks; model={MODEL} credit={CREDIT}", flush=True)
    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=384, max_batch=8)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch = rng.sample(TASK_IDS, min(TASKS_PER_ITER, len(TASK_IDS)))
        groups = []
        # NOTE: episodes run SEQUENTIALLY. Playwright's sync API is thread-affine
        # ("greenlet: cannot switch to a different thread"), so ThreadPoolExecutor
        # rollouts (as in endless/swesmith) do NOT work here. Concurrency would
        # require an async-Playwright rewrite or a subprocess rollout layer. This
        # is correct but slow -- WebArena training is deferred pending that work.
        for tid in batch:
            grp = []
            for _ in range(G):
                try:
                    grp.append(run_webarena_episode(tid, gen_srv.generate, tok,
                                                    "structural", MAX_STEPS, True, False))
                except Exception as exc:
                    print(f"episode error (task {tid}):", str(exc)[:120], flush=True)
            grp = [e for e in grp if e is not None]
            if len(grp) >= 2:
                groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print("no episodes this iter", flush=True); continue
        succ = sum(e.env.success for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        # CuP proxy: solved AND zero policy violations
        cup = sum(1 for e in eps if e.env.success and not e.env.violations) / len(eps)
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": succ, "cup": cup, "viol_per_ep": viol,
               "n_eps": len(eps), **m, "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    gen_srv.stop()
    print("WEBARENA TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
