"""RLVP training on TerminalBench (mirrors scripts/tau2_train.py).

Policy = Qwen3-4B + LoRA (our HF policy, exact token bookkeeping). Each agent
turn is a bash command in the task's Docker container; reward = the task's
pytest oracle (1.0/0.0). Process rules = terminal RuleTracker
(repeat_error, blind_destructive) with paired discharges (blind_destructive,
made_progress).

Usage:
    cd /home/ubuntu/rlvp && python3 scripts/termbench_train.py [iters] \
        [--credit c3|outcome] [--rule-mode structural] [--anneal N] [--out NAME]
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
from rlvp.termbench_adapter import GenServer, run_terminal_episode

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
CREDIT = "c3"
ANNEAL = 0
RULE_MODE = "structural"
OUT_NAME = "run_termbench"
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
G = 6
TASKS_PER_ITER = 4
MAX_STEPS = 15
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

# ~12 small-image, fast TerminalBench tasks (python-3-13/slim bases, short
# solutions) per PIPELINE_STATUS cost notes. hello-world-like file/shell ops.
TRAIN_TASKS = [
    "hello-world",
    "sha-puzzle",
    "count-dataset-tokens",
    "fix-git",
    "sanitize-git-repo",
    "oom",
    "regex-chess",
    "break-filter-js-from-html",
    "tune-mjcf",
    "logistic-regression-divergence",
    "cancel-async-tasks",
    "build-cython-ext",
]

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=ANNEAL,
                  inner_epochs=2, lr=2e-5, micro_token_budget=1024,
                  clip_eps=0.2, grad_clip=1.0, warmup=3,
                  max_episode_tokens=3500)


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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    print(f"{len(TRAIN_TASKS)} train tasks; credit={CREDIT} rule_mode={RULE_MODE}",
          flush=True)
    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=200, max_batch=12)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    torch.manual_seed(SEED)  # vary generation sampling across seeds
    rng = random.Random(SEED)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch_tasks = rng.sample(TRAIN_TASKS, min(TASKS_PER_ITER, len(TRAIN_TASKS)))
        groups = []
        # Each episode owns a Docker container; cap concurrency to be frugal.
        with ThreadPoolExecutor(max_workers=G) as ex:
            futs = {t: [ex.submit(run_terminal_episode, t, gen_srv.generate, tok,
                                  RULE_MODE, MAX_STEPS, True, False)
                        for _ in range(G)]
                    for t in batch_tasks}
            for tid, fs in futs.items():
                grp = []
                for f in fs:
                    try:
                        grp.append(f.result())
                    except Exception as exc:
                        print(f"episode error ({tid}):", str(exc)[:120], flush=True)
                grp = [e for e in grp if e is not None]
                if len(grp) >= 2:
                    groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print("no episodes this iter", flush=True)
            continue
        succ = sum(e.env.success for e in eps) / len(eps)
        rew = sum(e.env.outcome_reward() for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        disch = sum(len(e.env.discharges) for e in eps) / len(eps)
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": succ, "reward": rew, "viol_per_ep": viol,
               "disch_per_ep": disch, "n_eps": len(eps),
               **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    gen_srv.stop()
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT / "final")
    tok.save_pretrained(OUT / "final")
    print("TERMBENCH TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
