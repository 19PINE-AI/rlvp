"""Stage G: RLVP training on tau2-bench airline (run under .venv-tau2 python).

Policy = Qwen3-4B + LoRA (our HF policy, exact token bookkeeping), trained
WITHOUT the domain policy document in its prompt — compliance must come from
the reward. User simulator = tau2's, via litellm -> local vLLM Qwen3-8B.
Reward = tau2 ENV evaluator. Rules = structural tracker (write_before_lookup,
call_spam, unconfirmed_chain) with paired discharges.

Usage: .venv-tau2/bin/python scripts/tau2_train.py [iters] [--with-policy]
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TAU2_DATA_DIR", "/tmp/tau2-bench/data")
os.environ.setdefault("OPENAI_API_KEY", "local")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8011/v1")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8011/v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.tau2_adapter import GenServer, run_one_sim

from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
WITH_POLICY = "--with-policy" in sys.argv
CREDIT = "c3"
ANNEAL = 0
RULE_MODE = "structural"
OUT_NAME = "run_tau2"
for _i, _a in enumerate(sys.argv):
    if _a == "--credit":
        CREDIT = sys.argv[_i + 1]
    if _a == "--anneal":
        ANNEAL = int(sys.argv[_i + 1])
    if _a == "--rule-mode":
        RULE_MODE = sys.argv[_i + 1]
    if _a == "--out":
        OUT_NAME = sys.argv[_i + 1]
POLICY_MODEL = "Qwen/Qwen3-4B"
USER_LLM = "openai/Qwen/Qwen3-4B"
G = 6
TASKS_PER_ITER = 4
MAX_STEPS = 10
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig(credit=CREDIT, lam=0.25, beta=0.25, anneal_at=ANNEAL, inner_epochs=2,
                  lr=2e-5, micro_token_budget=768, clip_eps=0.2,
                  grad_clip=1.0, warmup=3)


def main():
    set_template(POLICY_MODEL)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
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

    tasks = get_tasks()
    train_tasks, eval_tasks = tasks[10:], tasks[:10]
    print(f"{len(train_tasks)} train tasks, {len(eval_tasks)} eval tasks; "
          f"with_policy={WITH_POLICY}", flush=True)

    gen_srv = GenServer(model, tok, temperature=1.0, max_batch=12)
    log = open(OUT / "train_log.jsonl", "a")

    import random
    rng = random.Random(7)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        batch_tasks = rng.sample(train_tasks, TASKS_PER_ITER)
        groups = []
        with ThreadPoolExecutor(max_workers=G * 2) as ex:
            futs = {t.id: [ex.submit(run_one_sim, t, gen_srv.generate, tok,
                                     WITH_POLICY, USER_LLM, MAX_STEPS, RULE_MODE) for _ in range(G)]
                    for t in batch_tasks}
            for tid, fs in futs.items():
                grp = [f.result() for f in fs]
                grp = [e for e in grp if e is not None]
                if len(grp) >= 2:
                    groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print("no episodes this iter", flush=True)
            continue
        if CREDIT == "llmcritic":
            from rlvp.tau2_adapter import label_tau2_episodes
            label_tau2_episodes(model, tok, eps)
        succ = sum(e.env.success for e in eps) / len(eps)
        rew = sum(e.env.outcome_reward() for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        adv = build_advantages(groups, cfg)
        model.config.use_cache = False
        m = update_policy(model, tok, adv, cfg, opt, sched)
        model.config.use_cache = True
        rec = {"iter": it, "succ": succ, "reward": rew, "viol_per_ep": viol,
               "n_eps": len(eps), **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    gen_srv.stop()
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT / "final")
    tok.save_pretrained(OUT / "final")
    print("TAU2 TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
