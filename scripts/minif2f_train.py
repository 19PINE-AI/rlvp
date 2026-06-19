"""RLVP training on REAL miniF2F theorems with Qwen3-30B-A3B.

Architecture (validated piece by piece):
  * rollouts: vLLM fp8 (VLLMGenServer) -- fast sampling, LoRA hot-swapped each iter
  * backward: HF 4-bit QLoRA + LoRA (peft) -> grpo.update_policy (recomputes its own
    old-logprobs, so vLLM is just a sampler; HF/HF ratio stays consistent)
  * oracle: warm Mathlib REPL pool (import Mathlib once/REPL, recycled to bound RAM)
  * data: curated miniF2F algebra/number-theory (drop induction)

vLLM (~34GB) + QLoRA backward (~43GB peak) coexist on the 96GB card.

Usage:
  python3 scripts/minif2f_train.py [iters] [--credit c3|outcome] [--seed N]
      [--anneal N] [--out run_minif2f] [--algebra]
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
ELAN_HOME = ROOT / "leanprove" / ".elan"
os.environ["ELAN_HOME"] = str(ELAN_HOME)
os.environ["PATH"] = f"{ELAN_HOME / 'bin'}:{os.environ.get('PATH', '')}"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer
# NB: bitsandbytes / peft init CUDA at import -> a forked vLLM EngineCore then can't
# re-init CUDA. So import them LAZILY inside main(), AFTER vLLM has forked.

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.lean_adapter import run_lean_episode
from rlvp.vllm_gen import VLLMGenServer

sys.path.insert(0, str(ROOT / "leanprove"))
from load_minif2f import load_minif2f          # noqa: E402
from mathlib_repl_wrapper import MathlibREPL    # noqa: E402

# ---- args ----
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40
CREDIT, ANNEAL, OUT_NAME, SEED, ALGEBRA = "c3", 0, "run_minif2f", 7, False
for _i, _a in enumerate(sys.argv):
    if _a == "--credit": CREDIT = sys.argv[_i + 1]
    if _a == "--anneal": ANNEAL = int(sys.argv[_i + 1])
    if _a == "--out": OUT_NAME = sys.argv[_i + 1]
    if _a == "--seed": SEED = int(sys.argv[_i + 1])
    if _a == "--algebra": ALGEBRA = True

MODEL_GEN = "Qwen/Qwen3-30B-A3B-FP8"   # vLLM rollouts
MODEL_HF = "Qwen/Qwen3-30B-A3B"        # 4-bit QLoRA backward (bf16 ckpt)
G = 8
TASKS_PER_ITER = 6
MAX_STEPS = 8
RULE_MODE = "structural"
OUT = ROOT / "results" / f"run_{OUT_NAME}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- warm Mathlib REPL pool (thread-local, recycled) ----
_TLS = threading.local()
_ALL_REPLS = []
_REPL_RECYCLE = 16   # re-warm a REPL after this many theorems (bounds env growth)


def _repl():
    r = getattr(_TLS, "repl", None)
    n = getattr(_TLS, "n", 0)
    if r is None or r.proc.poll() is not None or n >= _REPL_RECYCLE:
        if r is not None:
            try: r.close()
            except Exception: pass
        r = MathlibREPL()            # ~12s cold import Mathlib, then fast
        _TLS.repl = r
        _ALL_REPLS.append(r)
        n = 0
    _TLS.n = n + 1
    return r


def _rollout(thm, gen, tok):
    return run_lean_episode(thm, gen, tok, rule_mode=RULE_MODE,
                            max_steps=MAX_STEPS, repl=_repl())


def main():
    set_template(MODEL_GEN)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- vLLM fp8 generator FIRST, before ANY CUDA init in this process. vLLM forks
    # its EngineCore, and a forked subprocess cannot re-init CUDA -- so torch.manual_
    # seed (which inits CUDA RNG) and the HF model load MUST come after. ---
    print("starting vLLM fp8 generator ...", flush=True)
    gen_srv = VLLMGenServer(MODEL_GEN, tok, max_new_tokens=200, temperature=1.0,
                            gpu_mem=0.45, max_model_len=4096, max_batch=32,
                            enable_lora=True, max_lora_rank=32, enforce_eager=True)

    torch.manual_seed(SEED)   # now safe: vLLM's EngineCore already forked

    # --- HF 4-bit QLoRA model for the backward pass (lazy imports: bnb/peft init
    # CUDA, which is fine now that vLLM has already forked its EngineCore) ---
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                              get_constant_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("loading 30B HF 4-bit ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    cfg = TrainConfig(credit=CREDIT, anneal_at=ANNEAL)
    cfg.micro_token_budget = 2048   # smaller microbatches bound the 30B backward peak
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    thms = load_minif2f(split="Valid", easy_only=True, algebra_only=ALGEBRA)
    print(f"{len(thms)} miniF2F theorems (algebra_only={ALGEBRA}); credit={CREDIT} "
          f"G={G} iters={ITERS} seed={SEED}", flush=True)

    adapter_dir = OUT / "adapter_cur"
    log = open(OUT / "train_log.jsonl", "a")
    import random
    rng = random.Random(SEED)
    ex = ThreadPoolExecutor(max_workers=G)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        # sync the current LoRA into vLLM (iter 1: zero-init LoRA == base model)
        model.save_pretrained(adapter_dir)
        gen_srv.set_lora(str(adapter_dir))

        batch = rng.sample(thms, min(TASKS_PER_ITER, len(thms)))
        groups = []
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
        m = update_policy(model, tok, adv, cfg, opt, sched)
        rec = {"iter": it, "succ": round(succ, 3), "reward": round(rew, 3),
               "viol_per_ep": round(viol, 2), "disch_per_ep": round(disc, 2),
               "n_eps": len(eps), **{k: v for k, v in m.items()},
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)

    ex.shutdown(wait=True)
    for r in _ALL_REPLS:
        try: r.close()
        except Exception: pass
    gen_srv.stop()
    print("MINIF2F TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
