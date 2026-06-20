"""EXPERIMENT #3 (hard domain): RLVP on SWE-bench dask at 30B, where the outcome is
BLIND (0% solve) and a verifiable un-gameable PROGRESS metric is hard to define.

Same validated 30B stack as minif2f_train (vLLM-fp8 rollouts + 4-bit QLoRA backward +
Muon + LoRA hot-swap), but rollouts are SWE worktree+pytest episodes (no REPL pool).

The point: SWE's process discharges (reproduced / ran_tests) are BOTH gameable -- you
can farm them without fixing the bug. So in the blind (all-fail) regime we predict the
naive process credit (structural c3) FARMS the signal (high discharge, ~0 success) while
outcome-gating (c4) pays nothing without a real fix. This demonstrates the hard-domain
limit: no free un-gameable progress -> no safe dense learning signal.

Usage: swe_train_30b.py [iters] [--credit c3|c4|outcome] [--rule-mode structural|outcome]
                        [--muon] [--lr X] [--out NAME]
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

import torch
from transformers import AutoTokenizer

from rlvp.grpo import TrainConfig, build_advantages, update_policy
from rlvp.rollout import set_template
from rlvp.swe_adapter import load_small_patch_instances, run_swe_episode
from rlvp.vllm_gen import VLLMGenServer

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
CREDIT, OUT_NAME, SEED, MUON, LR, RMODE = "c3", "run_swe30b", 7, False, None, "structural"
for _i, _a in enumerate(sys.argv):
    if _a == "--credit": CREDIT = sys.argv[_i + 1]
    if _a == "--out": OUT_NAME = sys.argv[_i + 1]
    if _a == "--seed": SEED = int(sys.argv[_i + 1])
    if _a == "--muon": MUON = True
    if _a == "--lr": LR = float(sys.argv[_i + 1])
    if _a == "--rule-mode": RMODE = sys.argv[_i + 1]
if LR is None:
    LR = 1e-3 if MUON else 3e-5

MODEL_GEN, MODEL_HF = "Qwen/Qwen3-30B-A3B-FP8", "Qwen/Qwen3-30B-A3B"
G, TASKS_PER_ITER, MAX_STEPS = 6, 4, 16
OUT = ROOT / "results" / f"run_{OUT_NAME}"
OUT.mkdir(parents=True, exist_ok=True)


def _rollout(inst, gen, tok):
    return run_swe_episode(inst, gen, tok, rule_mode=RMODE, max_steps=MAX_STEPS,
                           oracle=True)   # oracle hint: small-patch localized fixes


def main():
    set_template(MODEL_GEN)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("starting vLLM fp8 generator ...", flush=True)
    gen_srv = VLLMGenServer(MODEL_GEN, tok, max_new_tokens=512, temperature=1.0,
                            gpu_mem=0.48, max_model_len=8192, max_batch=24,
                            enable_lora=True, max_lora_rank=32, enforce_eager=True)
    torch.manual_seed(SEED)

    from transformers import BitsAndBytesConfig, get_constant_schedule_with_warmup
    from transformers import AutoModelForCausalLM
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    if MUON:
        from rlvp.muon import Muon
        opt = Muon(trainable, lr=LR, momentum=0.95)
    else:
        opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"optimizer={'Muon' if MUON else 'AdamW'} lr={LR} credit={CREDIT} rule={RMODE}", flush=True)
    cfg = TrainConfig(credit=CREDIT)
    cfg.micro_token_budget = 2048
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)

    insts = load_small_patch_instances(max_changed=8, max_files=1, max_hunks=2)
    print(f"{len(insts)} small-patch SWE instances; G={G} iters={ITERS} seed={SEED}", flush=True)
    adapter_dir = OUT / "adapter_cur"
    log = open(OUT / "train_log.jsonl", "a")
    import random
    rng = random.Random(SEED)
    ex = ThreadPoolExecutor(max_workers=6)  # match SWE_VENV_SLOTS: avoid venv-build race
    for it in range(1, ITERS + 1):
        t0 = time.time()
        model.save_pretrained(adapter_dir)
        gen_srv.set_lora(str(adapter_dir))
        batch = rng.sample(insts, min(TASKS_PER_ITER, len(insts)))
        groups = []
        futs = {inst["instance_id"]: [ex.submit(_rollout, inst, gen_srv.generate, tok)
                                      for _ in range(G)] for inst in batch}
        for name, fs in futs.items():
            grp = []
            for f in fs:
                try:
                    e = f.result()
                except Exception:
                    e = None
                if e is not None and e.n_turns > 0:
                    grp.append(e)
            if len(grp) >= 2:
                groups.append(grp)
        eps = [e for g in groups for e in g]
        if not eps:
            print(f"iter {it}: no episodes", flush=True)
            continue
        succ = sum(e.env.success for e in eps) / len(eps)
        viol = sum(len(e.env.violations) for e in eps) / len(eps)
        disc = sum(len(e.env.discharges) for e in eps) / len(eps)
        adv = build_advantages(groups, cfg)
        m = update_policy(model, tok, adv, cfg, opt, sched)
        rec = {"iter": it, "succ": round(succ, 3), "viol_per_ep": round(viol, 2),
               "disch_per_ep": round(disc, 2), "n_eps": len(eps),
               **{k: v for k, v in m.items()}, "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    ex.shutdown(wait=True)
    gen_srv.stop()
    print("SWE30B TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
