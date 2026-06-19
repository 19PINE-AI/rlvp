#!/usr/bin/env python3
"""Does the oracle+small-patch SWE variant lift 30B off the 0% floor?
Runs small-patch dask instances WITH and WITHOUT the oracle file hint, to
measure the lift. Reuses the proven worktree+pytest oracle.

Usage: probe_swe_oracle.py [model] [n] [gpu_mem]
"""
import sys
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from rlvp.rollout import set_template            # noqa: E402
from rlvp.vllm_gen import VLLMGenServer          # noqa: E402
from rlvp.swe_adapter import load_small_patch_instances, run_swe_episode  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
GPU_MEM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55


def run_set(insts, g, tok, oracle):
    n_ep = n_succ = n_tests = 0
    for inst in insts:
        try:
            ep = run_swe_episode(inst, g, tok, rule_mode="structural",
                                 max_steps=16, oracle=oracle)
        except Exception as e:
            print(f"  {inst['instance_id']}: ERR {type(e).__name__}", flush=True)
            continue
        if ep is None:
            continue
        n_ep += 1
        r = ep.env.outcome_reward()
        n_succ += int(r > 0)
        n_tests += int(len(getattr(ep.env, "discharges", [])) > 0)
        print(f"  {'ORACLE' if oracle else 'plain '} {inst['instance_id']}: "
              f"reward={r}", flush=True)
    return n_ep, n_succ, n_tests


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    insts = load_small_patch_instances(max_changed=8, max_files=1, max_hunks=2, n=N)
    print(f"small-patch instances: {len(insts)} | loading {MODEL} ...", flush=True)
    gen = VLLMGenServer(MODEL, tok, max_new_tokens=640, temperature=0.6,
                        gpu_mem=GPU_MEM, max_model_len=12288, max_batch=24)
    g = gen.generate

    print("\n=== WITHOUT oracle (small-patch only) ===", flush=True)
    e0, s0, t0 = run_set(insts, g, tok, oracle=False)
    print("\n=== WITH oracle (small-patch + file hint) ===", flush=True)
    e1, s1, t1 = run_set(insts, g, tok, oracle=True)

    print(f"\nRESULT small-patch n={len(insts)}:")
    print(f"  plain : episodes={e0} solved={s0} ran-tests={t0}")
    print(f"  oracle: episodes={e1} solved={s1} ran-tests={t1}")
    gen.stop()
    print("SWE-ORACLE PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
