"""E-C: does SWE admit a verifiable potential Phi strictly FINER than the terminal
outcome, and is it REACHABLE?  Phi = (#FAIL_TO_PASS tests passing)/total.

The verifiable-potential claim (FINDINGS sec 11): a domain admits a useful dense
reward iff it has a verifiable Phi strictly finer than the outcome. For SWE:
  * SINGLE-F2P instance (total==1): Phi in {0,1} == the all-pass outcome -> NO finer
    structure. The claim predicts dense process reward CANNOT help here.
  * MULTI-F2P instance (total>=2): Phi can take intermediate values -> a strictly
    finer potential EXISTS. But it only HELPS if intermediate Phi is REACHABLE by the
    policy's rollouts (within-group variance > 0). If every rollout is all-or-nothing
    (Phi in {0,1}), the finer potential is VACUOUS -> predicts no benefit, consistent
    with the sec 15 SWE null (both arms 0%).

Modes:
  structural : split small-patch instances by |FAIL_TO_PASS|; report the single/multi
               counts + the F2P-count distribution. Instant, metadata only (no GPU/CPU
               test runs).
  validate   : CPU. For a sample, build the worktree and check the instrument extremes:
               base (no fix) -> Phi=(0,total); gold patch -> Phi=(total,total). Confirms
               Phi measures what we claim AND that the F2P extremes are reachable.
  rollout    : GPU (queued). G rollouts/instance with measure_phi -> record the Phi
               DISTRIBUTION per instance; compare within-group variance multi vs single.

Usage: ec_f2p.py <structural|validate|rollout> [--n N] [--iters K] [--seed S]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "swegym"))

from rlvp.swe_adapter import load_small_patch_instances, _node_ids  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "structural"
N = None
ITERS = 6
SEED = 7
for i, a in enumerate(sys.argv):
    if a == "--n":
        N = int(sys.argv[i + 1])
    if a == "--iters":
        ITERS = int(sys.argv[i + 1])
    if a == "--seed":
        SEED = int(sys.argv[i + 1])

OUTD = ROOT / "results" / "ec_f2p"
OUTD.mkdir(parents=True, exist_ok=True)


def split_instances():
    insts = load_small_patch_instances(max_changed=8, max_files=1, max_hunks=2)
    single, multi = [], []
    for inst in insts:
        nf = len(_node_ids(inst["FAIL_TO_PASS"]))
        (single if nf <= 1 else multi).append((inst, nf))
    return insts, single, multi


def mode_structural():
    insts, single, multi = split_instances()
    dist = Counter(len(_node_ids(i["FAIL_TO_PASS"])) for i in insts)
    print(f"small-patch instances: {len(insts)}")
    print(f"  SINGLE-F2P (|F2P|==1): {len(single)}  -> Phi==outcome, no finer potential")
    print(f"  MULTI-F2P  (|F2P|>=2): {len(multi)}  -> finer potential EXISTS (if reachable)")
    print(f"  F2P-count distribution: {dict(sorted(dist.items()))}")
    rec = {"n": len(insts), "single": len(single), "multi": len(multi),
           "dist": dict(sorted(dist.items())),
           "single_ids": [i["instance_id"] for i, _ in single],
           "multi_ids": [i["instance_id"] for i, _ in multi]}
    (OUTD / "structural.json").write_text(json.dumps(rec, indent=2))
    print(f"-> wrote {OUTD/'structural.json'}")


def mode_validate():
    """CPU: confirm Phi measures #F2P-pass and the extremes are reachable."""
    import swe_env_setup as H
    from rlvp.swe_adapter import SweWorktree
    insts, single, multi = split_instances()
    # validate a few of each kind
    sample = [x for x, _ in (multi[:3] + single[:3])]
    if N:
        sample = sample[:N]
    log = open(OUTD / "validate.jsonl", "a")
    for inst in sample:
        iid = inst["instance_id"]
        total = len(_node_ids(inst["FAIL_TO_PASS"]))
        wd = f"/tmp/ec_validate/{iid}"
        wt = SweWorktree(inst, wd)
        wt.setup()
        if not wt.setup_ok:
            print(f"[{iid}] SETUP FAILED: {wt.setup_error}", flush=True)
            wt.close()
            continue
        base = wt.phi()                      # no source fix -> expect (0, total)
        try:
            H._apply_patch(wt.repo_dir, inst["patch"], "goldfix", log=[])
            gold = wt.phi()                  # gold fix -> expect (total, total)
        except Exception as e:               # noqa: BLE001
            gold = (-1, total)
            print(f"[{iid}] gold apply failed: {type(e).__name__}: {e}", flush=True)
        wt.close()
        ok = (base == (0, total) and gold == (total, total))
        rec = {"iid": iid, "total_f2p": total, "base_phi": base, "gold_phi": gold,
               "instrument_ok": ok}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{iid}] total={total} base={base} gold={gold} ok={ok}", flush=True)
    print("VALIDATE DONE", flush=True)


def mode_rollout():
    """GPU: G rollouts/instance with measure_phi -> Phi distribution per group."""
    import random
    import torch
    from transformers import AutoTokenizer
    from rlvp.rollout import set_template
    from rlvp.swe_adapter import run_swe_episode
    from rlvp.vllm_gen import VLLMGenServer
    insts, single, multi = split_instances()
    # balance: take min count of each kind so groups are comparable
    k = min(len(single), len(multi), N or 16)
    chosen = [("multi", x) for x, _ in multi[:k]] + [("single", x) for x, _ in single[:k]]
    MODEL_GEN, MODEL_HF = "Qwen/Qwen3-30B-A3B-FP8", "Qwen/Qwen3-30B-A3B"
    set_template(MODEL_GEN)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen = VLLMGenServer(MODEL_GEN, tok, max_new_tokens=512, temperature=1.0,
                        gpu_mem=0.85, max_model_len=8192, max_batch=24, enforce_eager=True)
    rng = random.Random(SEED)
    import os as _os
    outfile = _os.environ.get("EC_OUTFILE", "rollout.jsonl")
    maxsteps = int(_os.environ.get("EC_MAXSTEPS", "16"))
    log = open(OUTD / outfile, "a")
    G = ITERS
    for kind, inst in chosen:
        iid = inst["instance_id"]
        total = len(_node_ids(inst["FAIL_TO_PASS"]))
        phis = []
        for _ in range(G):
            ep = run_swe_episode(inst, gen.generate, tok, rule_mode="structural",
                                 max_steps=maxsteps, oracle=True, measure_phi=True)
            if ep is None:
                continue
            np_, tot = getattr(ep, "_swe_phi", (0, total))
            phis.append(np_ / tot if tot else 0.0)
        if not phis:
            continue
        import statistics as st
        rec = {"iid": iid, "kind": kind, "total_f2p": total, "G": len(phis),
               "phi_vals": [round(p, 3) for p in phis],
               "phi_mean": round(sum(phis) / len(phis), 3),
               "phi_var": round(st.pvariance(phis), 4),
               "n_partial": sum(1 for p in phis if 0 < p < 1),
               "n_solved": sum(1 for p in phis if p >= 0.999)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    gen.stop()
    print("ROLLOUT DONE", flush=True)


if __name__ == "__main__":
    {"structural": mode_structural, "validate": mode_validate,
     "rollout": mode_rollout}[MODE]()
