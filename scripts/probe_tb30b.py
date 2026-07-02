#!/usr/bin/env python3
"""P1 probe: is TerminalBench (easy slice) usable for RL at 30B?

Zero-shot capability probe for the harm-at-real-capability experiment (paper
Limitations / P1). Runs Qwen3-30B-A3B-FP8 via vLLM on the 12 harm-training
tasks + a deterministic sample of easy-difficulty tasks, G rollouts each, and
reports the metrics that decide trainability:
  - overall zero-shot success (off the 4B ~0.1 floor?)
  - per-task solve rate; fraction of tasks with >=1 success (reachable)
  - fraction of informative groups (0 < n_succ < G) -- what GRPO can learn from
  - zero-shot violations/episode (headroom for the harm penalty)

GPU-GATED: waits for the p0_meta queue (/tmp/p0_meta.alldone) or sustained
free GPU memory before loading vLLM, so it never competes with the running
30B AdamW seeds.

Usage: python3 scripts/probe_tb30b.py [--g 4] [--n-easy 24] [--gpu-mem 0.45]
Output: results/probe_tb30b/{rollouts.jsonl,summary.json}
"""
import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "termbench"))

MODEL = "Qwen/Qwen3-30B-A3B-FP8"
ALLDONE = Path("/tmp/p0_meta.alldone")

# 12 tasks the 4B harm runs trained on (scripts/termbench_train.py) -- the
# distribution the 30B harm experiment would use.
TRAIN_TASKS = [
    "hello-world", "sha-puzzle", "count-dataset-tokens", "fix-git",
    "sanitize-git-repo", "oom", "regex-chess", "break-filter-js-from-html",
    "tune-mjcf", "logistic-regression-divergence", "cancel-async-tasks",
    "build-cython-ext",
]


def gpu_free_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total = map(int, r.stdout.strip().split(","))
    return total - used


def wait_for_gpu(need_mib=50000):
    """Block until the p0 queue is done (authoritative) or memory has been
    free for 5 consecutive 2-min checks (fallback if the queue was killed)."""
    streak = 0
    while True:
        if ALLDONE.exists() and gpu_free_mib() >= need_mib:
            print("p0_meta.alldone present and GPU free -> starting", flush=True)
            return
        free = gpu_free_mib()
        streak = streak + 1 if free >= need_mib else 0
        if streak >= 5:
            print(f"GPU free ({free} MiB) for 10 min -> starting", flush=True)
            return
        time.sleep(120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=4, help="rollouts per task")
    ap.add_argument("--n-easy", type=int, default=24,
                    help="extra easy-difficulty tasks beyond the 12 train tasks")
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "probe_tb30b"
    out_dir.mkdir(parents=True, exist_ok=True)

    from load_tasks import list_tasks, load_task  # termbench
    easy = []
    for t in list_tasks():
        try:
            m = load_task(t)
        except Exception:
            continue
        if m.get("difficulty") == "easy" and t not in TRAIN_TASKS:
            easy.append(t)
    rng = random.Random(7)
    extra = rng.sample(easy, min(args.n_easy, len(easy)))
    tasks = TRAIN_TASKS + extra
    print(f"{len(tasks)} tasks ({len(TRAIN_TASKS)} train + {len(extra)} easy), "
          f"G={args.g}", flush=True)

    if not args.no_wait:
        wait_for_gpu()

    from transformers import AutoTokenizer
    from rlvp.rollout import set_template
    from rlvp.termbench_adapter import run_terminal_episode
    from rlvp.vllm_gen import VLLMGenServer

    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"loading {MODEL} (gpu_mem={args.gpu_mem}) ...", flush=True)
    gen = VLLMGenServer(MODEL, tok, max_new_tokens=256, temperature=0.8,
                        gpu_mem=args.gpu_mem, max_model_len=8192, max_batch=32)

    log = open(out_dir / "rollouts.jsonl", "a")
    per_task = {}

    def one(task_id):
        try:
            ep = run_terminal_episode(task_id, gen.generate, tok,
                                      rule_mode="structural",
                                      max_steps=args.max_steps,
                                      keep_image=True, verbose=False)
        except Exception as exc:
            return {"task": task_id, "error": f"{type(exc).__name__}: {exc}"[:200]}
        if ep is None:
            return {"task": task_id, "error": "no-episode"}
        return {"task": task_id,
                "succ": int(ep.env.success),
                "viol": len(ep.env.violations),
                "disch": len(ep.env.discharges),
                "steps": len(ep.env.calls)}

    t_start = time.time()
    # 4 concurrent Docker episodes (same as the harm harness); vLLM batches
    # across threads. Group tasks so per-task G rollouts share the built image.
    for tid in tasks:
        with ThreadPoolExecutor(max_workers=4) as ex:
            recs = list(ex.map(lambda _: one(tid), range(args.g)))
        ok = [r for r in recs if "error" not in r]
        for r in recs:
            log.write(json.dumps(r) + "\n")
        log.flush()
        if not ok:
            per_task[tid] = {"skipped": True,
                             "reason": recs[0].get("error", "?")[:120]}
            print(f"  {tid}: SKIP ({per_task[tid]['reason']})", flush=True)
            continue
        n_succ = sum(r["succ"] for r in ok)
        per_task[tid] = {
            "g": len(ok), "n_succ": n_succ,
            "viol_per_ep": round(sum(r["viol"] for r in ok) / len(ok), 2),
            "steps_per_ep": round(sum(r["steps"] for r in ok) / len(ok), 1),
        }
        print(f"  {tid}: {n_succ}/{len(ok)} solved, "
              f"viol/ep={per_task[tid]['viol_per_ep']}", flush=True)

    done = {k: v for k, v in per_task.items() if not v.get("skipped")}
    n_ep = sum(v["g"] for v in done.values())
    n_succ = sum(v["n_succ"] for v in done.values())
    informative = sum(1 for v in done.values() if 0 < v["n_succ"] < v["g"])
    summary = {
        "model": MODEL, "g": args.g, "temperature": 0.8,
        "n_tasks_attempted": len(tasks), "n_tasks_ran": len(done),
        "n_episodes": n_ep,
        "success_rate": round(n_succ / max(n_ep, 1), 3),
        "tasks_solved_ge1": sum(1 for v in done.values() if v["n_succ"] > 0),
        "frac_groups_informative": round(informative / max(len(done), 1), 3),
        "frac_groups_allfail": round(
            sum(1 for v in done.values() if v["n_succ"] == 0) / max(len(done), 1), 3),
        "viol_per_ep": round(
            sum(v["viol_per_ep"] * v["g"] for v in done.values()) / max(n_ep, 1), 2),
        "wall_min": round((time.time() - t_start) / 60, 1),
        "per_task": per_task,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== PROBE SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                     indent=2), flush=True)
    gen.stop()
    print("PROBE_TB30B DONE", flush=True)


if __name__ == "__main__":
    main()
