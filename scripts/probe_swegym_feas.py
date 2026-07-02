#!/usr/bin/env python3
"""P2 probe, stage 1 (CPU-only): which SWE-Gym-Lite repo families have a
working no-Docker venv oracle on this box?

For every non-dask repo family in SWE-Gym-Lite, take the 2 instances with the
smallest gold patch and run the proven single-instance oracle check
(swegym/run_one.py: clone @ base_commit -> venv -> editable install ->
test_patch -> F2P fails -> gold patch -> F2P passes). A family is FEASIBLE if
at least one instance verifies end-to-end.

This is pure CPU (git + pip + pytest); it runs alongside the GPU queue.
Stage 2 (30B solve-rate on the feasible slice) is GPU-gated and separate.

Usage: python3 scripts/probe_swegym_feas.py [--per-repo 2] [--timeout 1500]
Output: results/probe_swegym/feasibility.jsonl + summary.json
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEGYM = ROOT / "swegym"
sys.path.insert(0, str(SWEGYM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1500, help="sec per instance")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    out_dir = ROOT / "results" / "probe_swegym"
    out_dir.mkdir(parents=True, exist_ok=True)

    from load_tasks import load_tasks
    tasks = load_tasks()
    by_repo = {}
    for t in tasks:
        if t["repo"] == "dask/dask":
            continue  # known: oracle works, 30B solve rate 0% (FINDINGS 15/16)
        by_repo.setdefault(t["repo"], []).append(t)

    picks = []
    for repo, ts in sorted(by_repo.items()):
        ts = sorted(ts, key=lambda t: len(t.get("patch") or ""))
        picks += [(repo, t["instance_id"]) for t in ts[:args.per_repo]]
    print(f"{len(by_repo)} repo families, {len(picks)} instances to verify",
          flush=True)

    log = open(out_dir / "feasibility.jsonl", "a")

    def check(item):
        repo, iid = item
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, str(SWEGYM / "run_one.py"), iid],
                capture_output=True, text=True, timeout=args.timeout,
                cwd=str(SWEGYM))
            ok = r.returncode == 0
            tail = (r.stdout + r.stderr)[-400:]
        except subprocess.TimeoutExpired:
            ok, tail = False, "TIMEOUT"
        rec = {"repo": repo, "instance_id": iid, "oracle_ok": ok,
               "wall_s": round(time.time() - t0, 1),
               "tail": None if ok else tail}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  {iid}: {'OK' if ok else 'FAIL'} ({rec['wall_s']}s)", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(check, picks))

    fam = {}
    for r in recs:
        fam.setdefault(r["repo"], []).append(r["oracle_ok"])
    from collections import Counter
    cnt = Counter(t["repo"] for t in tasks)
    feasible = sorted(k for k, v in fam.items() if any(v))
    summary = {
        "feasible_repos": feasible,
        "infeasible_repos": sorted(k for k, v in fam.items() if not any(v)),
        "per_repo": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(fam.items())},
        "n_instances_in_feasible_families": sum(cnt[r] for r in feasible),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FEASIBILITY SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    print("PROBE_SWEGYM_FEAS DONE", flush=True)


if __name__ == "__main__":
    main()
