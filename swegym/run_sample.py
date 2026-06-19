"""Verify the harness on a sample of dask instances spanning versions/groups.

Prints a table and (optionally) runs the full set. Tasks are grouped by
pin-group so the shared venv is built once per group.
"""
import json
import os
import sys
import time

from swe_env_setup import setup_and_verify, pin_group_for, cleanup_worktree

WORK = "/tmp/swegym_sample"


def pick_sample(insts, n_per_group=1, total=8):
    """One per pin-group, then fill up to `total` with more, spanning versions."""
    by_group = {}
    for r in insts:
        by_group.setdefault(pin_group_for(r["version"]), []).append(r)
    chosen, seen = [], set()
    for g in sorted(by_group):
        r = by_group[g][0]
        chosen.append(r); seen.add(r["instance_id"])
    # fill remaining by walking groups round-robin
    i = 1
    while len(chosen) < total:
        added = False
        for g in sorted(by_group):
            if i < len(by_group[g]):
                r = by_group[g][i]
                if r["instance_id"] not in seen:
                    chosen.append(r); seen.add(r["instance_id"]); added = True
                    if len(chosen) >= total:
                        break
        i += 1
        if not added:
            break
    return chosen[:total]


def main():
    insts = json.load(open(os.path.join(os.path.dirname(__file__),
                                         "cache/dask_instances.json")))
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        sample = insts
    else:
        total = int(sys.argv[1]) if len(sys.argv) > 1 else 8
        sample = pick_sample(insts, total=total)

    rows = []
    for inst in sample:
        wd = os.path.join(WORK, inst["instance_id"])
        t0 = time.time()
        res = setup_and_verify(inst, wd, run_gold=True, run_p2p=True,
                               p2p_limit=30)
        res["seconds"] = round(time.time() - t0, 1)
        cleanup_worktree(wd)
        rows.append(res)
        ok = (res["setup_ok"] and res["f2p_fail_on_base"]
              and res["f2p_pass_on_gold"])
        print(f"  done {inst['instance_id']:24s} v{res['version']:9s} "
              f"clean={ok} {res['seconds']}s err={res['error']}")

    hdr = ("instance_id", "ver", "group", "setup_ok", "f2p_fails_base",
           "f2p_passes_gold", "p2p_passes_gold", "seconds")
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in (
            r["instance_id"], r["version"], r["group"], r["setup_ok"],
            r["f2p_fail_on_base"], r["f2p_pass_on_gold"],
            r["p2p_pass_on_gold"], r["seconds"])) + " |")

    clean = [r["instance_id"] for r in rows
             if r["setup_ok"] and r["f2p_fail_on_base"]
             and r["f2p_pass_on_gold"]]
    print(f"\nCLEAN: {len(clean)}/{len(rows)}")
    json.dump(rows, open(os.path.join(os.path.dirname(__file__),
                                      "cache/sample_results.json"), "w"),
              indent=2, default=str)


if __name__ == "__main__":
    main()
