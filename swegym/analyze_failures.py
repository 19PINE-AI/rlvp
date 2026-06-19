"""Summarise all_results.json: clean count, per-version breakdown, failure
categorisation, and timing/disk stats."""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "cache/all_results.json")))

n = len(res)
clean = [r for r in res if r.get("clean")]
print(f"TOTAL {n} | CLEAN {len(clean)} | rate {len(clean)/n:.1%}\n")

# per-version
by_ver = collections.defaultdict(lambda: [0, 0])
for r in res:
    by_ver[r["version"]][0] += 1
    if r.get("clean"):
        by_ver[r["version"]][1] += 1
print("version | total | clean")
for v in sorted(by_ver):
    t, c = by_ver[v]
    print(f"  {v:9s} {t:3d} {c:3d}")

# failure categories
print("\nFAILURE BREAKDOWN (non-clean):")
cats = collections.Counter()
for r in res:
    if r.get("clean"):
        continue
    if not r.get("setup_ok"):
        cats["setup_failed"] += 1
    elif r.get("f2p_fail_on_base") is False:
        cats["f2p_passed_on_base (no real bug repro)"] += 1
    elif r.get("f2p_pass_on_gold") is False:
        cats["f2p_failed_after_gold"] += 1
    else:
        cats["other"] += 1
for k, v in cats.most_common():
    print(f"  {k}: {v}")

# timing
secs = [r.get("seconds", 0) for r in res]
secs_sorted = sorted(secs)
import statistics
print(f"\nTIMING per task (s): mean {statistics.mean(secs):.1f} "
      f"median {statistics.median(secs):.1f} "
      f"p90 {secs_sorted[int(0.9*len(secs))]:.1f} max {max(secs):.1f}")
print(f"  (these are with SHARED venvs; first task per group pays venv build)")
