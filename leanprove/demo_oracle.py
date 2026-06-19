#!/usr/bin/env python3
"""End-to-end demonstration of the per-tactic verifiable oracle.

For each self-contained theorem: apply the known-GOOD tactic (expect proof
complete, no error) and the known-BAD tactic (expect error / not done). Then
run a multi-step proof showing the intermediate process signal. Asserts the
oracle's verdicts match expectations -- this is the gate proof for RLVP.
"""
import time
from lean_repl import LeanREPL
from load_theorems import load_theorems

data = load_theorems()
passed, failed = 0, 0


def check(label, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")


with LeanREPL() as repl:
    t_start = time.perf_counter()
    print(f"=== oracle gate proof on '{data['source']}' theorem set ===\n")

    for t in data["self_contained"]:
        print(f"[{t['name']}]  {t['statement']}")

        # GOOD tactic -> expect done, no error.
        st = repl.start_theorem(t["statement"])
        assert st["error"] is None, f"statement failed to parse: {st['error']}"
        rg = repl.apply_tactic(st["proofState"], t["good_tactic"])
        print(f"    good '{t['good_tactic']}': done={rg['done']} "
              f"error={'yes' if rg['error'] else 'no'} ({rg['latency_ms']} ms)")
        check(f"{t['name']} good tactic => proof complete",
              rg["done"] and rg["error"] is None)

        # BAD tactic -> expect NOT done (error or open goals).
        st2 = repl.start_theorem(t["statement"])
        rb = repl.apply_tactic(st2["proofState"], t["bad_tactic"])
        first = (rb["error"] or "").splitlines()
        print(f"    bad  '{t['bad_tactic']}': done={rb['done']} "
              f"error={first[1] if len(first) > 1 else first[:1]} ({rb['latency_ms']} ms)")
        check(f"{t['name']} bad tactic => rejected (not done)", not rb["done"])
        print()

    # Multi-step process signal.
    m = data["multistep_demo"]
    print(f"[{m['name']}]  multi-step process signal")
    st = repl.start_theorem(m["statement"])
    ps = st["proofState"]
    for i, tac in enumerate(m["steps"]):
        r = repl.apply_tactic(ps, tac)
        last = (i == len(m["steps"]) - 1)
        kind = "TERMINAL" if last else "INTERMEDIATE"
        print(f"    step {i+1} '{tac}': done={r['done']} error="
              f"{'yes' if r['error'] else 'no'} goals_remaining={len(r['goals'] or [])} "
              f"[{kind}] ({r['latency_ms']} ms)")
        if not last:
            check(f"step {i+1} valid intermediate (no error, goals remain)",
                  r["error"] is None and not r["done"] and len(r["goals"]) >= 1)
        else:
            check(f"step {i+1} completes proof", r["done"] and r["error"] is None)
        ps = r["proofState"]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"\n=== RESULT: {passed} passed, {failed} failed "
          f"(total {total_ms:.0f} ms incl. REPL warm-up) ===")
    raise SystemExit(0 if failed == 0 else 1)
