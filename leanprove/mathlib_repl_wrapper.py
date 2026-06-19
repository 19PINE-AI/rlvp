#!/usr/bin/env python3
"""MathlibREPL: a LeanREPL that has all of Mathlib available.

Two things the base LeanREPL lacks for miniF2F:
  1. LEAN_PATH covering Mathlib + every dependency olean dir (batteries, aesop, Qq,
     ...), so `import Mathlib` resolves.
  2. A warmed base environment: it sends `import Mathlib` + the miniF2F `open`s ONCE
     (~12s cold load), stores the resulting env id, and every start_theorem builds
     on that env -- so the ~12s Mathlib load is paid once per REPL, not per theorem.

Reuse the base LeanREPL machinery (raw-read timeout, killpg, etc.) unchanged.
"""
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_PKGS = glob.glob(os.path.join(
    _HERE, "mathlib_repl", ".lake", "packages", "*", ".lake", "build", "lib", "lean"))
_REPLLIB = os.path.join(_HERE, "repl", ".lake", "build", "lib")
MATHLIB_PREAMBLE = ("import Mathlib\nset_option maxHeartbeats 400000\n"
                    "open BigOperators Real Nat Topology Rat")

from lean_repl import LeanREPL, _first_error  # noqa: E402


class MathlibREPL(LeanREPL):
    def __init__(self, timeout=30.0, warmup_timeout=240.0):
        # LEAN_PATH must be set BEFORE the subprocess launches (LeanREPL._env copies
        # os.environ). Process-global is fine: a Mathlib run uses only Mathlib REPLs.
        os.environ["LEAN_PATH"] = ":".join(_ML_PKGS + [_REPLLIB])
        super().__init__(timeout=timeout, warmup_timeout=warmup_timeout)
        self.base_env = None
        resp = self._send({"cmd": MATHLIB_PREAMBLE})  # the slow cold Mathlib load
        self.base_env = resp.get("env", 0)

    def start_theorem(self, statement: str) -> dict:
        """Open `statement := by sorry` ON TOP of the Mathlib base env."""
        stmt = statement.strip()
        if ":=" in stmt:
            stmt = stmt.split(":=")[0].strip()
        cmd = f"{stmt} := by sorry"
        resp = self._send({"cmd": cmd, "env": self.base_env})
        sorries = resp.get("sorries", [])
        err = _first_error(resp.get("messages", []))
        if not sorries:
            return {"proofState": None, "goal": None,
                    "error": err or "no sorry slot produced",
                    "latency_ms": resp["_latency_ms"], "raw": resp}
        s = sorries[0]
        return {"proofState": s["proofState"], "goal": s["goal"], "error": None,
                "latency_ms": resp["_latency_ms"], "raw": resp}


if __name__ == "__main__":
    import time
    r = MathlibREPL()
    print(f"Mathlib base env = {r.base_env} ({len(_ML_PKGS)} pkg dirs on LEAN_PATH)")
    for stmt, tac in [("theorem t1 (a b : Real) : a + b = b + a", "ring"),
                      ("theorem t2 : (2 : Nat) + 2 = 4", "rfl"),
                      ("theorem t3 : 9 ! % 10 = 0", "decide")]:
        st = r.start_theorem(stmt)
        if st["proofState"] is None:
            print(f"  {stmt[:30]}: OPEN-FAIL {st['error']}")
            continue
        t0 = time.time()
        res = r.apply_tactic(st["proofState"], tac)
        print(f"  {stmt[:34]:36s} by {tac:6s} -> done={res['done']} "
              f"err={str(res['error'])[:30]} ({time.time()-t0:.2f}s)")
    r.close()
    print("MATHLIB REPL WRAPPER OK")
