#!/usr/bin/env python3
"""Persistent Lean 4 REPL wrapper exposing a per-tactic verifiable oracle.

This is the dense verifiable-procedure signal for RLVP training on theorem
proving. A single long-lived `lake exe repl` subprocess is reused across many
tactic calls (important for rollout speed), communicating via blank-line
separated JSON on stdin/stdout.

Core flow:
  1. start_theorem(statement) -> {proofState, goal}   (creates a `... := by sorry`)
  2. apply_tactic(proofState, tactic) -> {done, error, goals, proofState, latency_ms}

`done == True and error is None`  => proof complete (REWARD).
`error is None and not done`      => valid intermediate step (process reward;
                                     caller can check goals shrank/changed).
`error is not None`               => invalid tactic (PENALTY).

The REPL numbers `proofState` globally across the session, so always thread the
`proofState` returned by the previous call.
"""
import json
import os
import select
import signal
import subprocess
import time
from pathlib import Path

REPL_DIR = Path(__file__).resolve().parent / "repl"
ELAN_HOME = Path(__file__).resolve().parent / ".elan"


def _env():
    env = dict(os.environ)
    env["ELAN_HOME"] = str(ELAN_HOME)
    env["PATH"] = f"{ELAN_HOME / 'bin'}:{env.get('PATH', '')}"
    return env


class LeanREPL:
    """A persistent Lean REPL subprocess. Reuse across tactics for speed."""

    def __init__(self, repl_dir=REPL_DIR, timeout=30.0, warmup_timeout=180.0):
        # `timeout`: per-tactic read budget once the process is warm (catches a
        # hanging elaborator). `warmup_timeout`: the FIRST command triggers the
        # cold Mathlib environment load (tens of seconds) -- legitimately slow,
        # so it gets the longer budget. `_warm` flips after the first response.
        self.timeout = timeout
        self.warmup_timeout = warmup_timeout
        self._warm = False
        # `lake exe repl` is a WRAPPER that spawns the real `repl` binary as a
        # child. start_new_session=True puts both in their own process group so
        # _killtree() can SIGKILL the WHOLE group -- killing only the lake parent
        # orphans the repl child, which leaks (576 live REPLs -> 189GB once).
        self.proc = subprocess.Popen(
            ["lake", "exe", "repl"],
            cwd=str(repl_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # DEVNULL, not PIPE: we never read stderr, and an undrained stderr
            # PIPE deadlocks the REPL once it writes ~64KB (it blocks on write,
            # emits no stdout, and our readline hangs forever).
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=_env(),
            start_new_session=True,
        )

    def _killtree(self):
        """SIGKILL the whole process group (lake wrapper + repl child)."""
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)  # reap so it can't linger as a zombie
        except Exception:
            pass

    def _send(self, payload: dict) -> dict:
        """Send one JSON command, read one JSON object back. Returns (obj, latency_ms).

        Enforces self.timeout on the READ: a tactic that makes the Lean elaborator
        hang (simp loop, decide/omega pathology) emits no output and would block
        readline() forever. On timeout we kill the REPL and raise -- the caller
        treats it as a failed tactic; the process is dead so it gets restarted."""
        if self.proc.poll() is not None:
            raise RuntimeError("REPL process has exited")
        line = json.dumps(payload)
        t0 = time.perf_counter()
        self.proc.stdin.write(line + "\n\n")
        self.proc.stdin.flush()
        # Read raw bytes via os.read on the fd (NOT buffered readline): the REPL
        # reply is multi-line, and a buffered readline() slurps the whole reply
        # into Python's text buffer, after which select() polls an empty fd and
        # blocks forever. The reply terminates with a blank line ("\n\n").
        budget = self.timeout if self._warm else self.warmup_timeout
        fd = self.proc.stdout.fileno()
        deadline = t0 + budget
        buf = b""
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self._killtree()
                raise RuntimeError(f"REPL read timeout after {budget}s")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue  # loop re-checks the deadline
            chunk = os.read(fd, 65536)
            if chunk == b"":
                raise RuntimeError("REPL closed stdout unexpectedly")
            buf += chunk
            if b"\n\n" in buf.lstrip(b"\r\n "):  # complete JSON + blank-line terminator
                break
        self._warm = True  # first reply received; tighten the per-tactic budget
        latency_ms = (time.perf_counter() - t0) * 1000.0
        obj = json.loads(buf.decode("utf-8", errors="replace").strip().split("\n\n")[0])
        obj["_latency_ms"] = round(latency_ms, 2)
        return obj

    def start_theorem(self, statement: str) -> dict:
        """Submit a theorem whose body is `by sorry` and return the opening proof state.

        `statement` is the part before `:=`, e.g. "theorem t (n:Nat) : n+0=n".
        Returns {proofState, goal, latency_ms} or {error, latency_ms}.
        """
        stmt = statement.strip()
        if ":=" in stmt:
            stmt = stmt.split(":=")[0].strip()
        cmd = f"{stmt} := by sorry"
        resp = self._send({"cmd": cmd})
        sorries = resp.get("sorries", [])
        # Errors at the command level (e.g. statement doesn't parse).
        err = _first_error(resp.get("messages", []))
        if not sorries:
            return {"proofState": None, "goal": None, "error": err or "no sorry slot produced",
                    "latency_ms": resp["_latency_ms"], "raw": resp}
        s = sorries[0]
        return {"proofState": s["proofState"], "goal": s["goal"],
                "error": None, "latency_ms": resp["_latency_ms"], "raw": resp}

    def apply_tactic(self, proof_state: int, tactic: str) -> dict:
        """Apply one tactic to a proof state.

        Returns dict with:
          done      : bool  -- proof complete (no goals remain)
          error     : str|None -- Lean error message if the tactic is invalid
          goals     : list[str] -- remaining goals (empty if done)
          proofState: int|None -- new proof state to thread into the next call
          latency_ms: float
        """
        resp = self._send({"tactic": tactic, "proofState": proof_state})
        # Invalid tactic / parse error => {"message": "Lean error:\n..."}.
        if "message" in resp and "proofState" not in resp:
            return {"done": False, "error": resp["message"], "goals": None,
                    "proofState": None, "latency_ms": resp["_latency_ms"], "raw": resp}
        # Also surface error-severity messages if present.
        err = _first_error(resp.get("messages", []))
        goals = resp.get("goals", [])
        status = resp.get("proofStatus", "")
        done = (status == "Completed") or (not goals and err is None)
        return {"done": done and err is None, "error": err, "goals": goals,
                "proofState": resp.get("proofState"), "latency_ms": resp["_latency_ms"],
                "raw": resp}

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        # Always kill the whole group: terminate() on the lake parent alone
        # leaves the repl child orphaned (the leak).
        self._killtree()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _first_error(messages):
    for m in messages or []:
        if m.get("severity") == "error":
            return m.get("data", "error")
    return None


if __name__ == "__main__":
    # Smoke test: correct vs wrong tactic, with latency.
    with LeanREPL() as repl:
        print("=== start theorem (warm-up; includes first-call cost) ===")
        st = repl.start_theorem("theorem demo (n : Nat) : n + 0 = n")
        print(st["goal"], f"({st['latency_ms']} ms)")

        print("\n=== CORRECT tactic: omega ===")
        r = repl.apply_tactic(st["proofState"], "omega")
        print(f"done={r['done']} error={r['error']} goals={r['goals']} ({r['latency_ms']} ms)")

        print("\n=== WRONG tactic: rfl on commutativity ===")
        st2 = repl.start_theorem("theorem demo2 (a b : Nat) : a + b = b + a")
        r2 = repl.apply_tactic(st2["proofState"], "rfl")
        err_line = (r2["error"] or "").splitlines()[1] if r2["error"] else None
        print(f"done={r2['done']} error={err_line!r} ({r2['latency_ms']} ms)")

        print("\n=== latency over 20 repeated tactic calls (persistent process) ===")
        lats = []
        for _ in range(20):
            s = repl.start_theorem("theorem loop (n : Nat) : n + 0 = n")
            rr = repl.apply_tactic(s["proofState"], "omega")
            lats.append(rr["latency_ms"])
        lats.sort()
        print(f"n={len(lats)} min={lats[0]:.1f} median={lats[len(lats)//2]:.1f} max={lats[-1]:.1f} ms")
