#!/usr/bin/env python3
"""
run_one.py -- run ONE TerminalBench task end-to-end via direct Docker.

This bypasses the terminal-bench Python harness (which requires py>=3.12) and
reproduces its exact semantics with `docker build/run/exec/cp`:

  1. build the task image from its Dockerfile  (WORKDIR /app)
  2. start a container running `sleep infinity`  (== docker-compose.yaml)
  3. either:
       - run the gold solution.sh  (--solution), OR
       - run a sequence of agent bash commands  (--cmds "cmd1" "cmd2" ...), OR
       - run nothing (to prove the oracle FAILS without a solution)
  4. copy run-tests.sh + tests/ to /tests, run `bash /tests/run-tests.sh`
     with TEST_DIR=/tests  (== harness oracle invocation)
  5. parse pytest output -> pass/fail

This is the RLVP rollout primitive: start env, apply agent actions (bash
commands), score with the oracle at episode end.

Usage:
    python3 run_one.py hello-world --solution          # expect PASS
    python3 run_one.py hello-world                      # no soln -> expect FAIL
    python3 run_one.py hello-world --cmds 'echo "Hello, world!" > hello.txt'
    python3 run_one.py hello-world --solution --keep    # don't remove image

Returns exit 0 on oracle PASS, 1 on FAIL.
"""
import argparse
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from load_tasks import load_task


def sh(cmd, **kw):
    """Run a command list, return CompletedProcess (captures output)."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def run_task(task_id, solution=False, cmds=None, keep_image=False, verbose=True):
    t = load_task(task_id)
    task_dir = Path(t["dir"])
    tag = f"tbench-{task_id}:rlvp"
    name = f"tbench-run-{task_id}-{uuid.uuid4().hex[:8]}"

    metrics = {}

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # ---- 1. build ---------------------------------------------------------
    log(f"[build] {tag} from {t['dockerfile']}")
    t0 = time.time()
    b = sh(["docker", "build", "-t", tag, "-f", t["dockerfile"], str(task_dir)])
    metrics["build_sec"] = round(time.time() - t0, 2)
    if b.returncode != 0:
        print(b.stdout[-2000:], b.stderr[-2000:])
        raise RuntimeError("docker build failed")
    size = sh(["docker", "image", "inspect", tag, "--format", "{{.Size}}"]).stdout.strip()
    metrics["image_mb"] = round(int(size) / 1024 / 1024, 1) if size.isdigit() else None

    try:
        # ---- 2. start container ------------------------------------------
        t0 = time.time()
        r = sh(["docker", "run", "-d", "--rm", "--name", name, "-w", "/app",
                tag, "sh", "-c", "sleep infinity"])
        metrics["start_sec"] = round(time.time() - t0, 2)
        if r.returncode != 0:
            print(r.stderr)
            raise RuntimeError("docker run failed")

        # ---- 3. apply agent actions / solution ---------------------------
        if solution:
            sol = Path(t["solution"])
            if sol.suffix != ".sh":
                raise RuntimeError(f"--solution only supports .sh (got {sol.name}); "
                                   "use --cmds for yaml-solution tasks")
            sh(["docker", "exec", name, "mkdir", "-p", "/oracle"])
            sh(["docker", "cp", str(sol), f"{name}:/oracle/solution.sh"])
            log("[solution] bash /oracle/solution.sh")
            sh(["docker", "exec", "-w", "/app", name, "bash", "/oracle/solution.sh"])
        elif cmds:
            # RL action interface: each agent action is one bash command,
            # executed in the container; stdout/stderr is the observation.
            for c in cmds:
                ex = sh(["docker", "exec", "-w", "/app", name, "bash", "-lc", c])
                log(f"[action] $ {c}")
                if ex.stdout:
                    log(ex.stdout.rstrip())
                if ex.stderr:
                    log(ex.stderr.rstrip())
        else:
            log("[no-op] running oracle with NO solution (expect FAIL)")

        # ---- 4. run oracle ------------------------------------------------
        sh(["docker", "exec", name, "mkdir", "-p", "/tests"])
        sh(["docker", "cp", t["run_tests"], f"{name}:/tests/run-tests.sh"])
        if Path(t["tests_dir"]).exists():
            sh(["docker", "cp", t["tests_dir"] + "/.", f"{name}:/tests/"])
        log("[oracle] bash /tests/run-tests.sh  (TEST_DIR=/tests)")
        t0 = time.time()
        o = sh(["docker", "exec", "-e", "TEST_DIR=/tests", name,
                "bash", "/tests/run-tests.sh"])
        metrics["oracle_sec"] = round(time.time() - t0, 2)
        out = o.stdout + "\n" + o.stderr

        # ---- 5. parse pytest summary -------------------------------------
        m = re.search(r"=+ (\d+) (?:passed|failed)", out)
        passed = bool(re.search(r"=+ \d+ passed", out)) and "failed" not in (
            re.search(r"=+ .*? in [\d.]+s =+", out) or [""])[0].lower() \
            if False else None
        # simpler robust parse:
        n_fail = re.search(r"(\d+) failed", out)
        n_pass = re.search(r"(\d+) passed", out)
        n_fail = int(n_fail.group(1)) if n_fail else 0
        n_pass = int(n_pass.group(1)) if n_pass else 0
        success = (n_fail == 0 and n_pass > 0)

        if verbose:
            tail = "\n".join(out.splitlines()[-6:])
            log("[oracle output tail]\n" + tail)

        return {
            "task_id": task_id, "success": success,
            "passed": n_pass, "failed": n_fail, "metrics": metrics,
        }
    finally:
        sh(["docker", "rm", "-f", name])  # always clean container
        if not keep_image:
            sh(["docker", "rmi", "-f", tag])
            log(f"[cleanup] removed image {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--solution", action="store_true", help="run gold solution.sh")
    ap.add_argument("--cmds", nargs="*", help="agent bash commands to run in order")
    ap.add_argument("--keep", action="store_true", help="keep image after run")
    args = ap.parse_args()

    res = run_task(args.task_id, solution=args.solution, cmds=args.cmds,
                   keep_image=args.keep)
    print("\n=== RESULT ===")
    print(f"task={res['task_id']}  SUCCESS={res['success']}  "
          f"(passed={res['passed']} failed={res['failed']})")
    print(f"cost: {res['metrics']}")
    sys.exit(0 if res["success"] else 1)
