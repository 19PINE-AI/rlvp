#!/usr/bin/env python3
"""Endless Terminals episode runner via direct Docker (no Apptainer/Harbor).

Mirrors rlvp/termbench_adapter semantics: build the task image, start a
persistent container, apply agent bash commands (or the gold solve.sh),
then copy tests/ and run test.sh which writes /logs/verifier/reward.txt.

Modes:
  --gold          run solution/solve.sh (oracle validation; expect reward 1)
  --empty         run nothing           (expect reward 0)
  --cmds "c1" ..  run a sequence of bash commands (agent rollout primitive)

Usage:
  python3 et_runner.py slice/task_000000_003d339f --gold
  python3 et_runner.py slice/task_000000_003d339f --empty
"""
import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def run_task(task_dir, mode, cmds=None, keep=False, verbose=True):
    task_dir = Path(task_dir).resolve()
    tid = task_dir.name
    tag = f"et-{tid.lower()}:pilot"
    name = f"et-run-{tid}-{uuid.uuid4().hex[:8]}"
    env_dir = task_dir / "environment"
    metrics = {"task": tid}

    def log(*a):
        if verbose:
            print(*a, flush=True)

    t0 = time.time()
    b = sh(["docker", "build", "-t", tag, "-f", str(env_dir / "Dockerfile"),
            str(env_dir)])
    metrics["build_sec"] = round(time.time() - t0, 1)
    if b.returncode != 0:
        print(b.stdout[-1500:], b.stderr[-1500:])
        return {**metrics, "error": "build_failed"}
    size = sh(["docker", "image", "inspect", tag, "--format", "{{.Size}}"]).stdout.strip()
    metrics["image_mb"] = round(int(size) / 1e6, 1) if size.isdigit() else None

    try:
        r = sh(["docker", "run", "-d", "--rm", "--name", name, "-w", "/home/user",
                tag, "sh", "-c", "sleep infinity"])
        if r.returncode != 0:
            return {**metrics, "error": f"run_failed: {r.stderr[:200]}"}

        def dexec(bash, timeout=300):
            return sh(["docker", "exec", name, "bash", "-lc", bash], timeout=timeout)

        actions = []
        if mode == "gold":
            gold = (task_dir / "solution" / "solve.sh").read_text()
            dexec("cat > /tmp/solve.sh << 'ET_EOF'\n" + gold + "\nET_EOF")
            res = dexec("bash /tmp/solve.sh")
            actions.append(("gold", res.returncode))
        elif mode == "cmds":
            for c in (cmds or []):
                res = dexec(c)
                actions.append((c[:60], res.returncode))
        # mode == "empty": do nothing

        # score: copy tests, run test.sh, read reward
        dexec("mkdir -p /tests /logs/verifier")
        sh(["docker", "cp", str(task_dir / "tests") + "/.", f"{name}:/tests/"])
        ts0 = time.time()
        tres = dexec("bash /tests/test.sh", timeout=400)
        metrics["test_sec"] = round(time.time() - ts0, 1)
        reward = dexec("cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING").stdout.strip()
        metrics["reward"] = reward
        metrics["actions"] = actions
        if reward not in ("0", "1"):
            metrics["test_tail"] = tres.stdout[-500:]
        log(f"[{tid}] mode={mode} image={metrics.get('image_mb')}MB "
            f"build={metrics['build_sec']}s test={metrics['test_sec']}s -> reward={reward}")
        return metrics
    finally:
        sh(["docker", "rm", "-f", name])
        if not keep:
            sh(["docker", "image", "rm", "-f", tag])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--empty", action="store_true")
    ap.add_argument("--cmds", nargs="*", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    mode = "gold" if args.gold else "empty" if args.empty else "cmds"
    m = run_task(args.task_dir, mode, cmds=args.cmds, keep=args.keep)
    import json
    print(json.dumps(m, indent=1))
    sys.exit(0 if m.get("reward") in ("0", "1") else 2)
