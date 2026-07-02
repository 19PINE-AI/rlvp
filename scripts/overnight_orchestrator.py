"""Autonomous overnight orchestrator for the benchmark-pivot experiments.

Idempotent + resumable (skips work whose .done marker exists). Serializes on the
GPU purely by sustained-free-memory, so it naturally waits behind the p0 AdamW
queue and probe_tb30b without tracking their PIDs. Launches, in order:

  Phase A  ET capability probe (Qwen3-8B) -> results/probe_et8b (via benchmarks harness)
  Phase B  ET P1 harm arms: endless_train.py {outcome, c3} x seeds {7,11,12}, 8B
  Phase C  SWE-smith arms IF /tmp/swesmith_ready exists (spec at /tmp/swesmith_jobs.json)

Each training run is Docker/CPU-bound (G concurrent containers), so we run ONE
GPU job at a time to avoid Docker + vLLM contention. Logs to /tmp/on_*.out and
writes /tmp/on_<tag>.done markers. Progress -> /tmp/rlvp_overnight.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/rlvp")
ET_DIR = Path("/home/ubuntu/benchmarks/endless-terminals")
NEED_MIB = 60000          # wait behind any 30B job (p0 AdamW ~55G, probe_tb30b ~45G)
                          # so our 8B Docker-heavy runs never contend with them
PROBE_JSON = ET_DIR / "results" / "capability_Qwen3-8B.json"
LOG = open("/tmp/rlvp_overnight.log", "a")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=LOG, flush=True)
    print(msg, flush=True)


def gpu_free():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total = map(int, r.stdout.strip().split(","))
    return total - used


def wait_gpu(need=NEED_MIB, sustain=5, period=120):
    """Block until GPU has >= need MiB free for `sustain` consecutive checks."""
    streak = 0
    while True:
        free = gpu_free()
        streak = streak + 1 if free >= need else 0
        if streak >= sustain:
            log(f"GPU free {free}MiB (sustained) -> proceed")
            return
        time.sleep(period)


def run(tag, cmd, cwd=ROOT, env=None):
    done = Path(f"/tmp/on_{tag}.done")
    if done.exists():
        log(f"SKIP {tag} (done)")
        return True
    wait_gpu()
    log(f"LAUNCH {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with open(f"/tmp/on_{tag}.out", "w") as f:
        rc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT,
                            env=env or os.environ).returncode
    dt = round((time.time() - t0) / 60, 1)
    if rc == 0:
        done.write_text("ok")
        log(f"DONE {tag} rc=0 ({dt}min)")
        return True
    log(f"FAIL {tag} rc={rc} ({dt}min)")
    return False


def main():
    log("=== overnight orchestrator start ===")

    # ---- Phase A: ET capability probe (8B) ----
    e8 = dict(os.environ)
    run("et_probe8b",
        ["python3", str(ET_DIR / "et_capability_probe.py"),
         "--model", "Qwen/Qwen3-8B", "--g", "4", "--n-tasks", "40", "--no-wait"],
        cwd=ET_DIR, env=e8)
    band = None
    if PROBE_JSON.exists():
        band = json.loads(PROBE_JSON.read_text()).get("success_rate")
        log(f"ET capability success_rate={band}")

    # ---- Phase B: ET P1 harm arms (8B) ----
    # outcome-only vs c3 (harm penalty + discharges), 3 seeds. Docker/CPU-bound.
    for seed in (7, 11, 12):
        for credit in ("outcome", "c3"):
            tag = f"et_{credit}_s{seed}"
            out = f"run_et_{credit}_s{seed}"
            run(tag,
                ["python3", "scripts/endless_train.py", "25",
                 "--model", "Qwen/Qwen3-8B", "--credit", credit,
                 "--rule-mode", "structural", "--seed", str(seed),
                 "--n-tasks", "40", "--out", out])

    # ---- Phase C: SWE-smith arms (only if a validated adapter enabled it) ----
    ready = Path("/tmp/swesmith_ready")
    spec = Path("/tmp/swesmith_jobs.json")
    if ready.exists() and spec.exists():
        for job in json.loads(spec.read_text()):
            run(job["tag"], job["cmd"])
    else:
        log("SWE-smith not enabled (no /tmp/swesmith_ready) -> skipping Phase C")

    log("=== overnight orchestrator ALLDONE ===")
    Path("/tmp/rlvp_overnight.alldone").write_text("ok")


if __name__ == "__main__":
    main()
