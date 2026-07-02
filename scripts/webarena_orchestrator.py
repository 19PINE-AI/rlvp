"""WebArena/ST-WebAgentBench training orchestrator (Phase D, separate process).

Runs strictly AFTER the main ET+SWE-smith orchestrator finishes
(/tmp/rlvp_overnight.alldone) so it never contends with it on the GPU, and only
if the combined torch+browsergym env is ready (/tmp/webarena_ready). Trains the
CuP arms (outcome vs c3) via the wa_venv python with the confirmed site env vars.

Idempotent via /tmp/onwa_<tag>.done. Logs to /tmp/webarena_orch.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/rlvp")
VENV_PY = "/home/ubuntu/benchmarks/webarena/wa_venv/bin/python"
NEED_MIB = 30000 if os.environ.get("WA_PARALLEL") else 60000  # parallel: sit beside ET+SWE
LOG = open("/tmp/webarena_orch.log", "a")

# Site env vars confirmed working in the live validation (see benchmarks/webarena/env.sh).
SITE_ENV = {
    "PYTHONPATH": "/home/ubuntu/benchmarks/webarena/ST-WebAgentBench:/home/ubuntu/rlvp",
    "SHOPPING_ADMIN": "http://localhost:7780/admin", "WA_SHOPPING_ADMIN": "http://localhost:7780/admin",
    "GITLAB": "http://localhost:8023", "WA_GITLAB": "http://localhost:8023",
    "SUITECRM": "http://localhost:8080", "WA_SUITECRM": "http://localhost:8080",
    "SHOPPING": "", "REDDIT": "", "WIKIPEDIA": "", "MAP": "", "HOMEPAGE": "", "IPA_HOME": "",
}
def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", file=LOG, flush=True)
    print(m, flush=True)


def _http_ok(url):
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                       capture_output=True, text=True)
    return r.stdout.strip() in ("200", "302")


def select_task_ids():
    """Task ids (comma-joined) for whichever ST-WAB sites are currently healthy;
    vision tasks 295-334 already excluded when /tmp/wa_taskids.json was built."""
    by = json.loads(Path("/tmp/wa_taskids.json").read_text())
    ids = []
    if _http_ok("http://localhost:7780/admin"):
        ids += by.get("shopping_admin", [])
    if _http_ok("http://localhost:8023"):
        ids += by.get("gitlab", [])
    if _http_ok("http://localhost:8080/public"):
        ids += by.get("suitecrm", [])
    return ",".join(str(i) for i in sorted(ids))


def gpu_free():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, t = map(int, r.stdout.strip().split(","))
    return t - u


def wait(cond, period=120):
    while not cond():
        time.sleep(period)


def main():
    log("=== webarena orchestrator start ===")
    if not Path("/tmp/webarena_ready").exists():
        log("no /tmp/webarena_ready (combined env unverified) -> exit")
        return
    if os.environ.get("WA_PARALLEL"):
        log("WA_PARALLEL set -> running in parallel with the ET/SWE tracks")
    else:
        log("waiting for main overnight run to finish (rlvp_overnight.alldone) ...")
        wait(lambda: Path("/tmp/rlvp_overnight.alldone").exists())
    env = {**os.environ, **SITE_ENV}
    task_ids = select_task_ids()
    if not task_ids:
        log("no ST-WAB sites healthy -> exit"); return
    log(f"training on {len(task_ids.split(','))} tasks: {task_ids[:80]}...")
    for seed in (7, 11):
        for credit in ("outcome", "c3"):
            tag = f"wa_{credit}_s{seed}"
            done = Path(f"/tmp/onwa_{tag}.done")
            if done.exists():
                log(f"SKIP {tag}"); continue
            streak = 0
            while streak < 5:
                streak = streak + 1 if gpu_free() >= NEED_MIB else 0
                time.sleep(60)
            log(f"LAUNCH {tag}")
            cmd = [VENV_PY, "scripts/webarena_train.py", "20", "--model", "Qwen/Qwen3-8B",
                   "--credit", credit, "--seed", str(seed), "--task-ids", task_ids,
                   "--out", f"run_{tag}"]
            with open(f"/tmp/onwa_{tag}.out", "w") as f:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                    env=env).returncode
            if rc == 0:
                done.write_text("ok"); log(f"DONE {tag}")
            else:
                log(f"FAIL {tag} rc={rc}")
    log("=== webarena orchestrator ALLDONE ===")
    Path("/tmp/webarena_orch.alldone").write_text("ok")


if __name__ == "__main__":
    main()
