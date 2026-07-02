"""Run the SWE-smith arms IN PARALLEL with the ET arms (the GPU is verifier/
Docker-bound and 96% idle, so two 8B trainers coexist easily). Writes the same
/tmp/on_<tag>.done markers the main orchestrator's Phase C checks, so it SKIPS
these -> no double-run. Serializes the SWE arms among themselves (one at a time),
each gated only on there being >=30G GPU free (runs alongside ET's ~27G)."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/rlvp")
LOG = open("/tmp/swe_parallel.log", "a")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", file=LOG, flush=True)


def gpu_free():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, t = map(int, r.stdout.strip().split(","))
    return t - u


def main():
    jobs = json.loads(Path("/tmp/swesmith_jobs.json").read_text())
    log(f"=== swesmith parallel start ({len(jobs)} jobs) ===")
    for j in jobs:
        tag, cmd = j["tag"], j["cmd"]
        done = Path(f"/tmp/on_{tag}.done")
        if done.exists():
            log(f"SKIP {tag}"); continue
        while gpu_free() < 30000:      # room to sit beside ET (~27G)
            time.sleep(120)
        log(f"LAUNCH {tag}")
        t0 = time.time()
        with open(f"/tmp/on_{tag}.out", "w") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                env=os.environ).returncode
        if rc == 0:
            done.write_text("ok"); log(f"DONE {tag} ({round((time.time()-t0)/60,1)}min)")
        else:
            log(f"FAIL {tag} rc={rc}")
    log("=== swesmith parallel ALLDONE ===")
    Path("/tmp/swe_parallel.alldone").write_text("ok")


if __name__ == "__main__":
    main()
