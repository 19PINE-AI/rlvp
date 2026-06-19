"""Set up and verify ONE SWE-Gym/SWE-bench instance via the no-Docker venv path.

Proves the verifiable oracle for a single task:
  1. clone repo at base_commit (shallow, from GitHub)
  2. create a venv and pip install the repo (editable) + test deps
  3. apply test_patch, run FAIL_TO_PASS test -> expect FAIL
  4. apply gold `patch`, run FAIL_TO_PASS test -> expect PASS

This avoids the heavy official SWE-bench Docker images (multi-GB conda images,
and the SWE-Gym repos are not in the upstream swebench spec map anyway).

Usage:
    python run_one.py dask__dask-8597 [--workdir /tmp/swegym_work] [--keep]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from load_tasks import get_instance

GITHUB = "https://github.com/{repo}.git"

# pip extras to install so a task's real test dependencies are present.
INSTALL_EXTRAS = {
    "dask/dask": "[array,dataframe,test]",
}

# Version pins per (repo, version) so contemporaneous deps are used. SWE-bench's
# own specs encode these; we hardcode the few we need for the single-task proof.
PIN_DEPS = {
    ("dask/dask", "2022.01"): [
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scipy==1.7.3",
    ],
}


def run(cmd, cwd=None, env=None, check=True, capture=True):
    print(f"  $ {cmd}  (cwd={cwd})", flush=True)
    r = subprocess.run(
        cmd, cwd=cwd, env=env, shell=True,
        capture_output=capture, text=True,
    )
    if capture and r.stdout:
        print(r.stdout[-3000:])
    if capture and r.stderr:
        print(r.stderr[-3000:])
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def apply_patch(repo_dir, patch_text, label):
    pf = os.path.join(repo_dir, f"_{label}.patch")
    with open(pf, "w") as f:
        f.write(patch_text)
    # git apply handles the a/ b/ prefixes in these diffs.
    run(f"git apply -v {pf}", cwd=repo_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_id")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true", help="keep workdir")
    args = ap.parse_args()

    inst = get_instance(args.instance_id)
    repo = inst["repo"]
    base = inst["base_commit"]
    f2p = inst["FAIL_TO_PASS"]
    print(f"Instance:   {inst['instance_id']}")
    print(f"Repo:       {repo} @ {base}")
    print(f"Version:    {inst['version']}")
    print(f"FAIL_TO_PASS: {f2p}")

    workdir = args.workdir or tempfile.mkdtemp(prefix="swegym_")
    os.makedirs(workdir, exist_ok=True)
    repo_dir = os.path.join(workdir, repo.split("/")[-1])
    print(f"Workdir:    {workdir}")

    try:
        # 1. clone at base_commit (fetch just that commit)
        if not os.path.isdir(repo_dir):
            run(f"git init {repo_dir}")
            run(f"git remote add origin {GITHUB.format(repo=repo)}", cwd=repo_dir)
            run(f"git fetch --depth 1 origin {base}", cwd=repo_dir)
            run(f"git checkout {base}", cwd=repo_dir)

        # 2. venv + install. Use `virtualenv` (bundles pip) rather than
        # `python3 -m venv`, since ensurepip is not available system-wide here.
        venv = os.path.join(workdir, "venv")
        if not os.path.isdir(venv):
            run(f"python3 -m virtualenv {venv}")
        py = os.path.join(venv, "bin", "python")
        pip = os.path.join(venv, "bin", "pip")
        run(f"{pip} install -U pip setuptools wheel -q")
        # install the package itself (editable) + pytest. Many repos gate
        # subpackage deps behind extras; install a reasonable extras set so the
        # target test's real dependencies (e.g. numpy for dask.array) are present.
        # Pin contemporaneous deps BEFORE the editable install so the resolver
        # keeps them (avoids newer numpy/pandas that break old code).
        pins = PIN_DEPS.get((repo, inst["version"]), [])
        if pins:
            run(f"{pip} install -q " + " ".join(f"'{p}'" for p in pins))
        extras = INSTALL_EXTRAS.get(repo, "")
        run(f"{pip} install -e '.{extras}' -q", cwd=repo_dir)
        run(f"{pip} install pytest -q")

        test_id = f2p[0]
        test_cmd = f"{py} -m pytest -p no:cacheprovider -q -rN '{test_id}'"

        # 3. apply test_patch, run -> expect FAIL
        apply_patch(repo_dir, inst["test_patch"], "testpatch")
        print("\n=== STEP A: run FAIL_TO_PASS BEFORE gold patch (expect FAIL) ===")
        r_before = run(test_cmd, cwd=repo_dir, check=False)
        before_failed = r_before.returncode != 0

        # 4. apply gold patch, run -> expect PASS
        apply_patch(repo_dir, inst["patch"], "goldpatch")
        print("\n=== STEP B: run FAIL_TO_PASS AFTER gold patch (expect PASS) ===")
        r_after = run(test_cmd, cwd=repo_dir, check=False)
        after_passed = r_after.returncode == 0

        print("\n========== ORACLE RESULT ==========")
        print(f"  before gold patch -> {'FAIL' if before_failed else 'PASS'}")
        print(f"  after  gold patch -> {'PASS' if after_passed else 'FAIL'}")
        ok = before_failed and after_passed
        print(f"  ORACLE VERIFIED: {ok}")
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"Cleaned up {workdir}")
        else:
            print(f"Kept workdir at {workdir}")


if __name__ == "__main__":
    sys.exit(main())
