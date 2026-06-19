"""Reusable setup-and-verify harness for SWE-Gym dask/dask instances.

Builds an isolated checkout at base_commit (from a cached bare clone via
`git worktree`), a SHARED virtualenv per dask-version pin-group (numpy/pandas
era), applies the test_patch, and runs FAIL_TO_PASS / PASS_TO_PASS.

Public API:
    setup_and_verify(instance, workdir, run_gold=True, run_p2p=True) -> dict

Design notes (frugality + RL-rollout cost):
  * ONE cached bare clone at cache/dask.git (clone once, ~131 MB).
  * Per-task source tree is a `git worktree` (cheap, shares the object store).
  * venvs are SHARED across instances that map to the same pin-group, kept under
    cache/venvs/<group>. The editable install points at a per-group "anchor"
    checkout so `pip install -e .` resolves; per-task worktrees run the tests
    against their own source by prepending their dir to sys.path via PYTHONPATH
    -- but dask is import-by-package, so we instead install the package into the
    shared venv NON-editable per task is too slow. Simpler & correct: we make the
    shared venv hold only the third-party deps + pytest, and each task installs
    dask EDITABLE into the shared venv right before its run. Because pip editable
    just writes an egg-link/path, switching the editable target is fast.

    To keep tasks isolated when sharing a venv we serialize per venv: the caller
    runs tasks one at a time per group (fine for verification and for RL rollout
    batching by group). The editable target is repointed per task.

The pin-groups generalise the dask numpy<2 problem found earlier: dask released
before numpy 2.0 (June 2024) cannot import under numpy>=2 (removed
numpy.compat.basestring etc.), so every version here pins numpy<2 plus a
contemporaneous pandas/scipy.
"""
import json
import os
import re
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
BARE = os.path.join(CACHE, "dask.git")
VENV_ROOT = os.path.join(CACHE, "venvs")
ANCHOR_ROOT = os.path.join(CACHE, "anchors")

GITHUB = "https://github.com/dask/dask.git"
EXTRAS = "[array,dataframe,test]"

# --- pin-group selection -------------------------------------------------
# Map a dask `version` string to a pin-group name + the third-party pins.
# Eras chosen from dask's own contemporaneous numpy/pandas support matrix.
# All pre-2.0-numpy because all these dask versions predate numpy 2.0.
PIN_GROUPS = {
    # group_name: list of pip requirement strings.
    # NOTE: this box runs CPython 3.10, whose earliest available binary wheels
    # are numpy 1.21.2 / pandas 1.3.3 / scipy 1.7.2 (older versions have no
    # cp310 wheels and fail to build from source). So pre-2021 dask is pinned
    # to the 1.21.6 floor rather than its true-contemporaneous numpy<=1.20 --
    # dask imports fine under numpy 1.21, and avoiding source builds is what
    # keeps setup fast + reliable.
    "era_2020": ["numpy==1.21.6", "pandas==1.3.5", "scipy==1.7.3"],
    "era_2021a": ["numpy==1.21.6", "pandas==1.3.5", "scipy==1.7.3"],
    "era_2021b": ["numpy==1.21.6", "pandas==1.3.5", "scipy==1.7.3"],
    "era_2022a": ["numpy==1.21.6", "pandas==1.4.4", "scipy==1.7.3"],
    "era_2022b": ["numpy==1.22.4", "pandas==1.4.4", "scipy==1.8.1"],
    "era_2023": ["numpy==1.24.4", "pandas==1.5.3", "scipy==1.10.1"],
    "era_2024": ["numpy==1.26.4", "pandas==2.2.2", "scipy==1.13.1"],
}


def _ver_tuple(v):
    """Normalise a dask version string to a sortable (year, minor) tuple.

    dask versions are either calendar 'YYYY.MM' (2021.07, 2022.6) or the old
    '2.25', '2.30' scheme (which are 2020-era)."""
    parts = v.split(".")
    try:
        major = int(parts[0])
    except ValueError:
        return (0, 0)
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


def venv_name_for(group):
    """Identical pin lists share one venv (keyed by pin content)."""
    pins = PIN_GROUPS[group]
    for g in sorted(PIN_GROUPS):          # first group with the same pins wins
        if PIN_GROUPS[g] == pins:
            return g
    return group


def pin_group_for(version):
    major, minor = _ver_tuple(version)
    if major < 2000:        # old '2.25'/'2.30' scheme -> 2020 era
        return "era_2020"
    if major == 2020:
        return "era_2020"
    if major == 2021:
        return "era_2021a" if minor <= 5 else "era_2021b"
    if major == 2022:
        return "era_2022a" if minor <= 2 else "era_2022b"
    if major == 2023:
        return "era_2023"
    return "era_2024"       # 2024+


# --- shell helper --------------------------------------------------------
def _run(cmd, cwd=None, env=None, check=True, timeout=900, log=None):
    r = subprocess.run(
        cmd, cwd=cwd, env=env, shell=True,
        capture_output=True, text=True, timeout=timeout,
    )
    if log is not None:
        log.append((cmd, r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    if check and r.returncode != 0:
        raise RuntimeError(
            f"cmd failed ({r.returncode}): {cmd}\n"
            f"STDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}"
        )
    return r


# --- bare clone + worktree ----------------------------------------------
def ensure_bare_clone():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.isdir(BARE):
        _run(f"git clone --bare {GITHUB} {BARE}", timeout=1200)
    return BARE


def make_worktree(base_commit, dest):
    """Create a detached worktree of the bare clone at base_commit."""
    ensure_bare_clone()
    if os.path.isdir(dest):
        # remove a stale worktree registration + dir
        _run(f"git -C {BARE} worktree remove --force {dest}", check=False)
        shutil.rmtree(dest, ignore_errors=True)
    # the commit must exist in the bare clone; full clone has all history.
    _run(f"git -C {BARE} worktree add --detach --force {dest} {base_commit}",
         timeout=300)
    return dest


# --- shared venv per pin-group ------------------------------------------
def _constraints_path(group):
    return os.path.join(VENV_ROOT, group, "constraints.txt")


def _venv_dir_name(group, slot=None):
    """Resolved venv dir name. With slot!=None each concurrent episode gets its
    OWN venv copy (`<group>__s<slot>`) so its editable dask target is private --
    a shared editable target is global state that corrupts under concurrency."""
    g = venv_name_for(group)
    return g if slot is None else f"{g}__s{slot}"


def ensure_group_venv(group, log=None, slot=None):
    """Create (once) a shared venv for a pin-group with the pinned
    numpy/pandas/scipy + pytest installed. A constraints.txt holding those
    pins is written so the per-task editable install pulls dask's OTHER runtime
    deps (pyyaml, toolz, ...) without bumping numpy/pandas/scipy.
    With slot!=None, builds/returns a per-slot venv for concurrency isolation.
    Returns (python_path, pip_path)."""
    os.makedirs(VENV_ROOT, exist_ok=True)
    pins_group = venv_name_for(group)
    group = _venv_dir_name(group, slot)
    PIN_GROUPS[group] = PIN_GROUPS[pins_group]  # alias pins for the slot dir
    venv = os.path.join(VENV_ROOT, group)
    py = os.path.join(venv, "bin", "python")
    pip = os.path.join(venv, "bin", "pip")
    marker = os.path.join(venv, ".deps_ready")
    pins = PIN_GROUPS[group]
    if os.path.exists(marker):
        return py, pip
    if not os.path.isdir(venv):
        _run(f"python3 -m virtualenv {venv}", log=log, timeout=300)
    _run(f"{pip} install -U pip setuptools wheel -q", log=log, timeout=600)
    _run(f"{pip} install -q " + " ".join(f"'{p}'" for p in pins),
         log=log, timeout=900)
    _run(f"{pip} install -q pytest pytest-timeout", log=log, timeout=600)
    with open(_constraints_path(group), "w") as f:
        f.write("\n".join(pins) + "\n")
    with open(marker, "w") as f:
        f.write("ok\n")
    return py, pip


def install_editable(pip, repo_dir, group, log=None, slot=None):
    """Point this venv's dask at repo_dir (editable), pulling dask's
    runtime deps but holding numpy/pandas/scipy via the group constraints."""
    cons = _constraints_path(_venv_dir_name(group, slot))
    _run(f"{pip} install -e '.{EXTRAS}' -q -c {cons}",
         cwd=repo_dir, log=log, timeout=1200)


# --- patch application ---------------------------------------------------
def _apply_patch(repo_dir, patch_text, label, log=None):
    pf = os.path.join(repo_dir, f"_{label}.patch")
    with open(pf, "w") as f:
        f.write(patch_text)
    _run(f"git apply -v {pf}", cwd=repo_dir, log=log)


def _revert_patch(repo_dir, patch_text, label, log=None):
    pf = os.path.join(repo_dir, f"_{label}.patch")
    _run(f"git apply -R {pf}", cwd=repo_dir, log=log, check=False)


def _node_ids(tests):
    """tests may be a list already, or a JSON-ish string."""
    if isinstance(tests, list):
        return tests
    if isinstance(tests, str):
        try:
            return json.loads(tests)
        except Exception:
            return [tests]
    return list(tests)


def _pytest_cmd(py, node_ids, lenient=False):
    quoted = " ".join(f"'{n}'" for n in node_ids)
    extra = "--continue-on-collection-errors " if lenient else ""
    # Old dask (2020/early-2021) sets `filterwarnings = error:::dask.*` in
    # setup.cfg, which turns the py3.10 distutils `LooseVersion` DeprecationWarning
    # (raised at import from dask/compatibility.py) into a hard error. We run on
    # py3.10 (newer than these versions targeted), so override that one warning
    # class to non-fatal. `-W` flags take precedence over the ini `filterwarnings`.
    # This does NOT mask test logic -- these F2P tests assert on values, not on
    # the presence of DeprecationWarnings.
    warn = ("-W 'ignore::DeprecationWarning' "
            "-W 'ignore::ImportWarning' ")
    return (f"{py} -m pytest -p no:cacheprovider -q -rN "
            f"--timeout=300 -o addopts='' {warn}{extra}{quoted}")


_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|errors)")


def _parse_pytest(stdout):
    """Return (n_passed, n_failed) from a pytest -q summary line."""
    n_pass = n_fail = 0
    for m in re.finditer(r"(\d+)\s+(passed|failed|errors?)", stdout):
        n = int(m.group(1))
        if m.group(2) == "passed":
            n_pass = n
        else:
            n_fail += n
    return n_pass, n_fail


# --- main entry point ----------------------------------------------------
def setup_and_verify(instance, workdir, run_gold=True, run_p2p=True,
                     p2p_limit=50):
    """Set up an isolated env for one dask instance and verify the oracle.

    Returns a dict with: instance_id, version, group, setup_ok,
    f2p_fail_on_base, f2p_pass_on_gold, p2p_pass_on_gold, timing{...}, error.
    """
    iid = instance["instance_id"]
    base = instance["base_commit"]
    version = instance["version"]
    group = pin_group_for(version)
    f2p = _node_ids(instance["FAIL_TO_PASS"])
    p2p = _node_ids(instance["PASS_TO_PASS"])
    log = []
    t = {}
    res = {
        "instance_id": iid, "version": version, "group": group,
        "setup_ok": False, "f2p_fail_on_base": None,
        "f2p_pass_on_gold": None, "p2p_pass_on_gold": None,
        "timing": t, "error": None,
    }
    os.makedirs(workdir, exist_ok=True)
    repo_dir = os.path.join(workdir, "dask")
    try:
        # 1. worktree at base_commit
        t0 = time.time()
        make_worktree(base, repo_dir)
        t["worktree"] = round(time.time() - t0, 1)

        # 2. shared venv for the pin-group + editable re-point
        t0 = time.time()
        py, pip = ensure_group_venv(group, log=log)
        t["venv"] = round(time.time() - t0, 1)
        t0 = time.time()
        install_editable(pip, repo_dir, group, log=log)
        t["editable"] = round(time.time() - t0, 1)
        res["setup_ok"] = True

        # 3. apply test_patch
        _apply_patch(repo_dir, instance["test_patch"], "testpatch", log=log)

        # 4a. F2P on base (expect FAIL)
        t0 = time.time()
        r = _run(_pytest_cmd(py, f2p), cwd=repo_dir, check=False, log=log,
                 timeout=600)
        t["f2p_base"] = round(time.time() - t0, 1)
        res["f2p_fail_on_base"] = (r.returncode != 0)

        if run_gold:
            # 4b. apply gold patch, F2P (expect PASS)
            _apply_patch(repo_dir, instance["patch"], "goldpatch", log=log)
            t0 = time.time()
            r = _run(_pytest_cmd(py, f2p), cwd=repo_dir, check=False, log=log,
                     timeout=600)
            t["f2p_gold"] = round(time.time() - t0, 1)
            res["f2p_pass_on_gold"] = (r.returncode == 0)

            if run_p2p and p2p:
                sample = p2p[:p2p_limit]
                t0 = time.time()
                r = _run(_pytest_cmd(py, sample, lenient=True), cwd=repo_dir,
                         check=False, log=log, timeout=900)
                t["p2p_gold"] = round(time.time() - t0, 1)
                n_pass, n_fail = _parse_pytest(r.stdout)
                res["p2p_n_selected"] = len(sample)
                res["p2p_passed"] = n_pass
                res["p2p_failed"] = n_fail
                # "pass" = no genuine failures among collectable P2P tests
                # (uncollectable/renamed node-ids are tolerated, not counted).
                res["p2p_pass_on_gold"] = (n_fail == 0 and n_pass > 0)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        # detach worktree registration so the bare repo stays clean
        _run(f"git -C {BARE} worktree remove --force {repo_dir}", check=False)
        res["_log_tail"] = log[-3:]
    return res


def cleanup_worktree(workdir):
    repo_dir = os.path.join(workdir, "dask")
    _run(f"git -C {BARE} worktree remove --force {repo_dir}", check=False)
    shutil.rmtree(workdir, ignore_errors=True)
