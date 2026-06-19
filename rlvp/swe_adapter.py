"""SWE-bench / SWE-Gym (dask) <-> RLVP adapter.

Runs a single SWE episode with OUR HF policy editing a REAL dask worktree to fix
a REAL bug, with the hidden FAIL_TO_PASS / PASS_TO_PASS pytest suite as the
terminal oracle reward and verifiable per-step process signals.

Token bookkeeping is identical to tau2_adapter.run_one_sim so the trainer
(grpo.build_advantages / update_policy) is unchanged: each generated turn is a
span in episode.ids tagged with its turn index, with turn_violations /
turn_discharges accumulated incrementally and a ShimEnv carrying
success/violations/discharges/calls/outcome_reward.

Setup reuses the PROVEN harness in swegym/swe_env_setup.py:
  * one cached bare clone -> per-task `git worktree` at base_commit,
  * SHARED pinned venv per dask-era (numpy<2), dask installed editable into it,
  * test_patch applied (so the hidden tests exist on disk but the SOURCE fix
    does not -- the agent must write it).
We do NOT make a venv per episode; we clean up the worktree at episode end.

The agent's tool vocabulary mirrors fileops.py (read_file / list_dir /
write_file / run_tests / submit) but over the real repo + real pytest.
"""
from __future__ import annotations

import json
import os
import sys
import time
import re

# Make the swegym harness importable (it lives outside the rlvp package).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SWEGYM = os.path.join(os.path.dirname(_HERE), "swegym")
if _SWEGYM not in sys.path:
    sys.path.insert(0, _SWEGYM)

import swe_env_setup as H  # the proven harness (do NOT modify)

from .envs.base import parse_action
from .rollout import TEMPLATE, Episode, _ids
# Reuse the batched generation server + the trainer-facing env shim verbatim.
from .tau2_adapter import GenServer, ShimEnv  # noqa: F401  (re-exported)


# --------------------------------------------------------------------------
# Venv pool: each concurrent episode leases a PRIVATE venv copy so its editable
# dask target can't be re-pointed by another episode mid-run. Pool size caps
# true SWE rollout concurrency (default 6; first use of each slot builds it once,
# then cached). Without this, sharing one editable venv corrupts under threads.
# --------------------------------------------------------------------------
import queue as _queue

_VENV_SLOTS = int(os.environ.get("SWE_VENV_SLOTS", "6"))


class _SlotPool:
    def __init__(self, n):
        self._q = _queue.Queue()
        for i in range(n):
            self._q.put(i)

    def acquire(self):
        return self._q.get()

    def release(self, slot):
        self._q.put(slot)


_VENV_POOL = _SlotPool(_VENV_SLOTS)


# --------------------------------------------------------------------------
# Instance loading
# --------------------------------------------------------------------------
_INSTANCES_PATH = os.path.join(_SWEGYM, "cache", "dask_instances.json")
_CLEAN_PATH = os.path.join(_SWEGYM, "dask_clean_instances.json")


def load_clean_instances(n=None, ids=None):
    """Return up to n clean dask instance dicts (those whose oracle verified:
    F2P fails on base, F2P passes on gold). `ids` overrides the selection."""
    insts = {i["instance_id"]: i for i in json.load(open(_INSTANCES_PATH))}
    if ids is None:
        ids = json.load(open(_CLEAN_PATH))["clean_instance_ids"]
    out = [insts[i] for i in ids if i in insts]
    return out[:n] if n else out


def _node_ids(x):
    return H._node_ids(x)


def _is_test_path(path: str) -> bool:
    """A worktree-relative path that belongs to the (oracle) test suite."""
    p = (path or "").replace("\\", "/").lstrip("./")
    base = os.path.basename(p)
    return ("/tests/" in ("/" + p) or p.startswith("tests/")
            or base.startswith("test_") or base.endswith("_test.py")
            or "/test_" in ("/" + p))


# --------------------------------------------------------------------------
# Process-signal rule tracker (all verifiable, OUTCOME-INSTRUMENTAL for SWE)
# --------------------------------------------------------------------------
class SweRuleTracker:
    """Incremental per-turn violations/discharges for SWE rollouts.

    discharges (productive precursors of a correct fix):
      reproduced : ran the FAIL_TO_PASS test BEFORE editing any non-test file
                   (reproduce-before-patch -- you must SEE the failure first).
      ran_tests  : a run_tests call issued AFTER a source edit (verify the fix).
    violations (failure-guaranteeing / oracle-corrupting actions):
      untested_submit : submit after editing source without running tests since
                        the last edit (you can't know the fix works).
      edited_test_file: write_file to a tests/ or test_* path -- modifying the
                        hidden oracle. The reward runs the ORIGINAL F2P/P2P node
                        ids regardless, so this can only break PASS_TO_PASS; it
                        is always a wasted/harmful action.

    mode is accepted for parity with tau2's RuleTracker; the SWE signal is
    'structural' (these four), and 'none' disables tracking.
    """

    def __init__(self, mode="structural"):
        self.mode = mode
        self.ran_f2p = False             # ever ran the tests
        self.edited_source = False       # edited a non-test file
        self.reproduced_paid = False     # 'reproduced' discharge already given
        self.tests_since_edit = False    # ran tests after the last source edit
        self.turn_violations: dict = {}
        self.turn_discharges: dict = {}

    def observe_turn(self, turn_idx: int, call):
        """call is a parsed ToolCall (or None for an unparsable/respond turn)."""
        v, d = [], []
        if self.mode in (None, "none") or call is None:
            return v, d
        name = call.name
        a = call.args or {}
        path = str(a.get("path", ""))

        if name == "write_file":
            if _is_test_path(path):
                v.append("edited_test_file")          # never modify the oracle
            else:
                self.edited_source = True
                self.tests_since_edit = False
        elif name == "run_tests":
            # reproduce-before-patch: ran F2P before touching any source file
            if not self.reproduced_paid and not self.edited_source:
                self.reproduced_paid = True
                d.append("reproduced")
            self.ran_f2p = True
            if self.edited_source:
                d.append("ran_tests")                 # verified after a fix
                self.tests_since_edit = True
        elif name == "submit":
            if self.edited_source and not self.tests_since_edit:
                v.append("untested_submit")           # submit a fix you never ran

        if v:
            self.turn_violations[turn_idx] = self.turn_violations.get(turn_idx, []) + v
        if d:
            self.turn_discharges[turn_idx] = self.turn_discharges.get(turn_idx, []) + d
        return v, d


# --------------------------------------------------------------------------
# Worktree-backed SWE environment (real files + real pytest)
# --------------------------------------------------------------------------
TRUNC = 1500
P2P_SAMPLE = 2  # a couple of PASS_TO_PASS guards per run_tests (cost control)


class SweWorktree:
    """Owns one dask worktree at base_commit in a shared pinned venv, with the
    test_patch applied. Provides tool execution + the terminal oracle.

    Lifecycle: SweWorktree(instance, workdir).setup() ... .close()."""

    def __init__(self, instance, workdir):
        self.inst = instance
        self.workdir = workdir
        self.repo_dir = os.path.join(workdir, "dask")
        self.iid = instance["instance_id"]
        self.version = instance["version"]
        self.group = H.pin_group_for(self.version)
        self.f2p = _node_ids(instance["FAIL_TO_PASS"])
        self.p2p = _node_ids(instance["PASS_TO_PASS"])
        self.py = None
        self.slot = None
        self.setup_ok = False
        self.setup_error = None
        self._log: list = []

    # -- setup / teardown --------------------------------------------------
    def setup(self):
        os.makedirs(self.workdir, exist_ok=True)
        try:
            self.slot = _VENV_POOL.acquire()  # private venv -> editable target isolated
            H.make_worktree(self.inst["base_commit"], self.repo_dir)
            self.py, pip = H.ensure_group_venv(self.group, log=self._log,
                                               slot=self.slot)
            H.install_editable(pip, self.repo_dir, self.group, log=self._log,
                               slot=self.slot)
            # test_patch installs the hidden tests; the SOURCE fix is NOT applied.
            H._apply_patch(self.repo_dir, self.inst["test_patch"], "testpatch",
                           log=self._log)
            self.setup_ok = True
        except Exception as e:  # noqa: BLE001
            self.setup_error = f"{type(e).__name__}: {e}"
        return self

    def close(self):
        try:
            H._run(f"git -C {H.BARE} worktree remove --force {self.repo_dir}",
                   check=False)
        except Exception:
            pass
        if self.slot is not None:
            _VENV_POOL.release(self.slot)
            self.slot = None
        import shutil
        shutil.rmtree(self.workdir, ignore_errors=True)

    # -- path safety -------------------------------------------------------
    def _abs(self, rel):
        rel = (rel or "").replace("\\", "/").lstrip("/")
        full = os.path.normpath(os.path.join(self.repo_dir, rel))
        if not full.startswith(os.path.normpath(self.repo_dir)):
            return None  # escape attempt
        return full

    # -- tools -------------------------------------------------------------
    def read_file(self, path):
        full = self._abs(path)
        if full is None or not os.path.isfile(full):
            return f"ERROR: no such file: {path}"
        try:
            with open(full, "r", errors="replace") as f:
                data = f.read()
        except Exception as e:  # noqa: BLE001
            return f"ERROR: could not read {path}: {e}"
        if len(data) > TRUNC:
            data = data[:TRUNC] + f"\n... [truncated, {len(data)} bytes total]"
        return data

    def list_dir(self, path):
        full = self._abs(path or ".")
        if full is None or not os.path.isdir(full):
            return f"ERROR: not a directory: {path}"
        try:
            entries = sorted(os.listdir(full))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
        lines = []
        for e in entries[:80]:
            tag = "/" if os.path.isdir(os.path.join(full, e)) else ""
            lines.append(e + tag)
        if len(entries) > 80:
            lines.append(f"... ({len(entries)} entries)")
        return "\n".join(lines) if lines else "(empty)"

    def write_file(self, path, content):
        full = self._abs(path)
        if full is None:
            return f"ERROR: illegal path: {path}"
        if content is None or not isinstance(content, str):
            return 'ERROR: write_file needs {"path": ..., "content": ...}'
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: could not write {path}: {e}"
        return f"Wrote {len(content)} bytes to {path}"

    def _pytest(self, node_ids, timeout=600):
        cmd = H._pytest_cmd(self.py, node_ids, lenient=True)
        r = H._run(cmd, cwd=self.repo_dir, check=False, log=self._log,
                   timeout=timeout)
        n_pass, n_fail = H._parse_pytest(r.stdout)
        return r.returncode, n_pass, n_fail, r.stdout

    def run_tests(self):
        """Run F2P + a couple of P2P; return a truncated pass/fail summary."""
        sample = self.f2p + self.p2p[:P2P_SAMPLE]
        try:
            rc, n_pass, n_fail, out = self._pytest(sample)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: pytest failed to run: {e}"
        tail = out.strip().splitlines()[-12:]
        summary = "\n".join(tail)
        if len(summary) > TRUNC:
            summary = summary[-TRUNC:]
        verdict = "PASS" if (rc == 0 and n_fail == 0 and n_pass > 0) else "FAIL"
        return (f"Tests: {verdict} (passed={n_pass}, failed={n_fail}) over "
                f"{len(sample)} selected node ids.\n{summary}")

    def oracle(self):
        """Terminal reward: 1.0 iff FAIL_TO_PASS now PASSES and the sampled
        PASS_TO_PASS still pass; else 0.0. Runs the ORIGINAL node ids."""
        try:
            rc_f, p_f, fail_f, _ = self._pytest(self.f2p)
            f2p_ok = (rc_f == 0 and fail_f == 0 and p_f > 0)
            p2p_ok = True
            sample = self.p2p[:max(P2P_SAMPLE, 5)]
            if sample:
                _, p_p, fail_p, _ = self._pytest(sample)
                p2p_ok = (fail_p == 0 and p_p > 0)
            return 1.0 if (f2p_ok and p2p_ok) else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
TOOLS_DOC = (
    "Available tools:\n"
    '  list_dir {"path": "dask/array"}                 - list a directory\n'
    '  read_file {"path": "dask/array/core.py"}        - print file contents (truncated)\n'
    '  write_file {"path": "...", "content": "..."}    - OVERWRITE the whole file with content\n'
    "  run_tests {}                                     - run the bug's failing test(s) in the venv\n"
    "  submit {}                                        - finish; the hidden test suite is the grader\n"
)

RESPONSE_PROTOCOL = (
    "Think briefly, then end your reply with EXACTLY one line:\n"
    'Action: tool_name {"arg": "value"}\n'
    "One action per reply. write_file replaces the ENTIRE file, so read it first."
)


def build_system_prompt(rule_mode="structural"):
    return (
        "You are a software engineer fixing a bug in the dask repository.\n"
        "All paths are relative to the repo root (e.g. dask/array/core.py).\n"
        + TOOLS_DOC + RESPONSE_PROTOCOL
    )


def patch_stats(patch):
    """Parse a unified-diff gold patch -> {files, n_files, n_hunks, changed, hunks}.
    `hunks` is a list of (new_start_line, context_header) e.g. ('555', 'def _nonempty_series...')."""
    patch = patch or ""
    files = re.findall(r"^\+\+\+ b/(.+?)(?:\t.*)?$", patch, re.M)
    hunks = re.findall(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$", patch, re.M)
    added = len(re.findall(r"^\+(?!\+\+)", patch, re.M))
    removed = len(re.findall(r"^-(?!--)", patch, re.M))
    return {"files": files, "n_files": len(files), "n_hunks": len(hunks),
            "changed": added + removed, "hunks": hunks}


def load_small_patch_instances(max_changed=8, max_files=1, max_hunks=2, n=None):
    """The easy SWE subset: instances whose GOLD patch is tiny (single-file,
    few-line). These are the tractable ones where a fix is localized."""
    out = []
    for i in load_clean_instances():
        st = patch_stats(i.get("patch", ""))
        if (st["n_files"] <= max_files and st["n_hunks"] <= max_hunks
                and 0 < st["changed"] <= max_changed):
            out.append(i)
    return out[:n] if n else out


def oracle_hint(instance):
    """The 'oracle' setting: reveal WHICH file(s)/region the fix lives in, so the
    agent skips localization (SWE-bench's hardest half) and just has to fix it."""
    st = patch_stats(instance.get("patch", ""))
    if not st["files"]:
        return ""
    hint = "Hint: the bug is in: " + ", ".join(st["files"])
    locs = [f"`{ctx.strip()}` (near line {ln})" for ln, ctx in st["hunks"] if ctx.strip()]
    if locs:
        hint += ". Relevant region: " + "; ".join(locs[:3])
    return hint


def build_task_msg(instance, oracle=False):
    ps = (instance.get("problem_statement") or "").strip()
    if len(ps) > 1800:
        ps = ps[:1800] + "\n... [truncated]"
    f2p = _node_ids(instance["FAIL_TO_PASS"])
    f2p_show = "\n".join("  " + n for n in f2p[:6])
    msg = (
        f"Bug report:\n{ps}\n\n"
        f"The following test(s) currently FAIL and must PASS after your fix "
        f"(do not edit the test files):\n{f2p_show}\n\n"
    )
    if oracle:
        h = oracle_hint(instance)
        if h:
            msg += h + "\n\n"
    msg += ("Locate the source bug, edit the source file(s), run the test(s) to "
            "verify, then submit.")
    return msg


# --------------------------------------------------------------------------
# Episode driver
# --------------------------------------------------------------------------
def run_swe_episode(instance, gen, tok, rule_mode="structural", max_steps=12,
                    workdir=None, keep_worktree=False, verbose=False, oracle=False):
    """Roll out one SWE episode and return a trainer-ready Episode (or None if
    setup failed). `gen` is GenServer.generate (ids -> ids). One worktree per
    episode, cleaned up at the end unless keep_worktree."""
    iid = instance["instance_id"]
    if workdir is None:
        workdir = os.path.join("/tmp/swe_rollout", f"{iid}_{os.getpid()}_{int(time.time()*1000)%100000}")
    wt = SweWorktree(instance, workdir)
    wt.setup()
    if not wt.setup_ok:
        if verbose:
            print(f"[{iid}] SETUP FAILED: {wt.setup_error}", flush=True)
        if not keep_worktree:
            wt.close()
        return None

    tracker = SweRuleTracker(mode=rule_mode)
    sys_p = build_system_prompt(rule_mode)
    task_msg = build_task_msg(instance, oracle=oracle)

    ep = Episode(env=None, ids=_ids(tok, TEMPLATE.initial(sys_p, task_msg)))
    reward = 0.0
    tool_log: list = []  # (turn, tool_name) for reporting

    try:
        for _ in range(max_steps):
            g = gen(ep.ids)
            start = len(ep.ids)
            ep.ids.extend(g)
            turn = ep.n_turns
            ep.action_spans.append((start, start + len(g), turn))
            text = tok.decode(g, skip_special_tokens=True)
            call = parse_action(text)

            # rule tracking (token-exact: keyed by THIS turn index)
            tracker.observe_turn(turn, call)
            ep.turn_violations = tracker.turn_violations
            ep.turn_discharges = tracker.turn_discharges

            if call is None or call.name not in (
                    "read_file", "list_dir", "write_file", "run_tests", "submit"):
                tool_log.append((turn, call.name if call else "<parse_error>"))
                obs = ('ERROR: end your reply with exactly one line: '
                       'Action: tool_name {"arg": "value"}. Tools: read_file, '
                       'list_dir, write_file, run_tests, submit')
                ep.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue

            name, a = call.name, (call.args or {})
            tool_log.append((turn, name))
            if name == "read_file":
                obs = wt.read_file(a.get("path"))
            elif name == "list_dir":
                obs = wt.list_dir(a.get("path"))
            elif name == "write_file":
                obs = wt.write_file(a.get("path"), a.get("content"))
            elif name == "run_tests":
                obs = wt.run_tests()
            elif name == "submit":
                ep.done = True
                break

            if len(ep.ids) > 0:  # continue the conversation
                ep.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
        else:
            ep.truncated = True  # hit max_steps without submit

        # terminal oracle (real hidden test suite)
        reward = wt.oracle()
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[{iid}] rollout error: {str(exc)[:200]}", flush=True)
        reward = 0.0
    finally:
        if not keep_worktree:
            wt.close()

    ep.done = True
    ep.env = ShimEnv(reward, tracker)
    # attach tool log + per-tool flags for smoke reporting (ShimEnv.calls unused
    # by the trainer; we stuff a lightweight record there for convenience).
    ep.env.calls = list(tool_log)
    ep._swe_tools = [t for _, t in tool_log]            # type: ignore[attr-defined]
    ep._swe_ran_tests = tracker.ran_f2p                 # type: ignore[attr-defined]
    ep._swe_edited = tracker.edited_source              # type: ignore[attr-defined]
    return ep
