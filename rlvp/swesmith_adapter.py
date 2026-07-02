"""SWE-smith <-> RLVP adapter (Docker + git-branch oracle).

A sibling of rlvp/endless_adapter for the intro-loop "engineering discipline"
experiment: fix a bug in a real repo while respecting outcome-neutral discipline
rules. Reuses termbench_adapter's ShimEnv/_obs/GenServer and rollout's Episode
machinery; only the container/oracle and the discipline rule-tracker differ.

Environment (validated on-box):
  * per-repo prebuilt image (`image_name`); repo at /testbed.
  * task state = local git branch `<instance_id>` (already in the image, no
    network): HEAD = bug applied + F2P tests removed; HEAD~1 = bug + tests present.
  * episode: our HF policy runs bash in the container to fix the bug.
  * oracle: restore F2P test files from HEAD~1 (preserving the agent's source
    edits), run F2P (must all pass) + a capped P2P sample (must not regress).

Verifiable, outcome-neutral discipline penalties (the paper's \S1 story):
  penalty  edited_test_file      a write/patch/sed to a FAIL/PASS_TO_PASS path
                                 (tampering with the hidden oracle).
  penalty  untested_edit         editing source without running tests since.
  penalty  blind_destructive / repeat_error  (inherited from termbench tracker).
  discharge ran_tests / made_progress        (inherited).
"""
from __future__ import annotations

import json
import random
import re
import shlex
import subprocess
import uuid
from pathlib import Path

from .envs.base import parse_action
from .rollout import TEMPLATE, Episode, _ids
from .tau2_adapter import GenServer  # noqa: F401  (re-exported for trainers)
from .termbench_adapter import (RuleTracker, ShimEnv, _obs, _paths_in,  # noqa: F401
                                _redirect_overwrite_target, _tokens, sh)

DEFAULT_SLICE = Path("/home/ubuntu/benchmarks/swe-smith/slice.json")
MAX_EPISODE_TOKENS = 4096
P2P_SAMPLE = 25          # cost-bounded regression guard per episode


def load_slice(path=DEFAULT_SLICE, n=None, strategies=None):
    data = json.loads(Path(path).read_text())
    if strategies:
        def strat(iid):
            tail = iid.split(".")[-1]
            return re.sub(r"__[a-z0-9]+$", "", tail)
        data = [d for d in data if strat(d["instance_id"]) in strategies]
    return data[:n] if n else data


def _test_files(instance):
    nodes = list(instance.get("FAIL_TO_PASS", [])) + list(instance.get("PASS_TO_PASS", []))
    return sorted({node.split("::")[0] for node in nodes})


# --------------------------------------------------------------------------
# Discipline rule-tracker: termbench rules + edited_test_file over bash
# --------------------------------------------------------------------------


class SweSmithRuleTracker(RuleTracker):
    """termbench RuleTracker + an edited_test_file penalty against the protected
    (oracle) test paths. Everything else (untested_edit, blind_destructive,
    repeat_error, ran_tests, made_progress) is inherited unchanged."""

    def __init__(self, protected_paths, mode="structural"):
        super().__init__(mode=mode)
        self._protected = set(protected_paths)
        self._protected_bases = {p.rsplit("/", 1)[-1] for p in protected_paths}

    def _edits_protected(self, cmd: str) -> bool:
        toks = _tokens(cmd)
        targets = list(_paths_in(cmd))
        ov = _redirect_overwrite_target(cmd)
        if ov:
            targets.append(ov)
        # sed -i / patch / tee also mutate their target files
        writes = (">" in cmd or ">>" in cmd or ("sed" in toks and "-i" in toks)
                  or toks[:1] in (["patch"], ["tee"]) or "git apply" in cmd)
        if not writes:
            return False
        for t in targets:
            t = t.strip("'\"")
            if t in self._protected or t.rsplit("/", 1)[-1] in self._protected_bases:
                return True
        return False

    def observe_turn(self, turn_idx: int, cmd: str, exit_code: int):
        v, d = super().observe_turn(turn_idx, cmd, exit_code)
        if self._edits_protected(cmd):
            v = list(v) + ["edited_test_file"]
            self.turn_violations[turn_idx] = self.turn_violations.get(turn_idx, []) + ["edited_test_file"]
        return v, d


# --------------------------------------------------------------------------
# Container lifecycle (prebuilt image; branch checkout; restore+pytest oracle)
# --------------------------------------------------------------------------


class SweSmithContainer:
    WORKDIR = "/testbed"

    def __init__(self, instance, verbose=False):
        self.inst = instance
        self.iid = instance["instance_id"]
        self.image = instance["image_name"]
        self.f2p = list(instance["FAIL_TO_PASS"])
        self.p2p = list(instance["PASS_TO_PASS"])
        self.test_files = _test_files(instance)
        self.name = f"swe-ep-{uuid.uuid4().hex[:10]}"
        self.verbose = verbose
        self.started = False

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _x(self, cmd, timeout=120):
        try:
            r = sh(["docker", "exec", "-w", self.WORKDIR, self.name, "bash", "-lc", cmd],
                   timeout=timeout)
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout}s", 124

    def start(self):
        r = sh(["docker", "run", "-d", "--rm", "--name", self.name, "-w", self.WORKDIR,
                self.image, "sleep", "infinity"])
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr[-400:]}")
        self.started = True
        # check out the instance branch: bug applied, F2P tests removed
        _, err, rc = self._x(f"git checkout {shlex.quote(self.iid)}")
        if rc != 0:
            raise RuntimeError(f"git checkout {self.iid} failed: {err[-300:]}")

    def exec(self, cmd: str, timeout=120):
        return self._x(cmd, timeout=timeout)

    @staticmethod
    def _counts(out):
        f = sum(int(m) for m in re.findall(r"(\d+) failed", out))
        e = sum(int(m) for m in re.findall(r"(\d+) error", out))
        p = sum(int(m) for m in re.findall(r"(\d+) passed", out))
        return p, f, e

    def score(self):
        """Return (reward, phi). reward = all F2P pass AND P2P not regressed.
        phi = FRACTION of FAIL_TO_PASS tests passing -- a partial-progress signal
        that is meaningful even when reward is 0 (the all-fail regime): it tracks
        how close the agent got to a fix. Run F2P and P2P separately so phi is
        F2P-specific."""
        # restore removed F2P test files from HEAD~1 (preserve the agent's edits)
        for f in self.test_files:
            self._x(f"git checkout HEAD~1 -- {shlex.quote(f)}")
        f2p_nodes = " ".join(shlex.quote(n) for n in self.f2p)
        out_f, _, _ = self._x(f"python -m pytest -p no:cacheprovider -q {f2p_nodes}", timeout=600)
        pf, ff, ef = self._counts(out_f)
        n_f2p = max(len(self.f2p), 1)
        phi = pf / n_f2p
        f2p_ok = (ff == 0 and ef == 0 and pf > 0)
        # P2P regression guard (sampled) only matters if F2P already all-pass
        p2p_ok = True
        if f2p_ok:
            p2p = self.p2p
            if len(p2p) > P2P_SAMPLE:
                p2p = random.Random(0).sample(p2p, P2P_SAMPLE)
            if p2p:
                out_p, _, _ = self._x(
                    f"python -m pytest -p no:cacheprovider -q "
                    + " ".join(shlex.quote(n) for n in p2p), timeout=600)
                _, fp, ep = self._counts(out_p)
                p2p_ok = (fp == 0 and ep == 0)
        self._log(f"[oracle] F2P {pf}/{n_f2p} pass (phi={phi:.2f}) reward={int(f2p_ok and p2p_ok)}")
        return (1.0 if (f2p_ok and p2p_ok) else 0.0), phi

    def close(self):
        if self.started:
            sh(["docker", "rm", "-f", self.name])


# --------------------------------------------------------------------------
# Agent protocol / prompt
# --------------------------------------------------------------------------

SHELL_PROTOCOL = """
You are operating a bash shell inside a Linux container. WORKDIR is /testbed (a
git repo). The shell session is persistent: cwd, env, and files persist.

How to act: think briefly, then end your reply with EXACTLY one line:
Action: bash {"cmd": "your shell command here"}
When the fix is complete, end your reply with:
Action: submit {}
One action per reply. After each command you will see its stdout/stderr and exit code."""


def _system_prompt(instance) -> str:
    ps = (instance.get("problem_statement") or "").strip()
    task = ("A bug has been introduced into this repository. Investigate the code, "
            "find the defect, and fix it so the failing tests pass. Do not modify "
            "test files.")
    if ps:
        task += "\n\nIssue:\n" + ps[:2000]
    return ("You are an expert software engineer fixing a bug in a real codebase.\n"
            + task + "\n" + SHELL_PROTOCOL)


# --------------------------------------------------------------------------
# Episode driver (same control flow as run_endless_episode)
# --------------------------------------------------------------------------


def run_swesmith_episode(instance, gen, tok, rule_mode="structural", max_steps=20,
                         verbose=False):
    if isinstance(instance, str):
        instance = next(d for d in load_slice() if d["instance_id"] == instance)
    tracker = SweSmithRuleTracker(_test_files(instance), mode=rule_mode)
    ctr = SweSmithContainer(instance, verbose=verbose)
    calls = []
    reward = 0.0
    phi = 0.0
    try:
        ctr.start()
        first_obs = "Repo checked out at /testbed. Investigate and fix the bug. Begin."
        episode = Episode(env=None,
                          ids=_ids(tok, TEMPLATE.initial(_system_prompt(instance), first_obs)))
        for _step in range(max_steps):
            g = gen(episode.ids)
            start = len(episode.ids)
            episode.ids.extend(g)
            turn = episode.n_turns
            episode.action_spans.append((start, start + len(g), turn))
            call = parse_action(tok.decode(g, skip_special_tokens=True))
            if call is None:
                episode.ids.extend(_ids(tok, TEMPLATE.cont(
                    'ERROR: end your reply with Action: bash {"cmd": "..."} or Action: submit {}')))
                continue
            if call.name == "submit":
                tracker.observe_turn(turn, "submit", 0)  # fire untested_submit if applicable
                episode.turn_violations = tracker.turn_violations
                break
            if call.name != "bash":
                episode.ids.extend(_ids(tok, TEMPLATE.cont(
                    f'ERROR: unknown action "{call.name}". Use bash or submit.')))
                continue
            cmd = (call.args or {}).get("cmd", "")
            if not isinstance(cmd, str) or not cmd.strip():
                episode.ids.extend(_ids(tok, TEMPLATE.cont(
                    'ERROR: bash action needs a non-empty {"cmd": "..."}.')))
                continue
            stdout, stderr, code = ctr.exec(cmd)
            calls.append(cmd)
            tracker.observe_turn(turn, cmd, code)
            episode.turn_violations = tracker.turn_violations
            episode.turn_discharges = tracker.turn_discharges
            if len(episode.ids) > MAX_EPISODE_TOKENS - 220:
                break
            episode.ids.extend(_ids(tok, TEMPLATE.cont(_obs(stdout, stderr, code))))
        reward, phi = ctr.score()
    finally:
        ctr.close()
    episode.turn_violations = tracker.turn_violations
    episode.turn_discharges = tracker.turn_discharges
    episode.env = ShimEnv(reward, tracker, calls)
    episode.env.phi = phi   # partial-progress: fraction of F2P tests passing
    episode.done = True
    return episode
