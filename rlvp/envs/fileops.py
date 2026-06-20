"""FileOps: terminal-style file manipulation environment (terminal-bench flavored).

Tools: list_dir, read_file, write_file, delete, run_tests, submit
Rules (penalty-only, checked before the call applies):
  R1 blind_write      - write_file to an EXISTING file that was never read
  R2 blind_delete     - delete a path never read whose directory was never listed
  R3 untested_submit  - submit after mutations without a run_tests after the last mutation
  R4 repeat_failure   - re-issuing a call signature that has already errored twice
"""
from __future__ import annotations

import json
import os
import random

from .base import Rule, StepResult, ToolCall, ToolEnv

CONFIG_KEYS = ["timeout", "retries", "port", "log_level", "max_conn", "cache_ttl", "workers"]
LOG_LEVELS = ["debug", "info", "warning", "error"]


def _sig(call: ToolCall) -> str:
    return call.name + "|" + json.dumps(call.args, sort_keys=True)


class BlindWrite(Rule):
    name = "blind_write"

    def check(self, env, call):
        return (
            call.name == "write_file"
            and call.args.get("path") in env.fs
            and call.args.get("path") not in env.read_paths
        )


class BlindDelete(Rule):
    name = "blind_delete"

    def check(self, env, call):
        if call.name != "delete":
            return False
        path = str(call.args.get("path", ""))
        return path not in env.read_paths and os.path.dirname(path) not in env.listed_dirs


class UntestedSubmit(Rule):
    name = "untested_submit"

    def check(self, env, call):
        if call.name != "submit" or env.last_mutation_turn is None:
            return False
        return env.last_tests_turn is None or env.last_tests_turn < env.last_mutation_turn


class RepeatFailure(Rule):
    name = "repeat_failure"

    def check(self, env, call):
        return env.errored_sigs.get(_sig(call), 0) >= 2


class FileOpsEnv(ToolEnv):
    max_turns = 12
    rules = [BlindWrite(), BlindDelete(), UntestedSubmit(), RepeatFailure()]
    tool_names = ("list_dir", "read_file", "write_file", "delete", "run_tests", "submit")
    # auto-rule tags (structural, task-agnostic): the auto engine reconstructs
    # the same process signal from these alone
    TOOL_CATEGORIES = {"list_dir": "observe", "read_file": "observe",
                       "write_file": "mutate", "delete": "mutate",
                       "run_tests": "verify", "submit": "terminal"}
    TARGET_ARG = {"read_file": "path", "write_file": "path", "delete": "path",
                  "list_dir": "path"}

    def __init__(self, task: dict, **kw):
        super().__init__(task, **kw)
        self.fs = dict(task["fs"])
        self.read_paths: set = set()
        self.listed_dirs: set = set()
        self.last_mutation_turn = None
        self.last_tests_turn = None
        self.errored_sigs: dict = {}

    # ------------------------------------------------------------------
    def system_prompt(self, include_rules: bool = False) -> str:
        p = (
            "You are a software operations agent working in a virtual file system.\n"
            "Available tools:\n"
            '  list_dir {"path": "/dir"}            - list files in a directory\n'
            '  read_file {"path": "/dir/f"}         - print file contents\n'
            '  write_file {"path": "/dir/f", "content": "..."} - replace/create file with content\n'
            '  delete {"path": "/dir/f"}            - delete a file\n'
            "  run_tests {}                          - run the project's test suite\n"
            "  submit {}                             - declare the task finished\n"
            "Think briefly, then end your reply with EXACTLY one line of the form:\n"
            'Action: tool_name {"arg": "value"}\n'
            "One action per reply. After the task is done, use: Action: submit {}"
        )
        if include_rules:
            p += (
                "\n\nEngineering guidelines you MUST follow:\n"
                "1. Read a file before overwriting it.\n"
                "2. List a directory or read a file before deleting it.\n"
                "3. Run the tests after your last change, before submitting.\n"
                "4. Never repeat a command that has already failed twice."
            )
        return p

    def initial_user_msg(self) -> str:
        return "Task: " + self.task["instruction"]

    def discharge_rules(self, call):
        """A call discharges a rule when it satisfies a PENDING obligation."""
        out = []
        name, a = call.name, call.args
        if (name == "run_tests" and self.last_mutation_turn is not None
                and (self.last_tests_turn is None or self.last_tests_turn < self.last_mutation_turn)):
            out.append("untested_submit")   # tests now cover the pending mutations
        if name == "read_file" and a.get("path") in self.fs and a.get("path") not in self.read_paths:
            out.append("blind_write")       # informed about this file's contents
        if name == "list_dir":
            path = str(a.get("path", "")).rstrip("/") or "/"
            if path not in self.listed_dirs and any(f.startswith(path + "/") for f in self.fs):
                out.append("blind_delete")  # informed about this directory
        return out

    # ------------------------------------------------------------------
    def _check_solution(self):
        t = self.task
        if t["type"] == "chain":
            return all(_check_one(sub, self.fs) for sub in t["stages"])
        return _check_one(t, self.fs)

    def apply(self, call: ToolCall) -> StepResult:
        name, a = call.name, call.args
        err = None
        if name == "list_dir":
            path = str(a.get("path", "")).rstrip("/") or "/"
            entries = sorted(
                {("/" + f[len(path):].lstrip("/").split("/")[0]) for f in self.fs if f.startswith(path + "/")}
            )
            if entries:
                self.listed_dirs.add(path)
                obs = "\n".join(path + e for e in entries)
            else:
                err = f"ERROR: directory not found or empty: {path}"
        elif name == "read_file":
            path = str(a.get("path", ""))
            if path in self.fs:
                self.read_paths.add(path)
                obs = self.fs[path]
            else:
                err = f"ERROR: no such file: {path}"
        elif name == "write_file":
            path, content = str(a.get("path", "")), a.get("content", None)
            if not path or content is None or not isinstance(content, str):
                err = 'ERROR: write_file needs {"path": ..., "content": ...}'
            else:
                self.fs[path] = content
                self.last_mutation_turn = self.turn
                obs = f"Wrote {len(content)} bytes to {path}"
        elif name == "delete":
            path = str(a.get("path", ""))
            if path in self.fs:
                del self.fs[path]
                self.last_mutation_turn = self.turn
                obs = f"Deleted {path}"
            else:
                err = f"ERROR: no such file: {path}"
        elif name == "run_tests":
            self.last_tests_turn = self.turn
            if self.task["type"] == "chain":
                fails = [i + 1 for i, sub in enumerate(self.task["stages"])
                         if not _check_one(sub, self.fs)]
                n = len(self.task["stages"])
                obs = (f"Tests: PASS ({n}/{n})" if not fails else
                       f"Tests: FAIL ({n - len(fails)}/{n}) - failing stages: "
                       + ", ".join(str(i) for i in fails))
            elif self._check_solution():
                obs = "Tests: PASS (3/3)"
            else:
                obs = "Tests: FAIL (1/3) - " + self.task["fail_hint"]
        elif name == "submit":
            self.done = True
            self.success = self._check_solution()
            obs = "Submitted."
        if err is not None:
            self.errored_sigs[_sig(call)] = self.errored_sigs.get(_sig(call), 0) + 1
            obs = err
        return StepResult(observation=obs, done=self.done)


def _check_one(t: dict, fs: dict) -> bool:
    if t["type"] in ("edit_config", "create_file"):
        return fs.get(t["target_path"]) == t["target_content"]
    if t["type"] == "cleanup_tmp":
        return (all(p not in fs for p in t["tmp_files"])
                and all(p in fs for p in t["keep_files"]))
    return False


def _remap(task: dict, prefix: str) -> dict:
    """Prefix every path in a task dict so chained stages don't collide."""
    t = dict(task)
    t["fs"] = {prefix + p: c for p, c in task["fs"].items()}
    for key in ("target_path",):
        if key in t:
            t[key] = prefix + t[key]
    for key in ("tmp_files", "keep_files"):
        if key in t:
            t[key] = [prefix + p for p in t[key]]
    t["instruction"] = task["instruction"].replace("/app", prefix + "/app").replace("/data", prefix + "/data")
    return t


def make_chain_task(seed: int, n_stages: int) -> dict:
    """Horizon-scaling task: n_stages independent sub-tasks in one episode.
    Rule-relevant decision points grow ~3x per stage."""
    stages = [_remap(make_task(1_000_000 + seed * 31 + i), f"/s{i + 1}") for i in range(n_stages)]
    fs = {}
    for s in stages:
        fs.update(s["fs"])
    instruction = (f"Complete ALL {n_stages} stages, then submit once:\n"
                   + "\n".join(f"{i + 1}. {s['instruction']}" for i, s in enumerate(stages)))
    return {"type": "chain", "seed": seed, "n_stages": n_stages,
            "fs": fs, "stages": stages, "instruction": instruction,
            "fail_hint": "one or more stages incomplete"}


def make_chain_env(seed: int, n_stages: int, **kw) -> "FileOpsEnv":
    env = FileOpsEnv(make_chain_task(seed, n_stages), **kw)
    env.max_turns = 8 + 7 * n_stages
    return env


class ChainPotentialEnv(FileOpsEnv):
    """Exposes the VERIFIABLE POTENTIAL Phi = #satisfied stages as a granularity-
    controllable 'stage_progress' discharge (potential-only: track_rules off, so NO
    penalties). For experiments E-A/E-B validating 'RLVP helps iff the domain has a
    verifiable Phi strictly finer than the terminal outcome':
      granularity='fine'  : credit every -dPhi   (finest verifiable potential)
      granularity='mid'   : one milestone at >= half the stages
      granularity='coarse': nothing -> reduces to outcome (Phi == terminal outcome)
    n_stages is the SPARSITY knob (more stages -> lower success -> blinder outcome)."""

    def __init__(self, task: dict, granularity: str = "fine", **kw):
        kw.setdefault("track_rules", False)
        super().__init__(task, **kw)
        self.granularity = granularity
        self.n_stages = task.get("n_stages", 1)
        self._phi = 0
        self._mid_paid = False

    def step_text(self, model_text: str):
        res = super().step_text(model_text)
        phi = sum(1 for sub in self.task["stages"] if _check_one(sub, self.fs))
        new = []
        if phi > self._phi:
            if self.granularity == "fine":
                new = ["stage_progress"] * (phi - self._phi)
            elif self.granularity == "mid" and not self._mid_paid \
                    and phi >= (self.n_stages + 1) // 2:
                new = ["stage_progress"]
                self._mid_paid = True
            self._phi = phi
        for r in new:
            self.discharges.append((self.turn, r))
        res.discharges = (res.discharges or []) + new
        return res


def make_chain_potential_env(seed: int, n_stages: int, granularity: str = "fine",
                             **kw) -> "ChainPotentialEnv":
    env = ChainPotentialEnv(make_chain_task(seed, n_stages), granularity=granularity, **kw)
    env.max_turns = 8 + 7 * n_stages
    return env


def compliant_script(task: dict, imperfect: bool = False, skip_rules: tuple = ()) -> list:
    """Assistant texts for a compliant trajectory. With imperfect=True the
    workflow is identical but the task is botched (wrong content / partial
    cleanup) — compliant scaffolding around a failing solution. skip_rules
    removes the steps that demonstrate the named rules (clean holdout)."""
    import json as _json
    if task["type"] == "chain":  # compose per-stage scripts, one test+submit
        s = []
        for sub in task["stages"]:
            s.extend(compliant_script(sub, imperfect, skip_rules)[:-2])
        if "untested_submit" not in skip_rules:
            s.append("All stages changed files; run the tests before submitting.\nAction: run_tests {}")
        s.append("Submitting.\nAction: submit {}")
        return s
    s = []
    content = "TODO: fix" if imperfect else task.get("target_content")
    if task["type"] == "edit_config":
        if "blind_write" not in skip_rules:
            s.append('I should read the file before changing it.\nAction: read_file '
                     + _json.dumps({"path": task["target_path"]}))
        s.append('Now write the updated config.\nAction: write_file '
                 + _json.dumps({"path": task["target_path"], "content": content}))
    elif task["type"] == "cleanup_tmp":
        if "blind_delete" not in skip_rules:
            ddir = os.path.dirname(task["tmp_files"][0])
            s.append('I should list the directory before deleting anything.\nAction: list_dir '
                     + _json.dumps({"path": ddir}))
        tmp = task["tmp_files"][:1] if imperfect else task["tmp_files"]
        for p in tmp:
            s.append(f'Deleting a listed .tmp file.\nAction: delete ' + _json.dumps({"path": p}))
    else:  # create_file
        s.append('Creating the requested file.\nAction: write_file '
                 + _json.dumps({"path": task["target_path"], "content": content}))
    if "untested_submit" not in skip_rules:
        s.append('I changed files, so I must run the tests before submitting.\nAction: run_tests {}')
    s.append('Submitting.\nAction: submit {}')
    return s


# ---------------------------------------------------------------------------
def make_task(seed: int) -> dict:
    rng = random.Random(seed)
    ttype = ["edit_config", "cleanup_tmp", "create_file"][seed % 3]
    if ttype == "edit_config":
        keys = rng.sample(CONFIG_KEYS, 4)
        vals = {k: (rng.choice(LOG_LEVELS) if k == "log_level" else str(rng.randint(1, 900))) for k in keys}
        tgt_key = rng.choice(keys)
        new_val = str(rng.randint(1, 900)) if tgt_key != "log_level" else rng.choice(LOG_LEVELS)
        while new_val == vals[tgt_key]:
            new_val = str(rng.randint(1, 900)) if tgt_key != "log_level" else rng.choice(LOG_LEVELS)
        old = "\n".join(f"{k} = {vals[k]}" for k in keys)
        new = "\n".join(f"{k} = {new_val if k == tgt_key else vals[k]}" for k in keys)
        return {
            "type": ttype, "seed": seed,
            "fs": {"/app/config.ini": old, "/app/README.md": "Service config lives in /app/config.ini"},
            "instruction": (
                f"In /app/config.ini, set `{tgt_key}` to `{new_val}`. "
                "Keep every other setting unchanged."
            ),
            "target_path": "/app/config.ini", "target_content": new,
            "fail_hint": f"config value `{tgt_key}` is wrong or other settings changed",
        }
    if ttype == "cleanup_tmp":
        n_tmp = rng.randint(2, 3)
        tmp = [f"/data/{rng.choice(['job', 'cache', 'sess', 'out'])}{rng.randint(10, 99)}.tmp" for _ in range(n_tmp)]
        tmp = sorted(set(tmp))
        keep = sorted({f"/data/{rng.choice(['report', 'users', 'events'])}{rng.randint(10, 99)}.{rng.choice(['csv', 'log', 'json'])}" for _ in range(2)})
        fs = {p: f"tmpdata {rng.randint(0, 9999)}" for p in tmp}
        fs.update({p: f"data {rng.randint(0, 9999)}" for p in keep})
        return {
            "type": ttype, "seed": seed, "fs": fs,
            "instruction": "Delete every .tmp file in /data. Do not touch any other file.",
            "tmp_files": tmp, "keep_files": keep,
            "fail_hint": "a .tmp file still exists, or a non-tmp file was removed",
        }
    # create_file
    ver = f"{rng.randint(0, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    return {
        "type": ttype, "seed": seed,
        "fs": {"/app/main.py": "print('service starting')", "/app/README.md": "Versioning: keep /app/VERSION current."},
        "instruction": f"Create the file /app/VERSION containing exactly `{ver}` (no trailing newline).",
        "target_path": "/app/VERSION", "target_content": ver,
        "fail_hint": "/app/VERSION missing or has wrong content",
    }
