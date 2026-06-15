"""GatedEnv: a non-saturating ceiling testbed (silent precondition gate).

To edit a protected file you must (1) read /acl, then (2) request_access{path},
then (3) write_file{path}. Skipping the gate makes the write a SILENT NO-OP:
it reports "Wrote N bytes" but the file is unchanged, so the test fails with
no hint about WHY. The base model rarely discovers this blind multi-step gate,
so outcome-only stays stuck low (no gradient: it almost never samples success)
-> the task does not saturate. RLVP's verifiable rule ("write to a protected
path without granted access") guides the policy to the gate -> a CEILING gap,
not just a speed gap. This mirrors real auth/lock-before-write workflows
(and tau2's auth-before-action), so the difficulty is natural, not a trap.

The gate is hard-to-DISCOVER (silent failure) but easy-to-VERIFY (access was
granted for this path before the write), which is exactly the regime where a
specified process rule can raise the ceiling.
"""
from __future__ import annotations

import random

from .base import Rule, StepResult, ToolCall, ToolEnv

ACL_PATH = "/acl"
ACL_TEXT = ("ACCESS CONTROL: files under /protected are write-locked. After "
            "reading this file you may call request_access for a specific path; "
            "only then will a write to that path take effect.")


def _sig(call):
    import json
    return call.name + "|" + json.dumps(call.args, sort_keys=True, default=str)


class WriteWithoutAccess(Rule):
    name = "write_without_access"

    def check(self, env, call):
        return (call.name == "write_file"
                and str(call.args.get("path", "")).startswith("/protected")
                and str(call.args.get("path", "")) not in env.granted)


class AccessWithoutAcl(Rule):
    name = "access_without_acl"

    def check(self, env, call):
        return call.name == "request_access" and not env.acl_read


class RepeatFailure(Rule):
    name = "repeat_failure"

    def check(self, env, call):
        return env.errored_sigs.get(_sig(call), 0) >= 2


class GatedEnv(ToolEnv):
    max_turns = 14
    rules = [WriteWithoutAccess(), AccessWithoutAcl(), RepeatFailure()]
    tool_names = ("list_dir", "read_file", "request_access", "write_file",
                  "run_tests", "submit")
    TOOL_CATEGORIES = {"list_dir": "observe", "read_file": "observe",
                       "request_access": "observe",  # a precondition-gathering act
                       "write_file": "mutate", "run_tests": "verify",
                       "submit": "terminal"}
    TARGET_ARG = {"read_file": "path", "write_file": "path",
                  "request_access": "path", "list_dir": "path"}

    def __init__(self, task, **kw):
        super().__init__(task, **kw)
        self.fs = dict(task["fs"])
        self.acl_read = False
        self.granted: set = set()
        self.read_paths: set = set()
        self.last_mutation_turn = None
        self.last_tests_turn = None
        self.errored_sigs: dict = {}

    def system_prompt(self, include_rules: bool = False) -> str:
        p = (
            "You are a file-operations agent.\n"
            "Available tools:\n"
            '  list_dir {"path": "/dir"}\n'
            '  read_file {"path": "/dir/f"}\n'
            '  request_access {"path": "/dir/f"}   - request write access to a path\n'
            '  write_file {"path": "/dir/f", "content": "..."}\n'
            "  run_tests {}\n"
            "  submit {}\n"
            "End each reply with exactly one line: Action: tool_name {\"arg\": \"value\"}"
        )
        if include_rules:
            p += ("\n\nGuidelines: files under /protected are write-locked. Read "
                  + ACL_PATH + ", then request_access for the path, before writing it.")
        return p

    def initial_user_msg(self) -> str:
        return "Task: " + self.task["instruction"]

    def _solved(self):
        return self.fs.get(self.task["target_path"]) == self.task["target_content"]

    def discharge_rules(self, call):
        out = []
        a = call.args
        if call.name == "read_file" and a.get("path") == ACL_PATH and not self.acl_read:
            out.append("access_without_acl")   # gathered the precondition info
        if (call.name == "request_access" and self.acl_read
                and str(a.get("path", "")).startswith("/protected")
                and str(a.get("path", "")) not in self.granted):
            out.append("write_without_access")  # cleared the write precondition
        return out

    def apply(self, call: ToolCall) -> StepResult:
        name, a = call.name, call.args
        err = None
        if name == "list_dir":
            path = str(a.get("path", "")).rstrip("/") or "/"
            entries = sorted({("/" + f[len(path):].lstrip("/").split("/")[0])
                              for f in self.fs if f.startswith(path + "/")})
            obs = "\n".join(path + e for e in entries) if entries else None
            if obs is None:
                err = f"ERROR: empty/no dir: {path}"
        elif name == "read_file":
            path = str(a.get("path", ""))
            if path in self.fs:
                self.read_paths.add(path)
                if path == ACL_PATH:
                    self.acl_read = True
                obs = self.fs[path]
            else:
                err = f"ERROR: no such file: {path}"
        elif name == "request_access":
            path = str(a.get("path", ""))
            if self.acl_read:
                self.granted.add(path)
            # deliberately uninformative: same message whether or not it took
            obs = f"Access request submitted for {path}."
        elif name == "write_file":
            path, content = str(a.get("path", "")), a.get("content", None)
            if not path or not isinstance(content, str):
                err = 'ERROR: write_file needs {"path","content"}'
            else:
                protected = path.startswith("/protected")
                if (not protected) or (path in self.granted):
                    self.fs[path] = content        # the write takes
                self.last_mutation_turn = self.turn
                obs = f"Wrote {len(content)} bytes to {path}"  # SILENT no-op if locked
        elif name == "run_tests":
            self.last_tests_turn = self.turn
            obs = "Tests: PASS (3/3)" if self._solved() else \
                  "Tests: FAIL (1/3) - target content not applied"
        elif name == "submit":
            self.done = True
            self.success = self._solved()
            obs = "Submitted."
        if err is not None:
            self.errored_sigs[_sig(call)] = self.errored_sigs.get(_sig(call), 0) + 1
            obs = err
        return StepResult(observation=obs, done=self.done)


def make_task(seed: int) -> dict:
    rng = random.Random(7_000_003 * (seed + 1))
    keys = ["timeout", "retries", "port", "workers", "ttl"]
    k = rng.choice(keys)
    old, new = str(rng.randint(1, 99)), str(rng.randint(100, 999))
    tgt = "/protected/config.ini"
    return {
        "seed": seed,
        "fs": {ACL_PATH: ACL_TEXT,
               tgt: f"{k} = {old}",
               "/protected/README": "see /acl for the write policy"},
        "instruction": f"In {tgt}, set `{k}` to `{new}` (keep other settings).",
        "target_path": tgt, "target_content": f"{k} = {new}",
    }


def compliant_script(task, imperfect: bool = False, skip_rules: tuple = ()) -> list:
    import json as _json
    tgt = task["target_path"]
    s = [f"I must consult the access policy first.\nAction: read_file {_json.dumps({'path': ACL_PATH})}",
         f"Read the protected target.\nAction: read_file {_json.dumps({'path': tgt})}",
         f"Request write access.\nAction: request_access {_json.dumps({'path': tgt})}",
         f"Now the write will take.\nAction: write_file {_json.dumps({'path': tgt, 'content': task['target_content']})}",
         "Action: run_tests {}", "Action: submit {}"]
    return s
