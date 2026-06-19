"""TerminalBench <-> RLVP adapter.

Drives one TerminalBench task as an RL episode with OUR HF policy as the agent
(exact token bookkeeping preserved for token-level credit), the task's Docker
container as the environment, and the task's pytest oracle as the verifiable
terminal reward. Mirrors rlvp/tau2_adapter.py: same Episode shape (action_spans
+ turn_violations/turn_discharges with token-exact spans), same ShimEnv
interface (.success/.violations/.discharges/.calls/.outcome_reward()), reuses
GenServer.

Each agent turn = one bash command. The agent emits a line:
    Action: bash {"cmd": "..."}      -> exec in the persistent container
    Action: submit {}                -> end the episode, score with the oracle

Process signals (all VERIFIABLE from the shell, tracked incrementally):
  penalty  repeat_error      re-issuing a bash command whose previous run
                             exited non-zero (same command string).
  penalty  blind_destructive a destructive command (rm/rmdir/mv/dd/truncate or
                             a `>` overwrite) on a path never inspected.
  discharge blind_destructive an inspect command (ls/cat/stat/head/find/...) on
                             a path discharges its blind-destructive obligation.
  discharge made_progress    a non-error, non-pure-read command (heuristic: the
                             command exited 0 and is not a read-only inspect).

The container/build/oracle machinery is REUSED from termbench/ (run_one.py
proved it end-to-end): build image, `docker run -d ... sleep infinity`, exec
each command, copy tests + run-tests.sh, parse pytest -> {0,1}.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .envs.base import parse_action
from .rollout import TEMPLATE, Episode, _ids
# Reuse the batched generation server from the tau2 adapter (do not duplicate).
from .tau2_adapter import GenServer  # noqa: F401  (re-exported for trainers)

# Make the proven termbench harness importable (load_tasks.load_task).
_TERMBENCH = Path(__file__).resolve().parents[1] / "termbench"
if str(_TERMBENCH) not in sys.path:
    sys.path.insert(0, str(_TERMBENCH))
from load_tasks import load_task  # noqa: E402

OBS_TRUNC = 800  # truncate each observation to bound episode token length


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------
# Terminal RuleTracker (incremental, fed each agent turn = one bash command)
# --------------------------------------------------------------------------

DESTRUCTIVE = ("rm", "rmdir", "mv", "dd", "truncate", "shred")
INSPECT = ("ls", "cat", "stat", "head", "tail", "find", "file", "wc", "less",
           "more", "grep", "tree", "readlink", "du")
# read-only commands that should NOT earn made_progress (pure reads)
PURE_READ = INSPECT + ("echo", "pwd", "which", "whoami", "env", "printenv",
                       "date", "true", "test", "cd")


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except Exception:
        return cmd.split()


def _paths_in(cmd: str) -> list[str]:
    """Best-effort: path-like argv tokens (skip flags and the command word)."""
    toks = _tokens(cmd)
    out = []
    for t in toks[1:]:
        if t.startswith("-"):
            continue
        if t in (">", ">>", "<", "|", "&&", "||", ";"):
            continue
        out.append(t.strip("'\""))
    return out


def _redirect_overwrite_target(cmd: str) -> str | None:
    """If the command overwrites a file via `>` (not `>>`), return the target."""
    m = re.search(r"(?<!>)>(?!>)\s*([^\s;|&>]+)", cmd)
    return m.group(1).strip("'\"") if m else None


class RuleTracker:
    """Per-turn violations/discharges over bash commands. Same interface shape
    as the tau2 RuleTracker (turn_violations / turn_discharges dicts).

    mode='structural' (the only mode here; kept as a param for trainer
    symmetry with tau2_adapter). Signals are verifiable from the shell."""

    def __init__(self, mode="structural"):
        self.mode = mode
        self.errored_cmds: dict[str, int] = {}   # cmd string -> last exit code
        self.inspected: set[str] = set()         # path basenames/strings inspected
        self.turn_violations: dict[int, list] = {}
        self.turn_discharges: dict[int, list] = {}

    @staticmethod
    def _norm(cmd: str) -> str:
        return " ".join(cmd.split())

    def observe_turn(self, turn_idx: int, cmd: str, exit_code: int):
        """Called AFTER the command has run (we have its exit code)."""
        v, d = [], []
        norm = self._norm(cmd)
        toks = _tokens(cmd)
        head = toks[0] if toks else ""
        head = head.rsplit("/", 1)[-1]  # /bin/rm -> rm

        # --- repeat_error: re-issuing a command that previously exited non-zero
        if norm in self.errored_cmds and self.errored_cmds[norm] != 0:
            v.append("repeat_error")

        # --- blind_destructive: destructive op on a never-inspected path
        is_destructive = head in DESTRUCTIVE
        overwrite_tgt = _redirect_overwrite_target(cmd)
        targets = []
        if is_destructive:
            targets = _paths_in(cmd)
        if overwrite_tgt:
            targets.append(overwrite_tgt)
            is_destructive = True
        if is_destructive and targets:
            if not any(self._inspected(p) for p in targets):
                v.append("blind_destructive")

        # --- discharge blind_destructive: an inspect command on a path
        if head in INSPECT:
            for p in _paths_in(cmd):
                self.inspected.add(p)
                self.inspected.add(p.rsplit("/", 1)[-1])
            d.append("blind_destructive")

        # --- discharge made_progress: non-error command that isn't a pure read.
        # A pure-read head still counts as progress if it writes via redirect
        # (e.g. `echo ... > file`, `cat a > b`) — that mutates the filesystem.
        if exit_code == 0 and head and (head not in PURE_READ or overwrite_tgt
                                        or ">>" in cmd):
            d.append("made_progress")

        # record exit code for repeat_error tracking
        self.errored_cmds[norm] = exit_code

        if v:
            self.turn_violations[turn_idx] = v
        if d:
            self.turn_discharges[turn_idx] = d
        return v, d

    def _inspected(self, path: str) -> bool:
        return path in self.inspected or path.rsplit("/", 1)[-1] in self.inspected


# --------------------------------------------------------------------------
# ShimEnv: adapts a terminal episode result to the trainer's env interface
# --------------------------------------------------------------------------


class ShimEnv:
    def __init__(self, reward, tracker, calls):
        self._r = reward
        self.success = reward >= 0.999
        self.violations = [(t, r) for t, rs in tracker.turn_violations.items() for r in rs]
        self.discharges = [(t, r) for t, rs in tracker.turn_discharges.items() for r in rs]
        self.calls = calls
        self.format_errors = 0

    def outcome_reward(self):
        return self._r


# --------------------------------------------------------------------------
# Container lifecycle (reuses the proven run_one.py semantics)
# --------------------------------------------------------------------------


class TaskContainer:
    """Build image, start a persistent container, exec commands, run oracle."""

    def __init__(self, task_id, keep_image=True, verbose=False):
        self.t = load_task(task_id)
        self.task_id = task_id
        self.task_dir = Path(self.t["dir"])
        self.tag = f"tbench-{task_id}:rlvp"
        self.name = f"tbench-ep-{task_id}-{uuid.uuid4().hex[:8]}"
        self.keep_image = keep_image
        self.verbose = verbose
        self.started = False

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def start(self):
        # build (cached after first build; ~0.5-1.8s warm)
        b = sh(["docker", "build", "-t", self.tag, "-f", self.t["dockerfile"],
                str(self.task_dir)])
        if b.returncode != 0:
            raise RuntimeError(f"docker build failed: {b.stderr[-500:]}")
        r = sh(["docker", "run", "-d", "--rm", "--name", self.name, "-w", "/app",
                self.tag, "sh", "-c", "sleep infinity"])
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr[-500:]}")
        self.started = True

    def exec(self, cmd: str, timeout=60):
        """Run one bash command in the persistent container (state persists).
        Returns (stdout, stderr, exit_code)."""
        # capture exit code explicitly; bash -lc gives login-shell PATH
        try:
            ex = sh(["docker", "exec", "-w", "/app", self.name, "bash", "-lc", cmd],
                    timeout=timeout)
            return ex.stdout, ex.stderr, ex.returncode
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout}s", 124

    def score(self) -> float:
        """Copy oracle + tests, run run-tests.sh, parse pytest -> 1.0/0.0."""
        sh(["docker", "exec", self.name, "mkdir", "-p", "/tests"])
        sh(["docker", "cp", self.t["run_tests"], f"{self.name}:/tests/run-tests.sh"])
        if Path(self.t["tests_dir"]).exists():
            sh(["docker", "cp", self.t["tests_dir"] + "/.", f"{self.name}:/tests/"])
        o = sh(["docker", "exec", "-e", "TEST_DIR=/tests", self.name,
                "bash", "/tests/run-tests.sh"], timeout=600)
        out = (o.stdout or "") + "\n" + (o.stderr or "")
        n_fail = re.search(r"(\d+) failed", out)
        n_pass = re.search(r"(\d+) passed", out)
        n_fail = int(n_fail.group(1)) if n_fail else 0
        n_pass = int(n_pass.group(1)) if n_pass else 0
        success = (n_fail == 0 and n_pass > 0)
        self._log("[oracle tail]\n" + "\n".join(out.splitlines()[-5:]))
        return 1.0 if success else 0.0

    def close(self):
        if self.started:
            sh(["docker", "rm", "-f", self.name])
        if not self.keep_image:
            sh(["docker", "rmi", "-f", self.tag])


# --------------------------------------------------------------------------
# Agent protocol / prompt
# --------------------------------------------------------------------------

SHELL_PROTOCOL = """
You are operating a bash shell inside a Linux container. WORKDIR is /app.
The shell session is persistent: cwd, env, and files persist across commands.

How to act: think briefly, then end your reply with EXACTLY one line:
Action: bash {"cmd": "your shell command here"}
When the task is fully done, end your reply with:
Action: submit {}
One action per reply. After each command you will see its stdout/stderr and exit code."""


def _system_prompt(instruction: str) -> str:
    return ("You are an expert software engineer working at a terminal.\n"
            "Task:\n" + instruction.strip() + "\n" + SHELL_PROTOCOL)


def _obs(stdout: str, stderr: str, code: int) -> str:
    body = (stdout or "")
    if stderr:
        body += ("\n[stderr]\n" + stderr) if body else ("[stderr]\n" + stderr)
    body = body.rstrip()
    if len(body) > OBS_TRUNC:
        body = body[:OBS_TRUNC] + f"\n...[truncated, {len(body)} chars]"
    if not body:
        body = "(no output)"
    return f"$ exit={code}\n{body}"


# --------------------------------------------------------------------------
# Episode driver
# --------------------------------------------------------------------------


def run_terminal_episode(task, gen, tok, rule_mode="structural", max_steps=15,
                         keep_image=True, verbose=False):
    """Drive one TerminalBench task as an RL episode; return a trainer-ready
    Episode (with .env = ShimEnv). `task` is a task_id string or a dict with
    'task_id'. Token bookkeeping is identical to tau2_adapter."""
    task_id = task if isinstance(task, str) else task["task_id"]
    instruction = (load_task(task_id)["instruction"] if isinstance(task, str)
                   else task.get("instruction") or load_task(task_id)["instruction"])
    sys_prompt = _system_prompt(instruction)

    tracker = RuleTracker(mode=rule_mode)
    ctr = TaskContainer(task_id, keep_image=keep_image, verbose=verbose)
    calls = []
    reward = 0.0
    try:
        ctr.start()
        # initial observation = the shell banner / task hint
        first_obs = "Shell ready at /app. Begin."
        episode = Episode(env=None,
                          ids=_ids(tok, TEMPLATE.initial(sys_prompt, first_obs)))

        submitted = False
        for _step in range(max_steps):
            g = gen(episode.ids)
            start = len(episode.ids)
            episode.ids.extend(g)
            turn = episode.n_turns
            episode.action_spans.append((start, start + len(g), turn))
            text = tok.decode(g, skip_special_tokens=True)
            call = parse_action(text)

            if call is None:
                # malformed: tell the agent, no rule fires, continue
                obs = ('ERROR: could not parse action. End your reply with one line: '
                       'Action: bash {"cmd": "..."}  or  Action: submit {}')
                episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue
            if call.name == "submit":
                submitted = True
                break
            if call.name != "bash":
                obs = f'ERROR: unknown action "{call.name}". Use bash or submit.'
                episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue

            cmd = (call.args or {}).get("cmd", "")
            if not isinstance(cmd, str) or not cmd.strip():
                obs = 'ERROR: bash action needs a non-empty {"cmd": "..."}.'
                episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue

            stdout, stderr, code = ctr.exec(cmd)
            calls.append(cmd)
            tracker.observe_turn(turn, cmd, code)
            episode.turn_violations = tracker.turn_violations
            episode.turn_discharges = tracker.turn_discharges
            obs = _obs(stdout, stderr, code)

            if len(episode.ids) > 3500 - 200:  # near token budget: stop early
                break
            episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))

        # score with the oracle regardless of submit/max_steps
        reward = ctr.score()
    finally:
        ctr.close()

    episode.turn_violations = tracker.turn_violations
    episode.turn_discharges = tracker.turn_discharges
    episode.env = ShimEnv(reward, tracker, calls)
    episode.done = True
    return episode
