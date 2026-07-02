"""Endless Terminals <-> RLVP adapter.

Drives one Endless Terminals task as an RL episode with our HF policy as the
agent, the task's Docker container as the environment, and the task's pytest
oracle as the verifiable terminal reward. A sibling of rlvp/termbench_adapter:
same Episode shape, same ShimEnv interface, same GenServer, and it REUSES that
module's RuleTracker / ShimEnv / _obs verbatim so the verifiable-penalty signal
(repeat_error, blind_destructive, made_progress, untested_edit) is identical to
the TerminalBench harm setup. Only three things differ and are reimplemented
here: (1) tasks are on-disk directories from the `obiwan96/endless-terminals`
dataset rather than the termbench registry; (2) the container WORKDIR is
/home/user; (3) the oracle is the task's own `tests/test.sh`, which writes 1/0
to /logs/verifier/reward.txt (validated end-to-end in benchmarks/endless_terminals).

Each agent turn = one bash command:
    Action: bash {"cmd": "..."}      -> exec in the persistent container
    Action: submit {}                -> end the episode, score with the oracle
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from .envs.base import parse_action
from .rollout import TEMPLATE, Episode, _ids
# Reuse the batched generation server + the verifiable-rule machinery unchanged.
from .tau2_adapter import GenServer  # noqa: F401  (re-exported for trainers)
from .termbench_adapter import RuleTracker, ShimEnv, _obs, sh  # noqa: F401

# Default: the pilot slice lives outside the repo (task images + data untracked).
DEFAULT_TASK_ROOT = Path("/home/ubuntu/benchmarks/endless-terminals/slice")
MAX_EPISODE_TOKENS = 3500


# --------------------------------------------------------------------------
# Task discovery (on-disk dataset dirs)
# --------------------------------------------------------------------------


def load_task(task):
    """Resolve a task spec to its on-disk paths. `task` may be a task-id string
    (dir name under DEFAULT_TASK_ROOT), an absolute dir path, or a dict with
    'dir'/'task_id'. Returns a dict mirroring termbench's load_task fields."""
    if isinstance(task, dict):
        d = Path(task.get("dir") or (DEFAULT_TASK_ROOT / task["task_id"]))
    else:
        p = Path(task)
        d = p if p.is_absolute() or p.exists() else DEFAULT_TASK_ROOT / task
    d = d.resolve()
    instr = d / "instruction.md"
    return {
        "task_id": d.name,
        "dir": str(d),
        "dockerfile": str(d / "environment" / "Dockerfile"),
        "build_context": str(d / "environment"),
        "tests_dir": str(d / "tests"),
        "instruction": instr.read_text() if instr.exists() else "",
    }


def list_tasks(task_root=DEFAULT_TASK_ROOT):
    root = Path(task_root)
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("task_")
                  and (p / "environment" / "Dockerfile").exists())


# --------------------------------------------------------------------------
# Container lifecycle (Endless Terminals: WORKDIR /home/user, reward.txt oracle)
# --------------------------------------------------------------------------


class TaskContainer:
    """Build image, start a persistent container, exec commands, run oracle.
    Same shape as termbench_adapter.TaskContainer but with ET's paths/oracle."""

    WORKDIR = "/home/user"

    def __init__(self, task, keep_image=True, verbose=False):
        self.t = load_task(task)
        self.task_id = self.t["task_id"]
        self.tag = f"et-{self.task_id.lower()}:rlvp"
        self.name = f"et-ep-{self.task_id}-{uuid.uuid4().hex[:8]}"
        self.keep_image = keep_image
        self.verbose = verbose
        self.started = False

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def start(self):
        b = sh(["docker", "build", "-t", self.tag, "-f", self.t["dockerfile"],
                self.t["build_context"]])
        if b.returncode != 0:
            raise RuntimeError(f"docker build failed: {b.stderr[-500:]}")
        r = sh(["docker", "run", "-d", "--rm", "--name", self.name,
                "-w", self.WORKDIR, self.tag, "sh", "-c", "sleep infinity"])
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr[-500:]}")
        self.started = True

    def exec(self, cmd: str, timeout=60):
        """Run one bash command in the persistent container. Returns
        (stdout, stderr, exit_code)."""
        try:
            ex = sh(["docker", "exec", "-w", self.WORKDIR, self.name,
                     "bash", "-lc", cmd], timeout=timeout)
            return ex.stdout, ex.stderr, ex.returncode
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout}s", 124

    def score(self) -> float:
        """Copy tests/, run the task's test.sh, read /logs/verifier/reward.txt."""
        sh(["docker", "exec", self.name, "mkdir", "-p", "/tests", "/logs/verifier"])
        if Path(self.t["tests_dir"]).exists():
            sh(["docker", "cp", self.t["tests_dir"] + "/.", f"{self.name}:/tests/"])
        o = sh(["docker", "exec", self.name, "bash", "/tests/test.sh"], timeout=600)
        reward = sh(["docker", "exec", self.name, "bash", "-lc",
                     "cat /logs/verifier/reward.txt 2>/dev/null || echo 0"]).stdout.strip()
        self._log("[oracle tail]\n" + "\n".join(
            ((o.stdout or "") + (o.stderr or "")).splitlines()[-5:]))
        return 1.0 if reward == "1" else 0.0

    def close(self):
        if self.started:
            sh(["docker", "rm", "-f", self.name])
        if not self.keep_image:
            sh(["docker", "rmi", "-f", self.tag])


# --------------------------------------------------------------------------
# Agent protocol / prompt (WORKDIR /home/user)
# --------------------------------------------------------------------------

SHELL_PROTOCOL = """
You are operating a bash shell inside a Linux container. WORKDIR is /home/user.
The shell session is persistent: cwd, env, and files persist across commands.

How to act: think briefly, then end your reply with EXACTLY one line:
Action: bash {"cmd": "your shell command here"}
When the task is fully done, end your reply with:
Action: submit {}
One action per reply. After each command you will see its stdout/stderr and exit code."""


def _system_prompt(instruction: str) -> str:
    return ("You are an expert software engineer working at a terminal.\n"
            "Task:\n" + instruction.strip() + "\n" + SHELL_PROTOCOL)


# --------------------------------------------------------------------------
# Episode driver (identical control flow to run_terminal_episode)
# --------------------------------------------------------------------------


def run_endless_episode(task, gen, tok, rule_mode="structural", max_steps=15,
                        keep_image=True, verbose=False):
    """Drive one Endless Terminals task as an RL episode; return a trainer-ready
    Episode (with .env = ShimEnv). Token bookkeeping matches termbench_adapter."""
    t = load_task(task)
    sys_prompt = _system_prompt(t["instruction"])

    tracker = RuleTracker(mode=rule_mode)
    ctr = TaskContainer(task, keep_image=keep_image, verbose=verbose)
    calls = []
    reward = 0.0
    try:
        ctr.start()
        first_obs = "Shell ready at /home/user. Begin."
        episode = Episode(env=None,
                          ids=_ids(tok, TEMPLATE.initial(sys_prompt, first_obs)))

        for _step in range(max_steps):
            g = gen(episode.ids)
            start = len(episode.ids)
            episode.ids.extend(g)
            turn = episode.n_turns
            episode.action_spans.append((start, start + len(g), turn))
            text = tok.decode(g, skip_special_tokens=True)
            call = parse_action(text)

            if call is None:
                obs = ('ERROR: could not parse action. End your reply with one line: '
                       'Action: bash {"cmd": "..."}  or  Action: submit {}')
                episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue
            if call.name == "submit":
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

            if len(episode.ids) > MAX_EPISODE_TOKENS - 200:
                break
            episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))

        reward = ctr.score()
    finally:
        ctr.close()

    episode.turn_violations = tracker.turn_violations
    episode.turn_discharges = tracker.turn_discharges
    episode.env = ShimEnv(reward, tracker, calls)
    episode.done = True
    return episode
