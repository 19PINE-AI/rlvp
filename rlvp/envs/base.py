"""Base classes for RLVP tool-call environments.

An environment is a deterministic state machine over tool calls. Rules are
pure predicates over (env state BEFORE the call, the call); each violation
yields a fixed penalty. Outcome reward is computed only at episode end.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict
    raw: str = ""


@dataclass
class StepResult:
    observation: str
    done: bool = False
    violations: list = field(default_factory=list)  # rule names fired on this step
    discharges: list = field(default_factory=list)  # rule obligations discharged


ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{?)")


def _balanced_json(text: str, start: int) -> str | None:
    """Return the substring of `text` from `start` ('{') to its matching '}'."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_action(text: str) -> ToolCall | None:
    """Parse the LAST 'Action: tool_name {json args}' from a model response.

    Tolerates trailing junk after the closing brace (stray quotes, periods).
    """
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    name = m.group(1)
    args = {}
    if m.group(2):
        blob = _balanced_json(text, text.index("{", m.end(1)))
        if blob is None:
            return None
        try:
            args = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
    return ToolCall(name=name, args=args, raw=m.group(0))


class Rule:
    """A penalty-only rule: check(env, call) -> True if VIOLATED.

    Checked against env state BEFORE the call's effects are applied.
    """

    name: str = "rule"
    penalty: float = 1.0

    def check(self, env, call: ToolCall) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class ToolEnv:
    """Deterministic multi-turn tool environment."""

    max_turns: int = 12
    rules: list = []
    tool_names: tuple = ()
    # Auto-rule metadata: per-tool category + (optional) the arg naming the
    # tool's target object. The auto-rule engine derives the process signal
    # from THESE TAGS + the env's own error signals — no hand-written
    # predicates, no task knowledge (the R1-Zero-pure process reward).
    TOOL_CATEGORIES: dict = {}   # tool -> observe | mutate | verify | terminal | neutral
    TARGET_ARG: dict = {}        # tool -> arg name identifying its target object

    def __init__(self, task: dict, track_rules: bool = True, drop_rules: tuple = (),
                 guardrail: bool = False, auto_rules: bool = False):
        self.task = task
        self.track_rules = track_rules
        self.drop_rules = set(drop_rules)   # rules invisible during training (Arm 6)
        self.guardrail = guardrail          # runtime action masking (Arm 4)
        self.auto_rules = auto_rules        # capstone: derive rules from tool tags
        self._auto_obs_targets: set = set()
        self._auto_any_observe = False
        self._auto_unverified = False       # a mutate happened since last verify
        self._auto_err: dict = {}           # call signature -> error count
        self.blocked = 0
        self.turn = 0
        self.done = False
        self.success = False
        self.calls: list[ToolCall] = []          # all parsed calls, in order
        self.violations: list[tuple] = []        # (turn, rule_name)
        self.discharges: list[tuple] = []        # (turn, rule_name)
        self.format_errors = 0

    # -- to be implemented by subclasses -------------------------------------
    def system_prompt(self, include_rules: bool = False) -> str:
        raise NotImplementedError

    def initial_user_msg(self) -> str:
        raise NotImplementedError

    def apply(self, call: ToolCall) -> StepResult:
        """Apply tool effects and return observation. Sets self.done/success."""
        raise NotImplementedError

    def discharge_rules(self, call: ToolCall) -> list:
        """Rule names whose PENDING obligation this call discharges (checked
        against pre-call state). Default: none."""
        return []

    # -- auto-rule engine (R1-Zero-pure: tags + env errors, no predicates) ----
    def _auto_target(self, call):
        arg = self.TARGET_ARG.get(call.name)
        return str(call.args.get(arg)) if arg else None

    def _auto_observed(self, target):
        if target in self._auto_obs_targets:
            return True
        import os
        return os.path.dirname(target) in self._auto_obs_targets if target else False

    def _auto_check(self, call):
        """Universal meta-rules from (category, target_arg) + error signals."""
        cat = self.TOOL_CATEGORIES.get(call.name, "neutral")
        v, d = [], []
        sig = call.name + "|" + json.dumps(call.args, sort_keys=True, default=str)
        if self._auto_err.get(sig, 0) >= 2:        # repeat an env-errored call
            v.append("auto_repeat_error")
        # the "terminate-needs-verify" meta-rule only applies to a toolset that
        # HAS a verify step (e.g. code: edit->test->submit). Domains with no
        # verify tool (e.g. a phone call that is itself the goal) skip it.
        has_verify = "verify" in self.TOOL_CATEGORIES.values()
        tgt = self._auto_target(call)
        if cat == "mutate":                        # act on an unobserved target
            if tgt is not None and not self._auto_observed(tgt):
                v.append("auto_act_before_observe")
            elif tgt is None and not self._auto_any_observe:
                v.append("auto_act_before_observe")
        if has_verify and cat == "terminal" and self._auto_unverified:
            v.append("auto_unverified_terminal")
        if cat == "observe":                       # discharges
            if (tgt is not None and not self._auto_observed(tgt)) or \
               (tgt is None and not self._auto_any_observe):
                d.append("auto_act_before_observe")
        if has_verify and cat == "verify" and self._auto_unverified:
            d.append("auto_unverified_terminal")
        return v, d

    def _auto_update(self, call, errored):
        cat = self.TOOL_CATEGORIES.get(call.name, "neutral")
        if errored:
            sig = call.name + "|" + json.dumps(call.args, sort_keys=True, default=str)
            self._auto_err[sig] = self._auto_err.get(sig, 0) + 1
        if cat == "observe":
            self._auto_any_observe = True
            tgt = self._auto_target(call)
            if tgt is not None:
                self._auto_obs_targets.add(tgt)
        elif cat == "mutate":
            self._auto_unverified = True
        elif cat == "verify":
            self._auto_unverified = False

    # -- shared driver --------------------------------------------------------
    def step_text(self, model_text: str) -> StepResult:
        """Full step from raw model output: parse, check rules, apply."""
        self.turn += 1
        if self.turn >= self.max_turns:
            self.done = True
        call = parse_action(model_text)
        if call is None:
            self.format_errors += 1
            return StepResult(
                observation=(
                    "ERROR: could not parse action. End your reply with exactly one "
                    'line: Action: tool_name {"arg": "value"}'
                ),
                done=self.done,
            )
        if call.name not in self.tool_names:
            self.format_errors += 1
            return StepResult(
                observation=f"ERROR: unknown tool '{call.name}'. Tools: {', '.join(self.tool_names)}",
                done=self.done,
            )
        fired, disch = [], []
        if self.track_rules and self.auto_rules:
            try:
                fired, disch = self._auto_check(call)
            except Exception:
                fired, disch = [], []
        elif self.track_rules:
            for rule in self.rules:
                if rule.name in self.drop_rules:
                    continue
                try:
                    if rule.check(self, call):
                        fired.append(rule.name)
                except Exception:
                    pass
            try:
                disch = [d for d in self.discharge_rules(call) if d not in self.drop_rules]
            except Exception:
                disch = []
        if self.guardrail and fired:
            # runtime mask: reject the action, don't apply it, don't record a
            # violation (it was prevented) — the agent pays in turns instead
            self.blocked += 1
            return StepResult(
                observation="Guardrail: action blocked (violates: "
                            + ", ".join(fired) + "). Choose a different action.",
                done=self.done,
            )
        res = self.apply(call)
        if self.auto_rules:
            self._auto_update(call, errored=res.observation.startswith("ERROR"))
        self.calls.append(call)
        for rname in fired:
            self.violations.append((self.turn, rname))
        for rname in disch:
            self.discharges.append((self.turn, rname))
        res.violations = fired
        res.discharges = disch
        if self.done:
            res.done = True
        return res

    def outcome_reward(self) -> float:
        return 1.0 if self.success else 0.0
