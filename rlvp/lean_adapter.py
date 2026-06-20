"""Lean theorem-proving <-> RLVP adapter.

Produces the SAME Episode shape as tau2_adapter.py (token-exact action_spans +
turn_violations + turn_discharges + a ShimEnv with .success/.violations/
.discharges/.outcome_reward()/.calls), but the "environment" is the Lean kernel
via the proven per-tactic oracle (leanprove/lean_repl.py).

Per step the policy emits one tactic; the oracle (`apply_tactic`) is the
verifiable procedure signal:

  errored_tactic  (penalty)   : the tactic returned a Lean error (parse / unknown
                                / kernel reject) — a wrong proof step.
  goal_progress   (discharge) : a VALID tactic that strictly DECREASED the number
                                of open goals — a productive proof step.
  no_progress     (penalty)   : a valid tactic that left the goal count unchanged
                                for >1 consecutive step (spinning). Optional; only
                                fires in rule_mode='structural'.

Terminal reward = 1.0 iff the proof reaches `done` (no goals remain), else 0.0.

The REPL process is reused across an episode's tactics (fast); a FRESH proof
state is opened per episode/theorem via start_theorem so episodes don't bleed
into each other.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .rollout import TEMPLATE, Episode, _ids
from .tau2_adapter import GenServer  # reuse the batched generation server
from .envs.base import parse_action

# Make the proven oracle importable.
_LEANPROVE = Path(__file__).resolve().parents[1] / "leanprove"
if str(_LEANPROVE) not in sys.path:
    sys.path.insert(0, str(_LEANPROVE))
from lean_repl import LeanREPL  # noqa: E402


# --------------------------------------------------------------------------
# Process-signal tracker (the Lean analogue of tau2's RuleTracker)
# --------------------------------------------------------------------------

class LeanRuleTracker:
    """Per-turn violations/discharges derived ENTIRELY from the REPL oracle.

    mode='structural': errored_tactic + goal_progress + no_progress.
    mode='aligned'   : goal_progress discharge ONLY (no penalties). The errored/
                       no_progress PENALTIES are misaligned with proving (you can cut
                       them by degenerating) and drove a compliance-attractor collapse
                       at 30B; the discharge (goals strictly decreased) is aligned.
    mode='outcome'   : no process signal (pure terminal reward baseline).
    """

    def __init__(self, mode="structural"):
        self.mode = mode
        self.prev_n_goals = None          # goal count before the current step
        self.stale_steps = 0              # consecutive valid steps w/o progress
        self._n_progress = 0              # E-A granularity: count of goal-decreases so far
        self.turn_violations = {}
        self.turn_discharges = {}

    def observe_turn(self, turn_idx, *, errored, n_goals, done):
        """Called once per tactic. `n_goals` is the goal count AFTER the tactic
        (ignored when errored). Returns (violations, discharges).

        Un-gameability sweep modes (each row = pre-registered cheapest gaming policy):
          aligned   : goal_progress discharge only        -- UNGAMEABLE (must decrease goals)
          structural: + errored & no_progress PENALTIES   -- gameable: avoid errors by inaction
          valid     : discharge ANY non-errored tactic    -- gameable: pad trivial valid no-ops
          noerror   : errored PENALTY only, no discharge   -- gameable: stop attempting
          outcome   : no process signal
        """
        v, d = [], []
        m = self.mode
        if m == "outcome":
            return v, d
        if errored:
            if m in ("structural", "noerror"):
                v.append("errored_tactic")
        else:
            prev = self.prev_n_goals
            if m == "valid":
                d.append("valid_tactic")               # GAMEABLE: any non-error
            elif prev is not None and n_goals < prev:
                self._n_progress += 1
                if m in ("aligned", "structural"):     # FINE potential: every -dPhi
                    d.append("goal_progress")
                elif m == "pot_mid" and self._n_progress == 1:  # MID: 1 milestone only
                    d.append("goal_progress")
                self.stale_steps = 0
            elif prev is not None and n_goals >= prev and not done:
                self.stale_steps += 1
                if self.stale_steps > 1 and m == "structural":
                    v.append("no_progress")
            else:
                self.stale_steps = 0
            self.prev_n_goals = n_goals
        if v:
            self.turn_violations[turn_idx] = v
        if d:
            self.turn_discharges[turn_idx] = d
        return v, d


# --------------------------------------------------------------------------
# ShimEnv: same shape the trainer consumes (mirrors tau2_adapter.ShimEnv)
# --------------------------------------------------------------------------

class LeanShimEnv:
    def __init__(self, reward, tracker, calls):
        self._r = reward
        self.success = reward >= 0.999
        self.violations = [(t, r) for t, rs in tracker.turn_violations.items() for r in rs]
        self.discharges = [(t, r) for t, rs in tracker.turn_discharges.items() for r in rs]
        self.calls = calls                # list of tactic strings actually applied
        self.format_errors = 0

    def outcome_reward(self):
        return self._r


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

LEAN_SYSTEM = (
    "You are proving a Lean 4 theorem. Emit ONE tactic per step as exactly one "
    'line: Action: tactic {"t": "..."}\n'
    "The current proof state (the open goals) is shown each turn. Apply tactics "
    "to discharge every goal. When no goals remain the proof is complete.\n"
    'Examples: Action: tactic {"t": "intro h"}  /  Action: tactic {"t": "omega"}'
    '  /  Action: tactic {"t": "exact \\u27e8h.2, h.1\\u27e9"}\n'
    "One action per reply."
)


def _goals_text(goals):
    if not goals:
        return "No goals remain."
    if len(goals) == 1:
        return "Current goal:\n" + goals[0]
    return f"Current goals ({len(goals)}):\n" + "\n\n".join(
        f"[goal {i+1}]\n{g}" for i, g in enumerate(goals))


def _initial_obs(goal):
    return "Prove this Lean 4 theorem.\n" + _goals_text([goal] if goal else [])


# --------------------------------------------------------------------------
# Episode driver
# --------------------------------------------------------------------------

def run_lean_episode(theorem, gen, tok, rule_mode="structural", max_steps=10,
                     error_limit=3, repl=None):
    """Roll out one proof attempt and return a trainer-ready Episode.

    theorem    : dict with at least {"name", "statement"} (statement = header up
                 to `:=`, as in theorems_synth / load_theorems).
    gen        : callable(ids)->ids (e.g. GenServer.generate).
    tok        : tokenizer.
    rule_mode  : 'structural' (errored_tactic/goal_progress/no_progress) or
                 'outcome' (no process signal).
    max_steps  : hard cap on tactic emissions.
    error_limit: abort (reward 0) after this many errored tactics.
    repl       : an open LeanREPL to reuse; if None, one is created+closed here.

    Token bookkeeping is identical to tau2_adapter: each generated turn appends
    an (start, end, turn_idx) action span over the EXACT generated tokens.
    """
    own_repl = repl is None
    if own_repl:
        repl = LeanREPL()
    tracker = LeanRuleTracker(mode=rule_mode)
    calls = []
    try:
        try:
            st = repl.start_theorem(theorem["statement"])
        except Exception:
            st = {"error": "REPL failed to open theorem", "proofState": None}
        # Statement itself failed to open a goal -> unprovable rollout, reward 0.
        if st.get("error") or st.get("proofState") is None:
            ep = Episode(env=None,
                         ids=_ids(tok, TEMPLATE.initial(LEAN_SYSTEM, _initial_obs(None))))
            ep.env = LeanShimEnv(0.0, tracker, calls)
            ep.done = True
            return ep

        proof_state = st["proofState"]
        goals = [st["goal"]]
        tracker.prev_n_goals = len(goals)

        episode = Episode(env=None,
                          ids=_ids(tok, TEMPLATE.initial(LEAN_SYSTEM, _initial_obs(st["goal"]))))
        reward = 0.0
        n_errors = 0

        for step in range(max_steps):
            g = gen(episode.ids)
            start = len(episode.ids)
            episode.ids.extend(g)
            turn = episode.n_turns
            episode.action_spans.append((start, start + len(g), turn))

            text = tok.decode(g, skip_special_tokens=True)
            call = parse_action(text)

            # Parse / format failure: treat as an errored step (penalty), feed
            # back a format reminder, keep the same proof state.
            if call is None or call.name != "tactic" or "t" not in call.args:
                tracker.observe_turn(turn, errored=True, n_goals=len(goals), done=False)
                episode.turn_violations = dict(tracker.turn_violations)
                episode.turn_discharges = dict(tracker.turn_discharges)
                n_errors += 1
                if n_errors >= error_limit or step == max_steps - 1:
                    break
                obs = ('ERROR: emit exactly one line: Action: tactic {"t": "<tactic>"}\n'
                       + _goals_text(goals))
                episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))
                continue

            tactic = str(call.args["t"]).strip()
            calls.append(tactic)
            try:
                r = repl.apply_tactic(proof_state, tactic)
            except Exception:
                # REPL hung (tactic timeout -> process killed) or died. Score this
                # step as an errored tactic (penalty) and end the episode; the
                # trainer's _repl() rebuilds the dead process for the next episode.
                tracker.observe_turn(turn, errored=True, n_goals=len(goals), done=False)
                n_errors += 1
                episode.turn_violations = dict(tracker.turn_violations)
                episode.turn_discharges = dict(tracker.turn_discharges)
                break

            errored = r["error"] is not None
            if errored:
                tracker.observe_turn(turn, errored=True, n_goals=len(goals), done=False)
                n_errors += 1
                feedback = (str(r["error"]).splitlines() or ["Lean error"])
                err_line = feedback[1] if len(feedback) > 1 else feedback[0]
                obs = f"Lean error: {err_line}\n(proof state unchanged)\n" + _goals_text(goals)
            else:
                proof_state = r["proofState"] if r["proofState"] is not None else proof_state
                goals = r["goals"] or []
                tracker.observe_turn(turn, errored=False, n_goals=len(goals), done=r["done"])
                if r["done"]:
                    reward = 1.0
                    episode.turn_violations = dict(tracker.turn_violations)
                    episode.turn_discharges = dict(tracker.turn_discharges)
                    break
                obs = "Step accepted.\n" + _goals_text(goals)

            episode.turn_violations = dict(tracker.turn_violations)
            episode.turn_discharges = dict(tracker.turn_discharges)

            if n_errors >= error_limit or step == max_steps - 1:
                break
            episode.ids.extend(_ids(tok, TEMPLATE.cont(obs)))

        episode.env = LeanShimEnv(reward, tracker, calls)
        episode.done = True
        return episode
    finally:
        if own_repl:
            repl.close()
