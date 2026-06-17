"""tau2-bench <-> RLVP adapter (Stage G).

Runs tau2 simulations with OUR HF policy as the agent (exact token
bookkeeping preserved for token-level credit), tau2's native environment,
user simulator (litellm -> local vLLM) and evaluator. Adds an incremental
rule tracker so each assistant turn carries violations/discharges.

Run from the .venv-tau2 interpreter? No — tau2 is py3.12-only, torch lives in
the system python. We therefore import tau2 from its venv site-packages path
appended to sys.path (pure-python package, version-compatible with 3.10's
pydantic? -> verified at import time; if not, the train script must run under
.venv-tau2 with torch installed there).
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid

from .envs.base import parse_action
from .rollout import TEMPLATE, Episode, _ids

# --------------------------------------------------------------------------
# Generation server: batches concurrent single-turn generation requests
# --------------------------------------------------------------------------


class GenServer:
    def __init__(self, model, tok, max_new_tokens=220, temperature=1.0, top_p=1.0,
                 batch_window_s=0.05, max_batch=16):
        import torch
        self.torch = torch
        self.model, self.tok = model, tok
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_p = temperature, top_p
        self.window, self.max_batch = batch_window_s, max_batch
        self.q: queue.Queue = queue.Queue()
        self.eot = tok.convert_tokens_to_ids(TEMPLATE.eot)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def generate(self, ids: list) -> list:
        """Blocking: returns generated ids ending with EOT."""
        ev = threading.Event()
        box = {}
        self.q.put((ids, box, ev))
        ev.wait()
        if "err" in box:
            raise box["err"]
        return box["out"]

    def stop(self):
        self._stop = True

    def _loop(self):
        torch = self.torch
        while not self._stop:
            try:
                first = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            t0 = time.time()
            while len(batch) < self.max_batch and (time.time() - t0) < self.window:
                try:
                    batch.append(self.q.get(timeout=self.window))
                except queue.Empty:
                    break
            try:
                maxlen = max(len(b[0]) for b in batch)
                inp = torch.full((len(batch), maxlen), self.pad, dtype=torch.long)
                attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
                for j, (ids, _, _) in enumerate(batch):
                    inp[j, maxlen - len(ids):] = torch.tensor(ids)
                    attn[j, maxlen - len(ids):] = 1
                dev = next(self.model.parameters()).device
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=inp.to(dev), attention_mask=attn.to(dev),
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=max(self.temperature, 1e-4), top_p=self.top_p,
                        eos_token_id=self.eot, pad_token_id=self.pad)
                gen = out[:, maxlen:].cpu()
                for j, (_, box, ev) in enumerate(batch):
                    g = gen[j].tolist()
                    if self.pad in g:
                        cut = g.index(self.pad)
                        g = g[:cut] if (cut == 0 or g[cut - 1] != self.eot) else g[:cut]
                    if self.eot in g:
                        g = g[:g.index(self.eot) + 1]
                    else:
                        g = g + [self.eot]
                    box["out"] = g
                    ev.set()
            except Exception as exc:  # propagate to all waiters
                for _, box, ev in batch:
                    box["err"] = exc
                    ev.set()


# --------------------------------------------------------------------------
# Structural rules for tau2 domains (incremental, fed each assistant turn)
# --------------------------------------------------------------------------

WRITE_PREFIXES = ("update_", "cancel_", "book_", "modify_", "send_", "transfer_")
READ_PREFIXES = ("get_", "search_", "list_", "calculate_")


# OUTCOME-INSTRUMENTAL rules compiled from the airline policy.md. The policy
# repeatedly states "the API does not check these, so the agent must make sure":
# obtain the user id (get_user_details) and the reservation (get_reservation_details)
# BEFORE any booking-DB modification, and confirm before writing. These are
# instrumental to the reward, not generic hygiene: a correct modification REQUIRES
# the looked-up reservation state, so the discharge credit pulls toward the
# productive workflow rather than penalizing action (closing off the
# compliance-only attractor that generic structural rules fell into).
ALIGN_WRITE = ("update_reservation_", "cancel_reservation", "book_reservation",
               "send_certificate")


class RuleTracker:
    """Per-turn violations/discharges. mode='structural' (generic hygiene),
    'aligned' (procedural, policy-derived), or 'semantic' (aligned procedural
    PLUS verifiable content-validity checks against the actual DB state:
    do not modify a basic-economy reservation, do not change the passenger
    count, do not use a payment method not in the user's profile---each a
    policy constraint the API does not enforce, so violating it guarantees the
    task fails. These cover what the reward REQUIRES, not just the workflow)."""

    def __init__(self, mode="structural", db=None):
        self.mode = mode
        self.db = db                 # FlightDB, for semantic content checks
        self.seen_read = False
        self.first_read_paid = False
        self.got_user = False        # get_user_details called (aligned)
        self.got_resv = False        # get_reservation_details called (aligned)
        self.user_paid = False
        self.resv_paid = False
        self.confirmed_since_write = False   # require explicit confirm before each write
        self.sig = {}
        self.last_was_write = False
        self.turn_violations = {}
        self.turn_discharges = {}

    def observe_turn(self, turn_idx: int, tool_calls: list, is_respond: bool):
        if self.mode in ("aligned", "semantic"):
            v, d = self._aligned(turn_idx, tool_calls, is_respond)
            if self.mode == "semantic":
                self._add_semantic(turn_idx, tool_calls, v)
            return v, d
        return self._structural(turn_idx, tool_calls, is_respond)

    def _add_semantic(self, turn_idx, tool_calls, v):
        """Append verifiable content-validity violations (checked against the DB).
        A modification that violates these policy constraints guarantees task
        failure, so penalizing it prunes failure-guaranteed actions."""
        if self.db is None:
            return
        extra = []
        for tc in tool_calls:
            name, a = tc["name"], (tc.get("arguments") or {})
            rid = a.get("reservation_id")
            resv = getattr(self.db, "reservations", {}).get(rid) if rid else None
            resv = (resv if isinstance(resv, dict)
                    else resv.__dict__ if resv is not None else None)
            if name.startswith(("update_reservation_", "cancel_reservation")) and resv:
                if str(resv.get("cabin", "")).lower() == "basic_economy":
                    extra.append("modify_basic_economy")          # policy: cannot modify
            if name == "update_reservation_passengers" and resv:
                cur = len(resv.get("passengers", []) or [])
                new = len(a.get("passengers", []) or [])
                if new and cur and new != cur:
                    extra.append("change_passenger_count")         # policy: count fixed
            # payment method must already be in the user profile
            if name in ("update_reservation_flights", "book_reservation"):
                uid = a.get("user_id") or (resv or {}).get("user_id")
                user = getattr(self.db, "users", {}).get(uid) if uid else None
                user = (user if isinstance(user, dict)
                        else user.__dict__ if user is not None else None)
                pid = a.get("payment_id") or (a.get("payment") or {}).get("id") if isinstance(a.get("payment"), dict) else a.get("payment_id")
                if user and pid and pid not in (user.get("payment_methods", {}) or {}):
                    extra.append("payment_not_in_profile")
        if extra:
            self.turn_violations[turn_idx] = self.turn_violations.get(turn_idx, []) + extra
            v.extend(extra)

    def _structural(self, turn_idx, tool_calls, is_respond):
        v, d = [], []
        if is_respond and self.last_was_write:
            self.last_was_write = False
            d.append("unconfirmed_chain")
        for tc in tool_calls:
            name = tc["name"]
            sig = name + "|" + json.dumps(tc["arguments"], sort_keys=True, default=str)
            if name.startswith(READ_PREFIXES):
                if not self.first_read_paid:
                    self.first_read_paid = True
                    d.append("write_before_lookup")
                self.seen_read = True
            is_write = name.startswith(WRITE_PREFIXES)
            if is_write and not self.seen_read:
                v.append("write_before_lookup")
            self.sig[sig] = self.sig.get(sig, 0) + 1
            if self.sig[sig] == 3:
                v.append("call_spam")
            if is_write and self.last_was_write:
                v.append("unconfirmed_chain")
            if is_write:
                self.last_was_write = True
        if v:
            self.turn_violations[turn_idx] = v
        if d:
            self.turn_discharges[turn_idx] = d
        return v, d

    def _aligned(self, turn_idx, tool_calls, is_respond):
        v, d = [], []
        if is_respond:
            self.confirmed_since_write = True     # a respond turn IS the confirmation
            if self.last_was_write:               # contacted user after a write
                self.last_was_write = False
        for tc in tool_calls:
            name = tc["name"]
            sig = name + "|" + json.dumps(tc["arguments"], sort_keys=True, default=str)
            # discharge the prerequisite lookups (the productive precursors)
            if name == "get_user_details" and not self.user_paid:
                self.user_paid = True; self.got_user = True; d.append("need_user")
            elif name == "get_user_details":
                self.got_user = True
            if name == "get_reservation_details" and not self.resv_paid:
                self.resv_paid = True; self.got_resv = True; d.append("need_reservation")
            elif name == "get_reservation_details":
                self.got_resv = True
            is_write = name.startswith(ALIGN_WRITE)
            if is_write:
                # modify the DB without the looked-up state the change depends on
                if not self.got_user:
                    v.append("modify_without_user")
                if not self.got_resv and name != "book_reservation":
                    v.append("modify_without_reservation")
                if not self.confirmed_since_write:
                    v.append("write_without_confirm")
                self.confirmed_since_write = False
                self.last_was_write = True
            self.sig[sig] = self.sig.get(sig, 0) + 1
            if self.sig[sig] == 3:
                v.append("repeat_error")
        if v:
            self.turn_violations[turn_idx] = v
        if d:
            self.turn_discharges[turn_idx] = d
        return v, d


# --------------------------------------------------------------------------
# HF-policy agent speaking tau2's protocol
# --------------------------------------------------------------------------

RESPONSE_PROTOCOL = """
How to act: think briefly, then end your reply with EXACTLY one line:
Action: tool_name {"arg": "value"}
To speak to the customer instead of using a tool:
Action: respond {"message": "..."}
One action per reply."""


class ShimEnv:
    """Adapts a tau2 simulation result to the trainer's env interface."""

    def __init__(self, reward, tracker):
        self._r = reward
        self.success = reward >= 0.999
        self.violations = [(t, r) for t, rs in tracker.turn_violations.items() for r in rs]
        self.discharges = [(t, r) for t, rs in tracker.turn_discharges.items() for r in rs]
        self.calls = []
        self.format_errors = 0

    def outcome_reward(self):
        return self._r


def run_one_sim(task, gen, tok, include_policy, user_llm, max_steps=30, rule_mode='structural'):
    """One tau2 simulation with our HF policy; returns a trainer-ready Episode."""
    from tau2.domains.airline.environment import get_environment
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    env = get_environment()
    Agent = make_policy_agent_class()
    agent = Agent(tools=env.get_tools(), domain_policy=env.policy,
                  gen=gen, tok=tok, include_policy=include_policy, rule_mode=rule_mode,
                  db=getattr(env.tools, 'db', None))
    # disable Qwen3 "thinking" on the user sim: thinking-only replies parse to
    # empty content and crash tau2's orchestrator on UserMessage.validate()
    user = UserSimulator(llm=user_llm, instructions=str(task.user_scenario),
                         llm_args={"temperature": 0.7, "max_tokens": 400,
                                   "extra_body": {"chat_template_kwargs":
                                                  {"enable_thinking": False}}})
    orch = Orchestrator(domain="airline", agent=agent, user=user,
                        environment=env, task=task, max_steps=max_steps)
    try:
        sim = orch.run()  # a single malformed user/agent turn must not kill the run
    except Exception as exc:
        print("orch error:", str(exc)[:120], flush=True)
        ep = agent.episode
        if ep is None or ep.n_turns == 0:
            return None
        ep.env = ShimEnv(0.0, agent.tracker)  # partial episode counts as a failure
        ep.done = True
        return ep
    try:
        ri = evaluate_simulation(sim, task, EvaluationType.ENV, solo_mode=False,
                                 domain="airline")
        reward = float(ri.reward)
    except Exception as exc:
        print("eval error:", str(exc)[:120], flush=True)
        reward = 0.0
    ep = agent.episode
    if ep is None or ep.n_turns == 0:
        return None
    ep.env = ShimEnv(reward, agent.tracker)
    ep.done = True
    return ep


def make_policy_agent_class():
    """Deferred import factory (tau2 must be importable)."""
    from tau2.agent.llm_agent import LLMAgent, LLMAgentState
    from tau2.data_model.message import (AssistantMessage, MultiToolMessage,
                                         SystemMessage, ToolCall, ToolMessage,
                                         UserMessage)

    class HFPolicyAgent(LLMAgent):
        def __init__(self, tools, domain_policy, gen, tok, include_policy=True, rule_mode='structural', db=None):
            self._gen, self._tok = gen, tok
            self.include_policy = include_policy
            self.episode = None
            self.tracker = RuleTracker(mode=rule_mode, db=db)
            super().__init__(tools=tools, domain_policy=domain_policy,
                             llm="local/hf-policy", llm_args={})

        @property
        def system_prompt(self) -> str:
            tool_lines = []
            for t in self.tools:
                try:
                    sch = t.openai_schema["function"]
                    params = ", ".join((sch.get("parameters") or {}).get("properties", {}))
                    tool_lines.append(f"  {sch['name']} {{{params}}} - {sch.get('description', '')[:140]}")
                except Exception:
                    tool_lines.append(f"  {getattr(t, 'name', t)}")
            policy = (f"Domain policy:\n{self.domain_policy}\n\n"
                      if self.include_policy else "")
            return (f"You are a customer service agent.\n{policy}"
                    f"Available tools:\n" + "\n".join(tool_lines) + RESPONSE_PROTOCOL)

        def _obs_text(self, message) -> str:
            if isinstance(message, MultiToolMessage):
                return "\n".join(f"Tool result: {tm.content}" for tm in message.tool_messages)
            if isinstance(message, ToolMessage):
                return f"Tool result: {message.content}"
            return f"Customer: {message.content}"

        def _generate_next_message(self, message, state):
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            obs = self._obs_text(message)
            if self.episode is None:
                self.episode = Episode(env=None,
                                       ids=_ids(self._tok, TEMPLATE.initial(self.system_prompt, obs)))
            else:
                self.episode.ids.extend(_ids(self._tok, TEMPLATE.cont(obs)))
            g = self._gen(self.episode.ids)
            start = len(self.episode.ids)
            self.episode.ids.extend(g)
            turn = self.episode.n_turns
            self.episode.action_spans.append((start, start + len(g), turn))
            text = self._tok.decode(g, skip_special_tokens=True)
            call = parse_action(text)
            if call is None or call.name == "respond":
                content = (call.args.get("message", "") if call else text.strip()[:400])
                self.tracker.observe_turn(turn, [], is_respond=True)
                self.episode.turn_violations = self.tracker.turn_violations
                self.episode.turn_discharges = self.tracker.turn_discharges
                return AssistantMessage(role="assistant", content=content or "Could you clarify?")
            self.tracker.observe_turn(
                turn, [{"name": call.name, "arguments": call.args}], is_respond=False)
            self.episode.turn_violations = self.tracker.turn_violations
            self.episode.turn_discharges = self.tracker.turn_discharges
            return AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id=str(uuid.uuid4())[:8], name=call.name,
                                     arguments=call.args)])

    return HFPolicyAgent
