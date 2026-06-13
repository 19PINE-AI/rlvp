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


class RuleTracker:
    """violations/discharges per assistant turn, tau2 structural rules:
    R1 write_before_lookup (+discharge: first read call)
    R2 call_spam: identical (name,args) 3+ times
    R3 unconfirmed_chain (+discharge: user contact between writes)
    """

    def __init__(self):
        self.seen_read = False
        self.first_read_paid = False
        self.sig = {}
        self.last_was_write = False
        self.turn_violations = {}
        self.turn_discharges = {}

    def observe_turn(self, turn_idx: int, tool_calls: list, is_respond: bool):
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


def run_one_sim(task, gen, tok, include_policy, user_llm, max_steps=30):
    """One tau2 simulation with our HF policy; returns a trainer-ready Episode."""
    from tau2.domains.airline.environment import get_environment
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    env = get_environment()
    Agent = make_policy_agent_class()
    agent = Agent(tools=env.get_tools(), domain_policy=env.policy,
                  gen=gen, tok=tok, include_policy=include_policy)
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
        def __init__(self, tools, domain_policy, gen, tok, include_policy=True):
            self._gen, self._tok = gen, tok
            self.include_policy = include_policy
            self.episode = None
            self.tracker = RuleTracker()
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
