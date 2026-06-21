"""On-policy LLM self-critique: the policy model reflects on its OWN trajectory
and flags steps it judges to be mistakes, in its own domain language.

This is the engine behind two experiments:

  Exp 0 (offline gating) -- compare the critic's flagged turns against the
    deterministic rule oracle (env.turn_violations). If the on-policy model
    can't recover its own violations at decent precision/recall, then training
    on its critiques is strictly noisier than rules; measure it before spending
    a training run.

  Exp 1 (training)        -- use the critic's flagged turns as the dense
    per-turn penalty channel in GRPO (credit="llmcritic"), replacing the rule
    predicates. Same weights = policy = critic, so this is on-policy
    self-reflection, NOT distillation.

The critic is deliberately given ONLY the domain system prompt (tools + goal)
and the numbered transcript -- NO rule list (blind mode). The "rule-aware" mode
(told the rules) measures the detection CEILING: the gap between blind and
rule-aware separates "doesn't know the norm" from "can't apply a known norm".
"""
from __future__ import annotations

import json
import re

import torch

from .rollout import TEMPLATE


# --------------------------------------------------------------------------- #
# transcript reconstruction
# --------------------------------------------------------------------------- #
def build_transcript(e, tok) -> tuple[str, list]:
    """Render an episode as a numbered list of (action -> observation) steps.

    Returns (text, turn_ids) where turn_ids[i] is the episode turn_idx of the
    1-based Step (i+1) shown to the critic, so flags can be mapped back.
    """
    lines, turn_ids = [], []
    for k, (s, t, turn) in enumerate(e.action_spans):
        action = tok.decode(e.ids[s:t], skip_special_tokens=True).strip()
        # keep only the operative "Action: ..." tail if the model rambled
        m = list(re.finditer(r"Action:\s*\S+", action))
        if m:
            action = action[m[-1].start():].strip()
        obs = str(e.turn_obs.get(turn, "")).strip()
        if len(obs) > 400:
            obs = obs[:400] + " ...[truncated]"
        lines.append(f"Step {k + 1}:\n  action: {action}\n  result: {obs}")
        turn_ids.append(turn)
    return "\n".join(lines), turn_ids


CRITIC_SYS = (
    "You are reviewing the trajectory of an autonomous agent that just attempted "
    "a task using tools. You have the agent's domain instructions (its tools and "
    "goal) and a numbered list of the steps it took, each with the action and the "
    "result that came back.\n\n"
    "Carefully reflect on the trajectory. Identify steps where the agent took an "
    "action that was a MISTAKE for this kind of task: unsafe, premature, wasteful, "
    "out-of-order, or otherwise poor operating practice -- judging only from the "
    "trajectory and your own understanding of how this kind of work should be done. "
    "A step can be a mistake even if its result looked fine (no error). Do NOT flag "
    "a step merely because the overall task failed; flag the specific bad action.\n\n"
    "Reply with ONLY a JSON object on the last line of your response, of the form:\n"
    '{"mistakes": [{"step": <int>, "why": "<short reason>"}]}\n'
    "If the agent made no mistakes, reply {\"mistakes\": []}."
)

RULE_AWARE_EXTRA = (
    "\n\nWhen judging, apply these operating rules specifically:\n{rules}\n"
    "Flag every step that breaks one of these rules."
)


def _compose_prompt(domain, goal, transcript, n_steps, rules_block, mode):
    """Assemble the critic prompt from raw parts (shared by the live-Episode and
    the saved-record paths, so a fixed trajectory can be judged by any critic)."""
    sys = CRITIC_SYS
    if mode == "rule_aware" and rules_block:
        sys = CRITIC_SYS + RULE_AWARE_EXTRA.format(rules=rules_block)
    user = (f"Agent domain instructions:\n{domain}\n\n"
            f"Task given to the agent:\n{goal}\n\n"
            f"Trajectory ({n_steps} steps):\n{transcript}\n\n"
            "List the mistakes as specified.")
    return TEMPLATE.initial(sys, user)


def _rules_block(env):
    """The guideline text an env appends when include_rules=True (or '')."""
    domain = env.system_prompt(include_rules=False)
    full = env.system_prompt(include_rules=True)
    return full[len(domain):].strip() if full.startswith(domain) else full


def critic_prompt(e, tok, mode: str = "blind") -> tuple[str, list]:
    """Build the full critic prompt for one live episode. Returns (text, turn_ids)."""
    transcript, turn_ids = build_transcript(e, tok)
    text = _compose_prompt(e.env.system_prompt(include_rules=False),
                           e.env.initial_user_msg(), transcript, len(turn_ids),
                           _rules_block(e.env), mode)
    return text, turn_ids


def critic_prompt_from_record(rec: dict, mode: str = "blind") -> tuple[str, list]:
    """Build the critic prompt from a serialized trajectory record (no Episode /
    no env / no policy model needed) -- so the SAME trajectory can be judged by
    critics of any size. Record fields: domain_sys, goal, transcript, turn_ids,
    rules_block."""
    text = _compose_prompt(rec["domain_sys"], rec["goal"], rec["transcript"],
                           len(rec["turn_ids"]), rec.get("rules_block", ""), mode)
    return text, rec["turn_ids"]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_flags(text: str, n_steps: int) -> set:
    """Parse the critic's reply -> set of 1-based step numbers it flagged."""
    flags = set()
    # prefer a clean JSON object near the end
    obj = None
    for m in reversed(list(re.finditer(r"\{.*?\"mistakes\".*?\}\s*\]?\s*\}", text, re.S))):
        try:
            obj = json.loads(text[m.start():m.end()])
            break
        except Exception:
            continue
    if obj is None:  # broader: last balanced {...}
        last = text.rfind("{")
        while last != -1 and obj is None:
            try:
                obj = json.loads(text[last:])
            except Exception:
                last = text.rfind("{", 0, last)
    if isinstance(obj, dict) and isinstance(obj.get("mistakes"), list):
        for it in obj["mistakes"]:
            if isinstance(it, dict) and isinstance(it.get("step"), int):
                flags.add(it["step"])
            elif isinstance(it, int):
                flags.add(it)
    if not flags:  # last-ditch regex on "step": N
        for mm in re.finditer(r'"step"\s*:\s*(\d+)', text):
            flags.add(int(mm.group(1)))
    return {s for s in flags if 1 <= s <= n_steps}


# --------------------------------------------------------------------------- #
# generation (HF model, single-turn, batched, left-padded)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _generate(model, tok, prompts, max_new_tokens=400, batch=4, temperature=0.0):
    im_end = tok.convert_tokens_to_ids(TEMPLATE.eot)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = True
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = [tok(p, add_special_tokens=False).input_ids for p in chunk]
        maxlen = max(len(x) for x in enc)
        ids = torch.full((len(enc), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(enc), maxlen), dtype=torch.long)
        for j, x in enumerate(enc):
            ids[j, maxlen - len(x):] = torch.tensor(x)
            attn[j, maxlen - len(x):] = 1
        gen = model.generate(
            input_ids=ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=max(temperature, 1e-4), top_p=1.0,
            eos_token_id=im_end, pad_token_id=pad,
        )
        for j in range(len(enc)):
            g = gen[j, maxlen:].tolist()
            outs.append(tok.decode(g, skip_special_tokens=True))
    model.config.use_cache = use_cache
    if was_training:
        model.train()
    return outs


def label_episodes(model, tok, episodes, mode="blind", max_new_tokens=400,
                   batch=4, temperature=0.0):
    """Run the critic over each episode; set e.critic_turns (turn_idx set) and
    return per-episode flag detail. Skips episodes with no actions."""
    work = [e for e in episodes if e.action_spans]
    prompts, maps = [], []
    for e in work:
        p, turn_ids = critic_prompt(e, tok, mode)
        prompts.append(p)
        maps.append(turn_ids)
    replies = _generate(model, tok, prompts, max_new_tokens, batch, temperature)
    detail = []
    for e, turn_ids, reply in zip(work, maps, replies):
        flagged_steps = parse_flags(reply, len(turn_ids))
        flagged_turns = {turn_ids[s - 1] for s in flagged_steps}
        e.critic_turns = flagged_turns
        detail.append({"flagged_steps": sorted(flagged_steps),
                       "flagged_turns": sorted(flagged_turns),
                       "reply": reply})
    for e in episodes:
        if not e.action_spans:
            e.critic_turns = set()
    return detail


def episode_to_record(e, tok) -> dict:
    """Serialize a live episode into a critic-ready trajectory record (decouples
    rollout from critique, so any critic can later judge this exact trajectory)."""
    transcript, turn_ids = build_transcript(e, tok)
    return {
        "domain": type(e.env).__name__,
        "success": bool(e.env.success),
        "transcript": transcript,
        "turn_ids": turn_ids,
        "domain_sys": e.env.system_prompt(include_rules=False),
        "rules_block": _rules_block(e.env),
        "goal": e.env.initial_user_msg(),
        "gt_turns": {int(t): list(rs) for t, rs in e.turn_violations.items()},
        "turn_errors": sorted(int(t) for t in e.turn_errors),
    }


def label_records(model, tok, records, mode="blind", max_new_tokens=400,
                  batch=4, temperature=0.0):
    """Judge saved trajectory records with `model` as critic. Returns a list of
    flagged-turn sets (aligned to records)."""
    prompts, maps = [], []
    for r in records:
        p, turn_ids = critic_prompt_from_record(r, mode)
        prompts.append(p)
        maps.append(turn_ids)
    replies = _generate(model, tok, prompts, max_new_tokens, batch, temperature)
    out = []
    for turn_ids, reply in zip(maps, replies):
        steps = parse_flags(reply, len(turn_ids))
        out.append({turn_ids[s - 1] for s in steps})
    return out
