"""Batched multi-turn rollouts with exact token bookkeeping.

We manage the Qwen3 chat format manually (instead of re-applying the chat
template per turn) so that the training sequence is EXACTLY the token stream
the policy saw and produced. Every generated token is tagged with its turn
index, which is what allows C2 to attach penalties to the violating tool
call's tokens.

Format (Qwen3, thinking disabled):
  <|im_start|>system\n{sys}<|im_end|>\n
  <|im_start|>user\n{usr}<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\n   <- generation starts here
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_EPISODE_TOKENS = 3000


@dataclass
class Episode:
    env: object
    ids: list = field(default_factory=list)        # full token stream
    action_spans: list = field(default_factory=list)  # (start, end, turn_idx) generated tokens
    turn_violations: dict = field(default_factory=dict)  # turn_idx -> [rule names]
    turn_discharges: dict = field(default_factory=dict)  # turn_idx -> [rule names]
    done: bool = False
    truncated: bool = False

    @property
    def n_turns(self):
        return len(self.action_spans)


def _ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def start_episode(tok, env, include_rules=False) -> Episode:
    text = (
        f"<|im_start|>system\n{env.system_prompt(include_rules)}<|im_end|>\n"
        f"<|im_start|>user\n{env.initial_user_msg()}<|im_end|>\n" + ASSISTANT_PREFIX
    )
    return Episode(env=env, ids=_ids(tok, text))


@torch.no_grad()
def run_episodes(model, tok, episodes, temperature=1.0, top_p=1.0,
                 max_new_tokens=200, gen_batch=64, progress=None):
    """Drive all episodes to completion. Modifies episodes in place."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(model.parameters()).device
    rounds = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        rounds += 1
        for i in range(0, len(active), gen_batch):
            chunk = active[i:i + gen_batch]
            maxlen = max(len(e.ids) for e in chunk)
            input_ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for j, e in enumerate(chunk):  # left padding
                input_ids[j, maxlen - len(e.ids):] = torch.tensor(e.ids)
                attn[j, maxlen - len(e.ids):] = 1
            out = model.generate(
                input_ids=input_ids.to(device), attention_mask=attn.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-4), top_p=top_p,
                eos_token_id=im_end, pad_token_id=pad,
            )
            gen = out[:, maxlen:].cpu()
            for j, e in enumerate(chunk):
                g = gen[j].tolist()
                if pad in g and g.index(pad) > 0 and im_end not in g[:g.index(pad)]:
                    g = g[:g.index(pad)]
                elif pad in g:
                    g = g[:g.index(pad) + 1] if g[g.index(pad) - 1] != im_end else g[:g.index(pad)]
                # trim everything after the first im_end
                if im_end in g:
                    g = g[:g.index(im_end) + 1]
                    ended = True
                else:
                    ended = False
                start = len(e.ids)
                e.ids.extend(g)
                if not ended:  # hit token limit: force-close the assistant turn
                    e.ids.append(im_end)
                turn_idx = e.n_turns
                e.action_spans.append((start, start + len(g), turn_idx))
                text = tok.decode(g, skip_special_tokens=True)
                res = e.env.step_text(text)
                if res.violations:
                    e.turn_violations[turn_idx] = list(res.violations)
                if res.discharges:
                    e.turn_discharges[turn_idx] = list(res.discharges)
                if e.env.done:
                    e.done = True
                elif len(e.ids) > MAX_EPISODE_TOKENS:
                    e.done = True
                    e.truncated = True
                    e.env.done = True
                else:
                    e.ids.extend(_ids(tok, "\n<|im_start|>user\n" + res.observation
                                      + "<|im_end|>\n" + ASSISTANT_PREFIX))
        if progress:
            progress(rounds, sum(1 for e in episodes if e.done), len(episodes))
    return episodes


def episode_stats(episodes):
    """Aggregate metrics over a list of finished episodes."""
    n = len(episodes)
    calls = sum(len(e.env.calls) for e in episodes)
    viol = sum(len(e.env.violations) for e in episodes)
    per_rule = {}
    for e in episodes:
        for _, rname in e.env.violations:
            per_rule[rname] = per_rule.get(rname, 0) + 1
    return {
        "n": n,
        "success": sum(e.env.success for e in episodes) / n,
        "clean": sum(1 for e in episodes if not e.env.violations) / n,
        "success_and_clean": sum(1 for e in episodes if e.env.success and not e.env.violations) / n,
        "viol_per_100_calls": 100.0 * viol / max(calls, 1),
        "viol_per_episode": viol / n,
        "per_rule": per_rule,
        "format_errors_per_ep": sum(e.env.format_errors for e in episodes) / n,
        "calls_per_ep": calls / n,
        "truncated": sum(e.truncated for e in episodes) / n,
    }
