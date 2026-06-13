"""Minimal on-policy GRPO (Dr.GRPO-style: mean-centered, no std division) with
two-channel credit assignment for RLVP.

Credit variants:
  outcome - group-centered terminal reward on all action tokens (R1-style)
  c1      - penalties summed into the terminal reward, then group-centered
            (the "naive" baseline: dense verification, diluted credit)
  c2      - outcome channel as in `outcome`; penalty channel attached ONLY to
            the action tokens of the violating turn, raw scale, weight lambda
  c2pos   - sign-flipped c2: +lambda on every tool-call turn with no violation
            (predicted pathology: padding episodes with safe calls)

On-policy REINFORCE: one update per batch, no ratio/clip needed because the
update forward pass recomputes logprobs under the exact rollout policy.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .envs import ENVS, make_env
from .rollout import episode_stats, run_episodes, scripted_episode, start_episode


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    credit: str = "c2"               # outcome | c1 | c2 | c2pos
    lam: float = 0.5                 # penalty weight per violation
    beta: float = 0.5                # credit per discharged rule obligation
    c1_clip: float = 2.0             # per-episode penalty/credit clip for c1
    tasks_per_iter: int = 16         # split evenly across domains
    group_size: int = 8
    iters: int = 60
    lr: float = 6e-6
    inner_epochs: int = 3
    clip_eps: float = 0.2
    warmup: int = 5
    grad_clip: float = 1.0
    temp: float = 1.1
    gen_batch: int = 64
    micro_token_budget: int = 4096
    train_seed_lo: int = 0
    train_seed_hi: int = 500
    eval_every: int = 10
    eval_tasks: int = 24
    eval_k: int = 2
    eval_seed0: int = 1000
    include_rules_in_prompt: bool = False
    domains: tuple = ("fileops", "csops")
    out_dir: str = "results/run"
    data_seed: int = 7
    mix_scripted: bool = False    # Arm 1: 1 rule-engine-synthesized compliant ep per group
    anneal_at: int = 0            # Arm 3: lam,beta -> 0 after this iteration (0 = never)
    drop_rules: tuple = ()        # Arm 6: rules removed from TRAINING (eval keeps all)
    lora_r: int = 0               # Arm 8: LoRA rank for large models (0 = full FT)
    imperfect_scripts: bool = False  # mixing scripts are compliant but task-failing
    script_scalar: bool = True    # False: scripted eps get ZERO scalar advantage and
                                  # are excluded from the group baseline — they teach
                                  # only through the token-attached process channel
    strip_dropped_from_scripts: bool = False  # clean holdout: scripts must not
                                              # demonstrate dropped rules


def build_advantages(groups, cfg: TrainConfig):
    """groups: list of lists of Episode (one list per task). Returns per-episode
    (A_seq scalar, per_turn_adj dict turn->extra advantage)."""
    out = []
    for grp in groups:
        # with script_scalar=False, scripted episodes are invisible to the
        # scalar channel: excluded from the baseline, zero scalar advantage
        scalar_grp = grp if cfg.script_scalar else [e for e in grp if not e.scripted]
        if not scalar_grp:
            scalar_grp = grp
        if cfg.credit in ("outcome", "c2", "c2pos", "c4"):
            rs = {id(e): e.env.outcome_reward() for e in scalar_grp}
        elif cfg.credit in ("c1", "c3"):
            rs = {id(e): e.env.outcome_reward()
                  - cfg.lam * min(float(len(e.env.violations)), cfg.c1_clip)
                  + cfg.beta * min(float(len(e.env.discharges)), cfg.c1_clip)
                  for e in scalar_grp}
        else:
            raise ValueError(cfg.credit)
        mu = sum(rs.values()) / len(rs)
        base = [(rs[id(e)] - mu) if id(e) in rs else 0.0 for e in grp]
        for e, a in zip(grp, base):
            per_turn = {}
            if cfg.credit in ("c2", "c3", "c4"):
                for turn, rules in e.turn_violations.items():
                    per_turn[turn] = -cfg.lam * len(rules)
                # c4: discharge credit is outcome-gated — a compliance-only
                # episode that never finishes the task earns no process credit,
                # so the compliance-only attractor is unreachable.
                if cfg.credit != "c4" or e.env.success:
                    for turn, rules in e.turn_discharges.items():
                        per_turn[turn] = per_turn.get(turn, 0.0) + cfg.beta * len(rules)
            elif cfg.credit == "c2pos":
                viol_turns = set(e.turn_violations)
                for (s, t, turn) in e.action_spans:
                    if turn not in viol_turns:
                        per_turn[turn] = cfg.lam
            out.append((e, a, per_turn))
    return out


def update_policy(model, tok, adv_eps, cfg, optimizer, scheduler):
    """One REINFORCE update over all episodes. Returns metrics."""
    pad = tok.pad_token_id
    device = next(model.parameters()).device
    # Two channels with SEPARATE normalizers. The outcome advantage covers every
    # action token of an episode; penalties cover only the violating turn's
    # tokens. Normalizing both by total batch tokens makes the penalty gradient
    # vanish exactly when the outcome channel saturates (all-success groups), so
    # each channel is divided by its own token count before they are combined.
    raw = []
    for e, a_seq, per_turn in adv_eps:
        adv_out = torch.zeros(len(e.ids))
        adv_pen = torch.zeros(len(e.ids))
        mask = torch.zeros(len(e.ids))
        for (s, t, turn) in e.action_spans:
            adv_out[s:t] = a_seq
            adv_pen[s:t] = per_turn.get(turn, 0.0)
            mask[s:t] = 1.0
        raw.append((e.ids, adv_out, adv_pen, mask))
    n_out = sum(int(m.sum()) for _, _, _, m in raw)
    n_pen = sum(int((p != 0).sum()) for _, _, p, _ in raw)
    items = []
    for ids_, adv_out, adv_pen, mask in raw:
        adv = adv_out / max(n_out, 1) + adv_pen / max(n_pen, 1)
        if mask.sum() == 0 or adv.abs().sum() == 0:
            continue  # no learnable signal in this episode
        items.append((ids_, adv, mask))
    items.sort(key=lambda x: len(x[0]))
    if not items:
        return {"loss": 0.0, "entropy": 0.0}

    # pre-build microbatches (CPU tensors), reused across inner epochs
    micro = []
    i = 0
    while i < len(items):
        j, maxlen = i, 0
        while j < len(items):
            cand = max(maxlen, len(items[j][0]))
            if (j - i + 1) * cand > cfg.micro_token_budget and j > i:
                break
            maxlen = cand
            j += 1
        mb = items[i:j]
        i = j
        B, L = len(mb), max(len(x[0]) for x in mb)
        ids = torch.full((B, L), pad, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        adv = torch.zeros((B, L))
        am = torch.zeros((B, L))
        for b, (seq, a, m) in enumerate(mb):
            ids[b, :len(seq)] = torch.tensor(seq)
            attn[b, :len(seq)] = 1
            adv[b, :len(seq)] = a
            am[b, :len(seq)] = m
        micro.append({"ids": ids, "attn": attn, "adv": adv, "am": am, "old_logp": None})

    model.train()
    loss_sum, ent_sum, ent_n, gnorms = 0.0, 0.0, 0, []
    for epoch in range(cfg.inner_epochs):
        optimizer.zero_grad(set_to_none=True)
        for mb in micro:
            ids, attn = mb["ids"].to(device), mb["attn"].to(device)
            adv_t = mb["adv"][:, 1:].to(device)
            am_t = mb["am"][:, 1:].to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
            tgt = ids[:, 1:]
            lf = logits.float()
            logp = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1)
            del lf
            if epoch == 0:
                mb["old_logp"] = logp.detach().cpu()
                with torch.no_grad():  # entropy on detached logits, chunked
                    flat = logits.detach().reshape(-1, logits.shape[-1])
                    fm = am_t.reshape(-1).bool()
                    sel = flat[fm]
                    for c in range(0, sel.shape[0], 2048):
                        lp = F.log_softmax(sel[c:c + 2048].float(), dim=-1)
                        ent_sum += -(lp.exp() * lp).sum().item()
                    ent_n += int(fm.sum())
                    del flat, sel
            ratio = torch.exp(logp - mb["old_logp"].to(device))
            surr = torch.minimum(
                ratio * adv_t,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t,
            )
            loss = -(surr * am_t).sum()  # advantages are pre-normalized per channel
            loss.backward()
            loss_sum += loss.item()
            del logits, logp, ratio, surr
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)))
        optimizer.step()
        scheduler.step()
    model.eval()
    return {"loss": loss_sum, "entropy": ent_sum / max(ent_n, 1),
            "grad_norm": max(gnorms)}


def evaluate(model, tok, cfg, k=None, temp=0.7, include_rules=None):
    k = k or cfg.eval_k
    inc = cfg.include_rules_in_prompt if include_rules is None else include_rules
    out = {}
    for domain in cfg.domains:
        eps = []
        for s in range(cfg.eval_tasks):
            for _ in range(k):
                eps.append(start_episode(tok, make_env(domain, cfg.eval_seed0 + s), inc))
        run_episodes(model, tok, eps, temperature=temp, top_p=0.95, gen_batch=cfg.gen_batch)
        out[domain] = episode_stats(eps)
    return out


def train(cfg: TrainConfig):
    import random

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
    log_f = open(out_dir / "train_log.jsonl", "a")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16, device_map="cuda")
    if cfg.lora_r:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup)

    rng = random.Random(cfg.data_seed)
    seeds = list(range(cfg.train_seed_lo, cfg.train_seed_hi))

    for it in range(1, cfg.iters + 1):
        t0 = time.time()
        # ---- rollout ----
        groups, all_eps = [], []
        n_live = cfg.group_size - (1 if cfg.mix_scripted else 0)
        per_dom = cfg.tasks_per_iter // len(cfg.domains)
        for domain in cfg.domains:
            for s in rng.sample(seeds, per_dom):
                grp = [start_episode(tok, make_env(domain, s, drop_rules=cfg.drop_rules),
                                     cfg.include_rules_in_prompt)
                       for _ in range(n_live)]
                all_eps.extend(grp)
                if cfg.mix_scripted:
                    env_s = make_env(domain, s, drop_rules=cfg.drop_rules)
                    skip = cfg.drop_rules if cfg.strip_dropped_from_scripts else ()
                    grp = grp + [scripted_episode(
                        tok, env_s,
                        ENVS[domain].compliant_script(env_s.task, cfg.imperfect_scripts,
                                                      skip_rules=skip),
                        cfg.include_rules_in_prompt)]
                groups.append(grp)
        model.config.use_cache = True
        run_episodes(model, tok, all_eps, temperature=cfg.temp, top_p=1.0, gen_batch=cfg.gen_batch)
        roll_s = time.time() - t0
        st = episode_stats(all_eps)  # live episodes only
        # ---- update ----
        t1 = time.time()
        cfg_eff = cfg
        if cfg.anneal_at and it > cfg.anneal_at:
            from dataclasses import replace as _replace
            cfg_eff = _replace(cfg, lam=0.0, beta=0.0)
        adv_eps = build_advantages(groups, cfg_eff)
        model.config.use_cache = False
        m = update_policy(model, tok, adv_eps, cfg, optimizer, scheduler)
        model.config.use_cache = True
        rec = {"iter": it, "train": st, **m,
               "roll_s": round(roll_s, 1), "upd_s": round(time.time() - t1, 1)}
        # ---- periodic eval ----
        if it % cfg.eval_every == 0 or it == cfg.iters:
            rec["eval"] = evaluate(model, tok, cfg)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "train"} |
                         {"succ": st["success"], "viol100": st["viol_per_100_calls"],
                          "clean": st["clean"]}), flush=True)

    if cfg.lora_r:
        model = model.merge_and_unload()  # save a plain checkpoint for eval
    model.save_pretrained(out_dir / "final")
    tok.save_pretrained(out_dir / "final")
    log_f.close()
    return out_dir
