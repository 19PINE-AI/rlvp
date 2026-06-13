"""Arm 2: SFT behavior-cloning control — train ONLY on rule-engine-synthesized
compliant episodes. If this alone matches RLVP's perfect^8, the RL apparatus
is unjustified. Cross-entropy on action tokens, same data budget order as RL.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

from rlvp.envs import ENVS, make_env
from rlvp.rollout import scripted_episode

IMPERFECT = "--imperfect" in sys.argv
OFFSET = 0
for i, a in enumerate(sys.argv):
    if a == "--seed-offset":
        OFFSET = int(sys.argv[i + 1])
MODEL = "Qwen/Qwen3-4B"
N_SEEDS = 400          # tasks per domain
EPOCHS = 2
LR = 1e-5
MICRO_TOKENS = 4096
OUT = ROOT / (("results/run_sftbc_imp" if IMPERFECT else "results/run_sftbc")
              + (f"_o{OFFSET}" if OFFSET else ""))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=10)

items = []
for domain in ("fileops", "csops"):
    for s in range(OFFSET, OFFSET + N_SEEDS):
        env = make_env(domain, s)
        e = scripted_episode(tok, env, ENVS[domain].compliant_script(env.task, IMPERFECT))
        assert not env.violations
        assert env.success != IMPERFECT  # perfect scripts succeed, imperfect must fail
        mask = torch.zeros(len(e.ids))
        for (a, b, _t) in e.action_spans:
            mask[a:b] = 1.0
        items.append((e.ids, mask))
items.sort(key=lambda x: len(x[0]))
print(f"{len(items)} scripted episodes", flush=True)

pad = tok.pad_token_id
device = "cuda"
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "train_log.jsonl", "a")
step = 0
model.train()
for ep_i in range(EPOCHS):
    i = 0
    while i < len(items):
        j, maxlen = i, 0
        while j < len(items):
            cand = max(maxlen, len(items[j][0]))
            if (j - i + 1) * cand > MICRO_TOKENS and j > i:
                break
            maxlen = cand
            j += 1
        mb = items[i:j]
        i = j
        B, L = len(mb), max(len(x[0]) for x in mb)
        ids = torch.full((B, L), pad, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        am = torch.zeros((B, L))
        for b, (seq, m) in enumerate(mb):
            ids[b, :len(seq)] = torch.tensor(seq)
            attn[b, :len(seq)] = 1
            am[b, :len(seq)] = m
        ids, attn, am = ids.to(device), attn.to(device), am.to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        lf = logits.float()
        logp = lf.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1)
        del lf
        m_t = am[:, 1:]
        loss = -(logp * m_t).sum() / m_t.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            rec = {"epoch": ep_i, "step": step, "loss": float(loss)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)

model.eval()
model.save_pretrained(OUT / "final")
tok.save_pretrained(OUT / "final")
print("SFT DONE", flush=True)
