# RLVP: Reinforcement Learning from Verifiable Penalties

Dense, rule-derived process rewards for agentic RL, attached at the tool-call
boundary. Premise (verifier asymmetry): for long-horizon agents, intermediate
actions are hard to verify as *right* but easy to verify as *wrong* — so use
penalty-only deterministic rules (plus obligation-discharge credits) as the
dense channel, keep the sparse outcome reward as the only task signal, and
train R1-style GRPO with step-aware credit assignment. Headline metric is
pass^k / perfect^k (reliability), not pass@1.

This repo contains the local pilot: two deterministic tool-call environments,
a custom on-policy GRPO trainer with two-channel (scalar + token-attached)
credit, and the full experimental campaign on Qwen3-4B.

## Results in one line

Outcome-only RL reaches 100% task success while violating outcome-neutral
workflow rules in 240/240 episodes; the hybrid credit recipe (c3) reaches
**perfect^8 = 1.00** — all 8/8 trials succeed AND comply on held-out tasks —
with **no rules in the prompt**, beating the base model *with* rules in its
prompt. See `results/REPORT.md` for the full writeup and the four mechanisms
(per-channel normalization, discharge credits, the exploration wall, the
compliance-only attractor).

## Layout

- `RESEARCH_PLAN.md` — framing, hypotheses, cluster-scale design, pilot revisions
- `rlvp/envs/` — FileOps (terminal-style) and CSOps (customer-service) domains;
  4 penalty-only rules each + discharge definitions, pure functions of state
- `rlvp/rollout.py` — batched multi-turn rollouts, exact token/turn bookkeeping
- `rlvp/grpo.py` — trainer; credit variants: outcome / c1 / c2 / c2pos / c3 / c4
- `scripts/` — phase0 baseline grid, training, k=8 evaluation, full driver
- `tests/` — rule loophole suite + credit-assignment unit tests
- `results/` — phase0 grid, per-run train logs/configs, 16 final evals, REPORT.md

Trained checkpoints (7 x 7.6GB) are not committed; regenerate with
`bash scripts/run_all.sh` (~3h on one RTX PRO 6000).

## Running

```bash
python3 tests/test_rules.py && python3 tests/test_credit.py
python3 scripts/phase0_baseline.py          # base-model violation grid
bash scripts/run_all.sh                      # full Phase 1 (4 variants + evals)
python3 scripts/eval_checkpoint.py results/run_c3/final c3_norules
```

Requires: torch >= 2.x + CUDA, transformers; models pulled from HF hub
(Qwen3-1.7B/4B/8B).
