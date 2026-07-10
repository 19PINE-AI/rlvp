# RLVP: Reinforcement Learning from Verifiable Penalties

[![Paper](https://img.shields.io/badge/arXiv-2607.07435-b31b1b.svg)](https://arxiv.org/abs/2607.07435)
[![Website](https://img.shields.io/badge/website-01.me%2Fresearch%2Frlvp-1f6feb.svg)](https://01.me/research/rlvp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code for **[Penalize the Path, Reward the Outcome](https://arxiv.org/abs/2607.07435)**
(arXiv:2607.07435) — Bojie Li (Pine AI), Noah Shi (University of Washington).

Dense, rule-derived process rewards for agentic RL, attached at the tool-call
boundary. **Premise (verifier asymmetry):** for long-horizon agents,
intermediate actions are hard to verify as *right* but easy to verify as
*wrong* — so use penalty-only deterministic rules (plus obligation-fulfillment
credits) as the dense channel, keep the sparse outcome reward as the only task
signal, and train R1-style GRPO with step-aware credit assignment. The headline
metric is pass^k / perfect^k (reliability), not pass@1.

## Result in one line

Outcome-only RL reaches 100% task success while violating outcome-neutral
workflow rules in 240/240 episodes; the hybrid credit recipe (`c3`) reaches
**perfect^8 = 1.00** — all 8/8 trials succeed *and* comply on held-out tasks —
with **no rules in the prompt**, beating the base model *with* rules in its
prompt. See [`results/REPORT.md`](results/REPORT.md) for the full writeup and
the four mechanisms (per-channel normalization, fulfillment credits, the
exploration wall, the compliance-only attractor).

## Repository layout

| Path | Contents |
| --- | --- |
| [`rlvp/`](rlvp/) | The method. `grpo.py` (trainer; credit variants outcome / c1 / c2 / c2pos / c3 / c4), `rollout.py` (batched multi-turn rollouts with exact token/turn bookkeeping), `envs/` (FileOps + CSOps deterministic tool-call domains, 4 penalty-only rules each), and `*_adapter.py` (Lean/miniF2F, τ²-bench, TerminalBench, SWE-smith/SWE-gym, WebArena). |
| [`scripts/`](scripts/) | Baseline grids, training drivers, k=8/k=16 evaluation, phase-diagram and probe scripts. |
| [`tests/`](tests/) | Rule-loophole suite + credit-assignment unit tests. |
| [`results/`](results/) | Raw run logs, configs, and evaluations behind every number. See [`results/README.md`](results/README.md). |
| [`paper/`](paper/) | LaTeX source and figure generators. See [`paper/README.md`](paper/README.md). |
| [`docs/`](docs/) | Research notes — design rationale, findings, ablations. See [`docs/README.md`](docs/README.md). |
| [`website/`](website/) | The interactive results site at [01.me/research/rlvp](https://01.me/research/rlvp). |

## Installation

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. Models are pulled from the HF hub (Qwen3-1.7B/4B/8B).
Some domains (Lean/miniF2F, SWE, τ²-bench) need external toolchains — see
[`requirements.txt`](requirements.txt) and the relevant adapter.

## Quickstart

```bash
# 1. Unit tests: rules + credit assignment (no GPU needed)
python3 tests/test_rules.py && python3 tests/test_credit.py

# 2. Base-model violation grid (the "outcome RL can't buy compliance" baseline)
python3 scripts/phase0_baseline.py

# 3. Full Phase-1 campaign: credit variants + evals (~3h on one RTX PRO 6000)
bash scripts/run_all.sh

# 4. Evaluate a checkpoint at k=8, no rules in the prompt
python3 scripts/eval_checkpoint.py results/run_c3/final c3_norules
```

Trained checkpoints are not committed (multi-GB LoRA adapters); regenerate them
with the driver scripts above.

## Citation

```bibtex
@article{li2026rlvp,
  title   = {RLVP: Penalize the Path, Reward the Outcome},
  author  = {Li, Bojie and Shi, Noah},
  journal = {arXiv preprint arXiv:2607.07435},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.07435}
}
```

## License

[MIT](LICENSE) © 2026 Bojie Li and Noah Shi.
