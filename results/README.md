# Results

Raw result dumps behind every number in the paper and on the
[website](https://01.me/research/rlvp). Nothing here is hand-edited — the
figures, tables, and website are all generated from these files.

## Start here

- **[REPORT.md](REPORT.md)** — the seven findings and the method recipe, in prose.
- **[RESULTS_LOG.md](RESULTS_LOG.md)** — the full chronological data log.

## Layout

Per-training-run directories are named `run_<recipe>[_s<seed>]/` and contain:

- `config.json` — the exact run configuration
- `train_log.jsonl` — per-iteration training metrics (one JSON object per line)
- `final/`, `adapter*/` — LoRA checkpoints (**not committed**; regenerate by
  re-running — see the top-level README)

Aggregate evaluation files live at the `results/` root:

- `eval_<recipe>_rules.json` / `eval_<recipe>_norules.json` — held-out
  pass^k / perfect^k with and without rules in the prompt
- `eval_*_k16.json` — k=16 reliability sweeps

## Run-name conventions

| Prefix | Domain / experiment |
| --- | --- |
| `run_c1` … `run_c4` | credit-assignment variants (outcome / c1 / c2 / c2pos / c3 / c4) |
| `run_lean_*` | Lean / miniF2F sample-efficiency runs |
| `run_term_*` | TerminalBench harm runs |
| `run_flag_*` | flagship deployable-constraint runs |
| `tau2_cellc/`, `tau/` | τ²-bench airline customer-service episodes |
| `exp_selfcritic/` | self-critique vs verifiable-rules ablation |
| `phase_diagram/` | the "when do dense rewards help" phase-diagram sweep |
| `ec_f2p/` | fail-to-pass reachability probe |

Recipe suffixes: `outcome` (outcome-only baseline), `rlvp`/`aligned`
(penalize-the-path), `dapo` (a comparison optimizer), `_s7`/`_s11`/… (seed).
