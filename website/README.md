# RLVP — project website

An illustrative React website for the paper **RLVP: Penalize the Path, Reward the Outcome**.
It explains the mechanism with interactive illustrations and lets readers navigate the paper's
real evaluation results and rollout cases.

## Run

```bash
cd website
npm install
npm run dev      # local dev server (Vite)
npm run build    # production build -> dist/
npm run preview  # serve the production build
```

Requires Node 18+ (developed on Node 22).

## What's on the page

1. **Mechanism** — an interactive group-rollout demo. Toggle the verifiable path channel and watch
   an all-fail group go from a dead update (every advantage zero) to a real gradient (the path
   scores supply within-group variance).
2. **Penalize the path** — an interactive phone-agent trajectory showing the deterministic rule
   engine attach a penalty `−λ` to a violation and a discharge `+β` to a met precondition, next to
   the outcome-only view where the path is invisible.
3. **Reward verified progress** — step through a real miniF2F proof (`mathd_algebra_109`); the
   remaining obligations fall and each verified drop pays a dense `+β`, while the outcome stays
   silent until the proof closes.
4. **Results** — per-experiment tabs (deployable constraints, TerminalBench harm, Lean sample
   efficiency), each with its evaluation setting and a baseline-vs-ours comparison.
5. **Evaluation cases** — an explorer over real rollouts: the agent's actual actions, the tool
   results it saw, and the exact turns where a deterministic rule was broken.

## Data provenance

Everything shown is extracted from the repository's real result dumps by `extract_data.py`, which
writes `src/data/paperData.json`:

- `results/tau2_cellc/*/traj.json` — airline customer-service episodes (Qwen3-4B) with per-turn
  semantic violations.
- `results/exp_selfcritic/traj/Qwen3-1_7B.json` — FileOps episodes; rule violations are derived
  from each transcript exactly as the rule engine would (read-before-overwrite, untested submit).
- `results/eval_c3_rules.json` / `results/eval_outcome_rules.json` — aggregate clean-rate and
  per-rule violation metrics (ours vs. outcome-only baseline).
- TerminalBench harm and the Lean/miniF2F matrix come from the paper's Tables 1 and 2.

To regenerate the data after new runs:

```bash
python3 extract_data.py
```
