# WebArena / ST-WebAgentBench adapter (stretch: RL vs Completion-under-Policy)

Working tree: `/home/ubuntu/benchmarks/webarena/{webarena,BrowserGym,ST-WebAgentBench}`.

- `rlvp/webarena_adapter.py` — Episode/ShimEnv adapter over BrowserGym. Outcome
  reward = task oracle; penalty channel = ST-WebAgentBench per-step
  `info['safety_report']` (9 non-LLM policy evaluators). browsergym is LAZILY
  imported so the module is safe to import without Playwright/gymnasium.
  OFFLINE-VALIDATED against the real env source (step signature, safety_report,
  AXTree, HighLevelActionSet, extract_action). Live env loop NOT yet exercised.
- `scripts/webarena_train.py` — trainer (mirror of endless_train). Needs
  `--task-ids`.
- `benchmarks/webarena/validate_env.py` — turnkey LIVE smoke test (no GPU): run
  after booting a site + installing browsergym, asserts reset/step/safety_report.

## Status: GATED OFF in the overnight orchestrator (Phase D not enabled)
Bringing up the full RL loop is a deliberate, non-trivial step and was NOT
auto-run overnight (would risk the concurrently-running training jobs and needs
a ~45-min GitLab bringup). To enable:
  1. `bash scripts/setup_minimal_stwab.sh`  (boot GitLab/shopping_admin/SuiteCRM)
  2. venv with browsergym.stwebagentbench + playwright chromium **and** torch
     (the policy runs in-process; if deps conflict, split env-server/policy-client)
  3. `python3 benchmarks/webarena/validate_env.py <shopping_admin_task_id>`
  4. if it passes: `python3 scripts/webarena_train.py 20 --task-ids <...> --credit c3`
     (and `--credit outcome` for the baseline arm). Exclude vision tasks 295-334
     and the 31 fuzzy-oracle tasks.
