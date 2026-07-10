# Easier agentic benchmarks — survey + decision (2026-07-02)

Motivation: SWE-bench/TerminalBench/OSWorld are too hard for 4B–30B open models
(paper's harm result is stuck at the ~0.1 success floor; SWE is a 0% vacuum;
probe_swegym_feas: 0/19 non-dask SWE-Gym instances verify without per-repo env
engineering). We need benchmarks where Qwen3 4B–30B gets 10–60% zero-shot, with
cheap LOCAL verifiable oracles, multi-step tool-call episodes, fast resets, and
surface for verifiable PATH penalties. Constraints: no pure reasoning benchmarks
(MATH/GSM8K); stay agentic RL; no pixel-based computer use. Box: one 96GB GPU,
Docker, KVM present (bare metal; `ubuntu` not yet in kvm group), 148G disk free.

Four web surveys (terminal / coding / web / Android), key numbers verified
against primary sources by the research agents.

## The shortlist that matters

| # | Benchmark | Domain | Tasks | Small-model band | Oracle | Penalty fit | Infra on this box |
|---|-----------|--------|-------|------------------|--------|-------------|-------------------|
| 1 | **Endless Terminals** (arXiv:2601.16443, Apache-2.0) | terminal | 3,255 procedural Docker | Qwen2.5-7B 10.7%→53.3% w/ vanilla PPO; 3B 4→18% | end-state completion tests, local | destructive-cmd, repeat-error, precondition — ports directly from our termbench_adapter | light containers; SkyRL runs 80–100/replica |
| 2 | **SWE-smith** (arXiv:2504.21798, MIT/CC-BY-SA) | coding | ~52k over 128+ repos | tunable (procedural/LM-injected bugs ≪ real issues); pilot needed | FAIL_TO_PASS tests, local | edited-test-file, submit-without-running-tests, destructive cmds — all regex/AST-checkable from mini-SWE-agent .traj | ONE image per repo (tens of GB for 10–20 repos), reset = git checkout |
| 3 | **ST-WebAgentBench** (arXiv:2410.06703, Apache-2.0) + WebArena-Lite | web | 375 tasks + 3,057 policy instances; 165+647 WA-Lite | Qwen3-4B ≈54%, 30B-A3B ≈58% on WA-Lite (WebServ); no CuP baselines for small models | deterministic task check + 9 rule-based trace evaluators (CuP metric) — NO LLM judge | **purpose-built**: consent, scope, strict-execution, hierarchy policies as machine-checkable trace rules | self-hosted WebArena Docker; disk is the risk (official rec 1TB; we have 148G — need subset: GitLab+SuiteCRM only) |
| 4 | **AndroidWorld** (arXiv:2405.14573, Apache-2.0) | Android | 116 parameterized (∞ variants) | 7B ~20–22%, 32B ~44% | adb device-state checks (sqlite/files/settings), deterministic local | mid-episode adb probes: no-irrelevant-toggles, no-data-deleted, confirm-before-destructive | AVD+KVM (present); MobileRL Docker images; 2–4GB RAM/emulator, 8–16 parallel |
| 5 | **WebShop** (2022, MIT) | web | 12k instructions | 7B 7.8%→75% (GiGPO) | attribute/price match, dense score | moderate (open-item-before-buy, budget) | trivial: single server, ms resets |
| 6 | **InterCode-Bash** (arXiv:2306.14898) | terminal | ~200 | 8B ~50% post-RL (LEAP) | filesystem-diff + md5, continuous | strong (fs-diff = natural destructive-action surface) | light Docker, Gym API |

Also viable: R2E-Gym (8.1k tasks, 300–500MB/task images — the 14B–30B tier;
DeepSWE/SkyRL precedent), SWE-rebench V2 (32k tasks WITH per-task difficulty
labels — use `meta.llm_score.difficulty_score` to select an in-band slice),
TMax/Harbor (14.6k terminal envs, difficulty control), AppWorld (no Docker, LOOP
32B LoRA-PPO on 72 tasks — smallest-infra multi-step env), B-MoCA (131 easy
Android tasks, snapshot resets — curriculum floor), AndroidLab (138 tasks, pure
XML/text mode; ~4.6% raw so needs SFT warm-up), SafeArena (250 harmful web tasks
— refusal channel).

Rejected: Terminal-Bench 2.0 (7–8B at 0–7% = floor), WebChoreArena (GPT-4o
6.8%), Commit0/DevBench/MLE-bench (too hard/LLM-judge/too slow), WorkArena
(cloud-gated ServiceNow), WebVoyager/Mind2Web-live (live web + LLM judge),
DigiRL env (VLM judge, pixels), single-shot codegen (BigCodeBench, KodCode,
LiveCodeBench, DebugBench — not episodic; KodCode also CC-BY-NC).

## Mapping to the paper's needs

1. **P1 — harm-bounding at real capability (the attackable gap).**
   Primary: **Endless Terminals**. Same episode shape as our TerminalBench
   harness (bash-in-Docker, end-state oracle), so termbench_adapter's rules
   (blind_destructive, repeat_error) port nearly unchanged; published band
   means success will NOT be at floor at 7–30B → "equal success, ~Nx fewer
   violations" becomes measurable where success is 30–60%, killing the
   "harmless because incapable" attack. The running probe_tb30b still decides
   whether original-TerminalBench-easy is trainable at 30B as a fallback.

2. **Intro-loop closure — engineering discipline on real coding.**
   **SWE-smith easy slice** (procedural bugs, 10–20 repos) + mini-SWE-agent
   scaffold: outcome=F2P tests, penalties = edited-test-file /
   submit-without-running-tests / destructive commands. This is the intro's
   story (green tests ≠ deployable) demonstrated on the intro's domain.
   Replaces the dead SWE-Gym path (one image per REPO, not per task).

3. **Stretch / next-paper-grade — RL against Completion-under-Policy.**
   **ST-WebAgentBench + WebArena-Lite**: nobody has RL-trained against CuP;
   its 9 deterministic trace evaluators ARE our penalty channel, on a
   benchmark where Qwen3-4B already solves ~54%. Highest upside, highest
   infra risk (disk: need the GitLab+SuiteCRM subset only; or WebServ's CoW
   reimplementation if its repo checks out).

4. **Generality (optional): AndroidWorld** — text-only a11y actions work with
   non-VL Qwen3; KVM is present on this box. A second real-world domain for
   the recipe if reviewers want breadth beyond terminal+code+web.

## Recommended order

ET pilot (probe 8B/30B zero-shot on ~30 sampled tasks; ~a day incl. harness
port) → SWE-smith pilot (build 5-repo slice, probe difficulty dial) →
decide P1 domain → full P1 run (outcome vs outcome+penalty, n>=3 seeds) →
ST-WebAgentBench as the follow-on contribution.

Full survey details (per-benchmark tables, sources): agent reports 2026-07-02;
key sources inline above.
