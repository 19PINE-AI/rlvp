# Endless Terminals pilot (P1 harm-at-capability domain)

Working tree lives at `/home/ubuntu/benchmarks/endless-terminals` (task slice +
Docker images, not tracked). This dir holds the durable harness + report.

- `et_runner.py` — Docker oracle runner (gold/empty/cmds), no GPU. VALIDATED.
- `et_capability_probe.py` — GPU-gated zero-shot capability + penalty-surface probe.
- `PILOT_REPORT.md` — verdict (FEASIBLE) + measured oracle results.

Dataset: `obiwan96/endless-terminals` (HF, Apache-2.0). Download a slice with
`huggingface_hub.snapshot_download(..., allow_patterns=[f'{d}/**' ...])`.
