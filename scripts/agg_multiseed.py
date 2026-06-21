"""Aggregate the multi-seed + frozen-critic csops runs (Driver A).
Reports late viol/ep mean±std across seeds for each variant, plus critic P/R for
the self-critique variants. Tests: (multi-seed) is rule>>self-critique robust?
(frozen) does freezing the critic close the gap (=non-stationarity) or not (=FP)?
"""
import json
from pathlib import Path
from statistics import mean, pstdev

SEEDS = [11, 22, 33]
VARIANTS = [("c2nodis", "penalty-only RULE"),
            ("llmcritic", "live self-critic"),
            ("llmcriticfrozen", "frozen self-critic")]


def late_viol(tag, seed, k=4):
    p = Path(f"results/exp_sc_train_{tag}_s{seed}_csops/train_log.jsonl")
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if len(rs) < k:
        return None
    def g(r, key): return r.get("train", {}).get(key)
    late = rs[-k:]
    ve = mean(x for x in (g(r, "viol_per_episode") for r in late) if x is not None)
    cp = [g(r, "critic_precision") for r in late if g(r, "critic_precision") is not None]
    cr = [g(r, "critic_recall") for r in late if g(r, "critic_recall") is not None]
    return ve, (mean(cp) if cp else None), (mean(cr) if cr else None)


def main():
    print(f"{'variant':22s} {'late viol/ep (mean±std over seeds)':32s} {'critic P/R':12s} {'n seeds':7s}")
    print("-" * 78)
    for tag, desc in VARIANTS:
        ves, ps, rs_ = [], [], []
        for s in SEEDS:
            r = late_viol(tag, s)
            if r is None:
                continue
            ves.append(r[0])
            if r[1] is not None: ps.append(r[1])
            if r[2] is not None: rs_.append(r[2])
        if not ves:
            print(f"{tag:22s} (no runs yet)"); continue
        m = mean(ves); sd = pstdev(ves) if len(ves) > 1 else 0.0
        pr = f"{mean(ps):.2f}/{mean(rs_):.2f}" if ps else "-"
        print(f"{tag:22s} {m:.3f} ± {sd:.3f}{'':18s}".ljust(54) + f"{pr:12s} {len(ves)}")
    print("\ndesc:", "; ".join(f"{t}={d}" for t, d in VARIANTS))


if __name__ == "__main__":
    main()
