#!/usr/bin/env python3
"""Load real miniF2F theorems (Lean4 / Mathlib) from the cloned miniF2F-lean4 repo.
Each Valid/*.lean is one `theorem <name> ... := by sorry`. We return the statement
(signature before :=) so the policy must supply the proof tactics.

easy_only keeps the algebra / number-theory / induction families (the tractable
subset agreed for the 4B->30B capability regime); drop it for the full set.
"""
import glob
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "minif2f_src", "MiniF2F")
# Namespaces the miniF2F files open so statements parse (log=Real.log, !=factorial...)
PREAMBLE = ("import Mathlib\nset_option maxHeartbeats 400000\n"
            "open BigOperators Real Nat Topology Rat")
_EASY = re.compile(r"^(mathd_algebra|mathd_numbertheory|induction_|mathd_)")
# tighter curation: algebra/numbertheory computations (ring/norm_num/linarith/
# decide-provable), DROP induction_* (need multi-step induction the 4B/30B can't
# reliably do) -- raises zero-shot success so GRPO gets a real outcome signal.
_ALGEBRA = re.compile(r"^(mathd_algebra|mathd_numbertheory)")


def load_minif2f(split="Valid", easy_only=True, algebra_only=False, max_len=320, n=None):
    out = []
    for f in sorted(glob.glob(os.path.join(_SRC, split, "*.lean"))):
        name = os.path.basename(f)[:-5]
        if algebra_only and not _ALGEBRA.match(name):
            continue
        if easy_only and not algebra_only and not _EASY.match(name):
            continue
        txt = open(f).read()
        m = re.search(r"(theorem\s+\w+.*?)\s*:=\s*by\s+sorry", txt, re.S)
        if not m:
            continue
        stmt = re.sub(r"\s+", " ", m.group(1).strip())
        if len(stmt) > max_len:        # skip giant multi-hypothesis statements
            continue
        out.append({"name": name, "statement": stmt, "preamble": PREAMBLE})
    return out[:n] if n else out


if __name__ == "__main__":
    thms = load_minif2f()
    print(f"loaded {len(thms)} easy miniF2F (Valid) theorems")
    for t in thms[:6]:
        print(f"  {t['name']}: {t['statement'][:90]}")
