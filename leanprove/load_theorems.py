#!/usr/bin/env python3
"""Theorem statement loader for the RLVP theorem-proving oracle.

Returns theorem statements to feed the per-tactic oracle (lean_repl.py).

Two sources:

1. SELF_CONTAINED: theorems provable with ONLY Lean 4 core (no Mathlib).
   These are what the pipeline is currently PROVEN on, because the Lean REPL
   needs no Mathlib build to run them (fast, frugal, works today). Each entry
   carries a known-good tactic and a known-bad tactic so the oracle can be
   exercised on both reward and penalty cases.

2. miniF2F: the real benchmark (244 test + 244 valid = 488 statements,
   yangky11/miniF2F-lean4). EVERY statement does `import Mathlib`, so running
   it requires a full Mathlib build (multi-GB, long compile). Deferred as a
   scaling cost; `minif2f_available()` reports whether a Mathlib-enabled REPL
   project has been set up. One captured statement is included as a reference.
"""
from pathlib import Path

LEANPROVE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Self-contained theorems (no Mathlib). statement = part before `:=`.
#    good_tactic discharges the goal; bad_tactic should error or not finish.
# ---------------------------------------------------------------------------
SELF_CONTAINED = [
    {
        "name": "two_plus_two",
        "statement": "theorem two_plus_two : 2 + 2 = 4",
        "good_tactic": "rfl",
        "bad_tactic": "omega'",          # unknown tactic
    },
    {
        "name": "nat_add_zero",
        "statement": "theorem nat_add_zero (n : Nat) : n + 0 = n",
        "good_tactic": "omega",
        "bad_tactic": "rfl ; rfl",       # extra goal -> error
    },
    {
        "name": "nat_comm",
        "statement": "theorem nat_comm (a b : Nat) : a + b = b + a",
        "good_tactic": "omega",
        "bad_tactic": "rfl",             # not definitionally equal -> error
    },
    {
        "name": "and_swap",
        "statement": "theorem and_swap (p q : Prop) (h : p ∧ q) : q ∧ p",
        # multi-step is exercised separately; single-shot good tactic:
        "good_tactic": "exact ⟨h.2, h.1⟩",
        "bad_tactic": "exact h",         # type mismatch -> error
    },
    {
        "name": "le_trans3",
        "statement": "theorem le_trans3 (a b c : Nat) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c",
        "good_tactic": "omega",
        "bad_tactic": "exact h1",        # type mismatch -> error
    },
]

# A multi-step proof to demonstrate per-tactic process signal (intermediate
# goals remain after step 1, proof completes after step 2).
MULTISTEP_DEMO = {
    "name": "and_swap_multistep",
    "statement": "theorem and_swap2 (p q : Prop) (h : p ∧ q) : q ∧ p",
    "steps": ["obtain ⟨hp, hq⟩ := h", "exact ⟨hq, hp⟩"],
}

# ---------------------------------------------------------------------------
# 2. miniF2F reference (requires Mathlib; deferred).
# ---------------------------------------------------------------------------
MINIF2F_INFO = {
    "repo": "https://github.com/yangky11/miniF2F-lean4",
    "n_test": 244,
    "n_valid": 244,
    "requires": "import Mathlib (full Mathlib build needed)",
    "sample_statement": (
        "theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) "
        "(hw : 0 ≤ w)\n"
        "    (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40)\n"
        "    (h2 : Real.log w / Real.log (x * y * z) = 12) : "
        "Real.log w / Real.log z = 60 := by sorry"
    ),
}


def minif2f_available() -> bool:
    """True only if a Mathlib-enabled REPL project has been built under leanprove."""
    return (LEANPROVE / "mathlib_repl" / ".lake" / "build").exists()


def load_theorems():
    """Return the active theorem set.

    If a Mathlib-enabled environment is available, the caller can extend this to
    stream miniF2F statements; until then we return the self-contained set the
    oracle is proven on.
    """
    return {
        "source": "minif2f" if minif2f_available() else "self_contained",
        "self_contained": SELF_CONTAINED,
        "multistep_demo": MULTISTEP_DEMO,
        "minif2f": MINIF2F_INFO,
        "minif2f_available": minif2f_available(),
    }


if __name__ == "__main__":
    import json
    data = load_theorems()
    print(f"active source: {data['source']}")
    print(f"self-contained theorems: {len(data['self_contained'])}")
    print(f"miniF2F available locally: {data['minif2f_available']} "
          f"({data['minif2f']['n_test']}+{data['minif2f']['n_valid']} statements, "
          f"{data['minif2f']['requires']})")
    print("\nself-contained set:")
    for t in data["self_contained"]:
        print(f"  - {t['name']}: {t['statement']}")
    print(f"\nminiF2F sample:\n{data['minif2f']['sample_statement']}")
