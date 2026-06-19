#!/usr/bin/env python3
"""Self-contained multi-step Lean 4 theorems for the RLVP Lean adapter.

These are provable with ONLY Lean 4 core (NO Mathlib), so they validate the
adapter immediately while the (slow) Mathlib build runs in the background. Each
needs a MULTI-STEP proof (2-5 tactics) so a per-tactic process signal matters:
an `errored_tactic` penalty and a `goal_progress` discharge fire mid-proof.

Each entry:
  name      : identifier
  statement : the theorem header up to (and excluding) `:=` — fed to
              LeanREPL.start_theorem, which appends `:= by sorry`.
  gold      : a known-good tactic sequence that reaches `done` (sanity check).

`gold` is for the sanity check only; the policy generates its own tactics.
"""

THEOREMS_SYNTH = [
    {
        "name": "and_swap",
        "statement": "theorem and_swap (p q : Prop) (h : p ∧ q) : q ∧ p",
        "gold": ["obtain ⟨hp, hq⟩ := h", "exact ⟨hq, hp⟩"],
    },
    {
        "name": "and_comm_iff",
        "statement": "theorem and_comm_iff (p q : Prop) : p ∧ q → q ∧ p",
        "gold": ["intro h", "exact ⟨h.2, h.1⟩"],
    },
    {
        "name": "or_swap",
        "statement": "theorem or_swap (p q : Prop) (h : p ∨ q) : q ∨ p",
        # one tactic per REPL call: `rcases` splits, then discharge each goal.
        "gold": ["rcases h with hp | hq", "exact Or.inr hp", "exact Or.inl hq"],
    },
    {
        "name": "imp_trans",
        "statement": "theorem imp_trans (p q r : Prop) (h1 : p → q) (h2 : q → r) : p → r",
        "gold": ["intro hp", "apply h2", "exact h1 hp"],
    },
    {
        "name": "modus_ponens",
        "statement": "theorem modus_ponens (p q : Prop) (hp : p) (hpq : p → q) : q",
        "gold": ["apply hpq", "exact hp"],
    },
    {
        "name": "nat_add_zero",
        "statement": "theorem nat_add_zero (n : Nat) : n + 0 = n",
        "gold": ["rw [Nat.add_zero]"],  # single core step; kept simple but non-rfl-named
    },
    {
        "name": "nat_succ_add",
        "statement": "theorem nat_succ_add (n m : Nat) : Nat.succ n + m = Nat.succ (n + m)",
        # single-call induction: branches inline so the whole tactic parses at once.
        "gold": ["induction m with | zero => rfl | succ k ih => rw [Nat.add_succ, Nat.add_succ, ih]"],
    },
    {
        "name": "nat_add_comm",
        "statement": "theorem nat_add_comm (a b : Nat) : a + b = b + a",
        "gold": ["omega"],
    },
    {
        "name": "le_trans3",
        "statement": "theorem le_trans3 (a b c : Nat) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c",
        "gold": ["omega"],
    },
    {
        "name": "double_neg_intro",
        "statement": "theorem double_neg_intro (p : Prop) (hp : p) : ¬¬p",
        "gold": ["intro hnp", "exact hnp hp"],
    },
    {
        "name": "exists_succ",
        "statement": "theorem exists_succ (n : Nat) : ∃ m, m = n + 1",
        "gold": ["exists n + 1"],
    },
    {
        "name": "and_intro_two",
        "statement": "theorem and_intro_two (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p",
        "gold": ["constructor", "exact hp", "exact ⟨hq, hp⟩"],
    },
    {
        "name": "nat_zero_add",
        "statement": "theorem nat_zero_add (n : Nat) : 0 + n = n",
        "gold": ["induction n with | zero => rfl | succ k ih => rw [Nat.add_succ, ih]"],
    },
    {
        "name": "iff_of_eq_bool",
        "statement": "theorem contrapos (p q : Prop) (h : p → q) : ¬q → ¬p",
        "gold": ["intro hnq hp", "exact hnq (h hp)"],
    },
    {
        "name": "le_refl_add",
        "statement": "theorem le_refl_add (a b : Nat) : a ≤ a + b",
        "gold": ["omega"],
    },
]


def load_synth_theorems():
    return THEOREMS_SYNTH


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lean_repl import LeanREPL

    print(f"{len(THEOREMS_SYNTH)} synthetic theorems. Sanity-checking gold sequences...\n")
    ok = 0
    with LeanREPL() as repl:
        for t in THEOREMS_SYNTH:
            st = repl.start_theorem(t["statement"])
            if st["error"] or st["proofState"] is None:
                print(f"  [STMT-ERR] {t['name']}: {str(st['error'])[:80]}")
                continue
            ps = st["proofState"]
            failed = None
            for tac in t["gold"]:
                r = repl.apply_tactic(ps, tac)
                if r["error"]:
                    failed = (tac, r["error"])
                    break
                ps = r["proofState"]
                if r["done"]:
                    break
            else:
                r = {"done": r["done"]}
            if failed:
                print(f"  [GOLD-FAIL] {t['name']} @ '{failed[0]}': {str(failed[1]).splitlines()[1] if len(str(failed[1]).splitlines())>1 else str(failed[1])[:80]}")
            elif r.get("done"):
                print(f"  [OK]   {t['name']}")
                ok += 1
            else:
                print(f"  [INCOMPLETE] {t['name']} (gold ran but goals remain)")
    print(f"\n{ok}/{len(THEOREMS_SYNTH)} gold sequences reach done.")
