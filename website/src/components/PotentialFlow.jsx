import React, { useState } from 'react'

// Real miniF2F theorem used in the paper (Valid/mathd_algebra_109).
const THEOREM = '(3a + 2b = 12), (a = 4)  ⊢  b = 0'
const PROOF = [
  { state: '3a+2b=12,  a=4  ⊢  b=0', left: 3, tactic: null },
  { state: '12+2b=12  ⊢  b=0', left: 2, tactic: 'subst h₁' },
  { state: '2b=0  ⊢  b=0', left: 1, tactic: 'norm_num' },
  { state: '∎  proved', left: 0, tactic: 'linarith' },
]

export default function PotentialFlow() {
  const [i, setI] = useState(0)
  const cur = PROOF[i]
  const rewardsSoFar = i // one small reward per verified step so far
  const finalReward = cur.left === 0 ? 1 : 0
  const done = cur.left === 0

  return (
    <div>
      <div className="card" style={{ padding: '12px 16px', background: 'var(--bg-soft)', marginBottom: 18 }}>
        <span className="kicker">A real math-proof task (miniF2F):&nbsp;</span>
        <code style={{ fontWeight: 700 }}>{THEOREM}</code>
      </div>

      <div className="grid two" style={{ alignItems: 'stretch' }}>
        <div className="card" style={{ padding: 20 }}>
          <div className="lbl" style={{ fontWeight: 700, color: 'var(--ink-soft)', marginBottom: 10 }}>Where the proof stands</div>
          <div className="step act" style={{ fontSize: 14 }}>{cur.state}</div>
          <div style={{ margin: '16px 0 6px', display: 'flex', justifyContent: 'space-between' }}>
            <span className="lbl">sub-goals still to prove</span>
            <strong style={{ color: done ? 'var(--green)' : 'var(--ink)' }}>{cur.left}</strong>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[0, 1, 2].map(k => (
              <div key={k} style={{
                flex: 1, height: 12, borderRadius: 6,
                background: k < cur.left ? 'var(--blue)' : 'var(--green-l)',
                transition: 'background .4s',
              }} />
            ))}
          </div>
          <div style={{ marginTop: 18, display: 'flex', gap: 10 }}>
            <button className="btn primary" disabled={done} onClick={() => setI(v => Math.min(v + 1, PROOF.length - 1))}>
              {done ? 'Proof finished ✓' : `Take the next step  (${PROOF[i + 1].tactic})`}
            </button>
            <button className="btn" onClick={() => setI(0)}>Reset</button>
          </div>
        </div>

        <div className="grid" style={{ gridTemplateColumns: '1fr', gap: 14 }}>
          <div className="cmp" style={{ display: 'grid' }}>
            <div className="col ours">
              <div className="lbl" style={{ color: 'var(--green-d)', fontWeight: 800 }}>Progress reward · every step</div>
              <div className="num g" style={{ fontSize: 30 }}>{rewardsSoFar > 0 ? `+${rewardsSoFar}` : '0'}</div>
              <div className="kicker">a small reward each time a sub-goal is cleared</div>
              <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                {PROOF.slice(1).map((p, k) => (
                  <span key={k} className={`chip ${k < i ? 'dis' : 'neu'}`} style={{ opacity: k < i ? 1 : 0.4, fontFamily: 'var(--sans)' }}>✓</span>
                ))}
              </div>
            </div>
          </div>
          <div className="cmp" style={{ display: 'grid' }}>
            <div className="col base">
              <div className="lbl" style={{ fontWeight: 800 }}>Final reward · only at the end</div>
              <div className="num" style={{ fontSize: 30, color: finalReward ? 'var(--green)' : 'var(--gray)' }}>
                {finalReward ? '+1' : '0'}
              </div>
              <div className="kicker">pays once, and only if the whole proof is finished</div>
            </div>
          </div>
        </div>
      </div>

      <div className="note" style={{ marginTop: 18 }}>
        <b>Why this helps.</b> The agent gets useful feedback long before it ever finishes a proof, so
        it learns from far fewer attempts. The catch: this only works where "getting closer" is
        measurable. On tasks like fixing code — where a patch either passes all the tests or none —
        there's no partial progress to reward, and this idea adds nothing.
      </div>
    </div>
  )
}
