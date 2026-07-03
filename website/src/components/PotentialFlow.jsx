import React, { useState } from 'react'

// Real miniF2F theorem used in the paper (Valid/mathd_algebra_109).
const THEOREM = '(3a + 2b = 12), (a = 4)  ⊢  b = 0'
const PROOF = [
  { state: '3a+2b=12,  a=4  ⊢  b=0', obligations: 3, tactic: null },
  { state: '12+2b=12  ⊢  b=0', obligations: 2, tactic: 'subst h₁' },
  { state: '2b=0  ⊢  b=0', obligations: 1, tactic: 'norm_num' },
  { state: '∎  proved', obligations: 0, tactic: 'linarith' },
]
const BETA = 1

export default function PotentialFlow() {
  const [i, setI] = useState(0)
  const cur = PROOF[i]
  const potential = i * BETA // dense: one +β per verified drop so far
  const outcome = cur.obligations === 0 ? 1 : 0
  const done = cur.obligations === 0

  return (
    <div>
      <div className="card" style={{ padding: '12px 16px', background: 'var(--bg-soft)', marginBottom: 18 }}>
        <span className="kicker">miniF2F&nbsp; <code>mathd_algebra_109</code>:&nbsp;</span>
        <code style={{ fontWeight: 700 }}>{THEOREM}</code>
      </div>

      <div className="grid two" style={{ alignItems: 'stretch' }}>
        <div className="card" style={{ padding: 20 }}>
          <div className="lbl" style={{ fontWeight: 700, color: 'var(--ink-soft)', marginBottom: 10 }}>Proof state</div>
          <div className="step act" style={{ fontSize: 14 }}>{cur.state}</div>
          <div style={{ margin: '16px 0 6px', display: 'flex', justifyContent: 'space-between' }}>
            <span className="lbl">obligations remaining</span>
            <strong style={{ color: done ? 'var(--green)' : 'var(--ink)' }}>{cur.obligations}</strong>
          </div>
          {/* obligations bar (falls to zero) */}
          <div style={{ display: 'flex', gap: 6 }}>
            {[0, 1, 2].map(k => (
              <div key={k} style={{
                flex: 1, height: 12, borderRadius: 6,
                background: k < cur.obligations ? 'var(--blue)' : 'var(--green-l)',
                transition: 'background .4s',
              }} />
            ))}
          </div>
          <div style={{ marginTop: 18, display: 'flex', gap: 10 }}>
            <button className="btn primary" disabled={done} onClick={() => setI(v => Math.min(v + 1, PROOF.length - 1))}>
              {done ? 'Proof closed ✓' : `Apply  \`${PROOF[i + 1].tactic}\``}
            </button>
            <button className="btn" onClick={() => setI(0)}>Reset</button>
          </div>
        </div>

        <div className="grid" style={{ gridTemplateColumns: '1fr', gap: 14 }}>
          <div className="cmp" style={{ display: 'grid' }}>
            <div className="col ours">
              <div className="lbl" style={{ color: 'var(--green-d)', fontWeight: 800 }}>Aligned potential — dense</div>
              <div className="num g" style={{ fontSize: 30 }}>{potential > 0 ? `+${potential}β` : '0'}</div>
              <div className="kicker">every kernel-verified drop in obligations pays +β</div>
              <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                {PROOF.slice(1).map((p, k) => (
                  <span key={k} className={`chip ${k < i ? 'dis' : 'neu'}`} style={{ opacity: k < i ? 1 : 0.4 }}>
                    +β
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="cmp" style={{ display: 'grid' }}>
            <div className="col base">
              <div className="lbl" style={{ fontWeight: 800 }}>Outcome reward — sparse</div>
              <div className="num" style={{ fontSize: 30, color: outcome ? 'var(--green)' : 'var(--gray)' }}>
                {outcome ? '+1' : '0'}
              </div>
              <div className="kicker">pays once, only when the proof closes (∎)</div>
            </div>
          </div>
        </div>
      </div>

      <div className="note" style={{ marginTop: 18 }}>
        <b>Reachability is the gate.</b> On the early all-fail groups — where no rollout closes the
        proof — rollouts that get <em>further</em> (lower obligations) score higher, so the potential
        supplies within-group variance the outcome cannot. Where partial progress is <em>not</em>{' '}
        reachable (e.g. software repair, where every episode passes zero tests), the same potential is
        vacuous and centers out to nothing.
      </div>
    </div>
  )
}
