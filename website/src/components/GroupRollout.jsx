import React, { useState } from 'react'

// Four rollouts of ONE bug-fix task; all fail the hidden test (outcome = 0 each).
// With the path channel ON, they score differently -> within-group variance -> gradient.
const ROLLOUTS = [
  { desc: 'read files, run the tests', path: +2.0, chip: '+β ran tests', kind: 'dis' },
  { desc: 'edit, then rm -rf build/', path: -2.0, chip: '−λ destructive', kind: 'pen' },
  { desc: 'localize the bug + edit the fix', path: +1.0, chip: '+β progress', kind: 'dis' },
  { desc: 'overwrite the failing test', path: -1.5, chip: '−λ edited test', kind: 'pen' },
]

export default function GroupRollout() {
  const [on, setOn] = useState(true)
  const mean = ROLLOUTS.reduce((s, r) => s + r.path, 0) / ROLLOUTS.length
  const SC = 34 // px per unit advantage
  return (
    <div>
      <div className="toggle-row">
        <label className="switch" onClick={() => setOn(v => !v)}>
          <span className={`track ${on ? 'on' : ''}`}><span className="knob" /></span>
          Verifiable path channel {on ? 'ON' : 'OFF'}
        </label>
        <span className="kicker">Outcome is <strong>0</strong> for all four rollouts (the hidden test fails).</span>
      </div>

      <div className="rollout">
        {ROLLOUTS.map((r, i) => {
          const adv = on ? r.path - mean : 0
          const w = Math.abs(adv) * SC
          const color = adv > 0.01 ? 'var(--green)' : adv < -0.01 ? 'var(--red)' : 'var(--gray)'
          return (
            <React.Fragment key={i}>
              <div className="desc">
                <div><span className="path">$ </span>{r.desc}</div>
                <div style={{ marginTop: 4 }}>
                  <span className="tag base">outcome 0</span>{' '}
                  {on && <span className={`chip ${r.kind}`}>{r.chip}</span>}
                </div>
              </div>
              <div className="barwrap" title={`advantage = ${adv.toFixed(2)}`}>
                <div className="zero" />
                <div
                  className="bar"
                  style={{
                    background: color,
                    left: adv >= 0 ? '50%' : `calc(50% - ${w}px)`,
                    width: on ? Math.max(w, 3) : 3,
                    opacity: on ? 1 : 0.5,
                  }}
                />
              </div>
            </React.Fragment>
          )
        })}
      </div>

      <div style={{ marginTop: 18, display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="legend">
          <span className="sw"><span className="dot" style={{ background: 'var(--green)' }} />positive advantage</span>
          <span className="sw"><span className="dot" style={{ background: 'var(--red)' }} />negative advantage</span>
          <span className="sw"><span className="dot" style={{ background: 'var(--gray)' }} />zero (no gradient)</span>
        </div>
      </div>

      <div className="note" style={{ marginTop: 18 }}>
        {on ? (
          <><b>Path scores differ → within-group variance → gradient at 0% task success.</b> The
          advantage is each rollout's path score minus the group mean, so the group learns to
          repeat what earned <span style={{ color: 'var(--green)', fontWeight: 700 }}>+β</span> and
          avoid what earned <span style={{ color: 'var(--red)', fontWeight: 700 }}>−λ</span> — even
          though none of them solved the task.</>
        ) : (
          <><b>All-fail group → every advantage is 0 → dead update.</b> With only the outcome reward,
          all four rewards equal the group mean, so GRPO produces no gradient and the expensive
          rollouts are wasted.</>
        )}
      </div>
    </div>
  )
}
