import React, { useState } from 'react'

// Four attempts at ONE bug-fix task; all fail the hidden test (result = 0 each).
// With step-grading ON, they score differently -> the group can still learn.
const ROLLOUTS = [
  { desc: 'read the files, then run the tests', path: +2.0, chip: '✓ ran the tests', kind: 'dis' },
  { desc: 'edit, then delete the build folder', path: -2.0, chip: '✕ destructive command', kind: 'pen' },
  { desc: 'find the bug and edit the fix', path: +1.0, chip: '✓ real progress', kind: 'dis' },
  { desc: 'overwrite the failing test', path: -1.5, chip: '✕ gamed the test', kind: 'pen' },
]

export default function GroupRollout() {
  const [on, setOn] = useState(true)
  const mean = ROLLOUTS.reduce((s, r) => s + r.path, 0) / ROLLOUTS.length
  const SC = 34 // px per unit of score difference
  const sansChip = { fontFamily: 'var(--sans)' }
  return (
    <div>
      <div className="toggle-row">
        <label className="switch" onClick={() => setOn(v => !v)}>
          <span className={`track ${on ? 'on' : ''}`}><span className="knob" /></span>
          Grade each step of the path: {on ? 'ON' : 'OFF'}
        </label>
        <span className="kicker">All four attempts fail the hidden test — the result is a <strong>0</strong> for every one.</span>
      </div>

      <div className="rollout">
        {ROLLOUTS.map((r, i) => {
          const adv = on ? r.path - mean : 0
          const w = Math.abs(adv) * SC
          const color = adv > 0.01 ? 'var(--green)' : adv < -0.01 ? 'var(--red)' : 'var(--gray)'
          return (
            <React.Fragment key={i}>
              <div className="desc">
                <div><span className="path">attempt {i + 1}: </span>{r.desc}</div>
                <div style={{ marginTop: 4 }}>
                  <span className="tag base">failed</span>{' '}
                  {on && <span className={`chip ${r.kind}`} style={sansChip}>{r.chip}</span>}
                </div>
              </div>
              <div className="barwrap" title={`score vs. group average = ${adv.toFixed(2)}`}>
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

      <div style={{ marginTop: 18 }}>
        <div className="legend">
          <span className="sw"><span className="dot" style={{ background: 'var(--green)' }} />helps — a good move</span>
          <span className="sw"><span className="dot" style={{ background: 'var(--red)' }} />hurts — a bad move</span>
          <span className="sw"><span className="dot" style={{ background: 'var(--gray)' }} />no signal</span>
        </div>
      </div>

      <div className="note" style={{ marginTop: 18 }}>
        {on ? (
          <><b>Good moves score higher, bad moves lower — so the group still improves,</b> even though
          none of the four solved the task. The agent learns to repeat what helped (running the tests,
          making progress) and avoid what hurt (destructive commands, gaming the test). The bars show
          each attempt's step-score minus the group average.</>
        ) : (
          <><b>With only a success-or-fail score, all four are tied at 0.</b> No differences means no
          signal — the four expensive attempts teach the agent nothing, and the update is wasted.</>
        )}
      </div>
    </div>
  )
}
