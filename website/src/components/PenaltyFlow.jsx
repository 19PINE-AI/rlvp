import React, { useState } from 'react'

// A real-world phone-agent episode. Each action is checked by a deterministic rule engine.
const STEPS = [
  { act: 'call user', sub: 'first attempt', kind: 'ok' },
  { act: 'call again', sub: 'no answer yet', kind: 'pen', rule: 'do not re-call before the retry window', tag: '−λ over-call' },
  { act: 'authenticate', sub: 'collect DOB + last 4', kind: 'dis', rule: 'required precondition before any account action', tag: '+β precondition met' },
  { act: 'resolve dispute', sub: 'update the account', kind: 'good', tag: 'outcome +1' },
]

export default function PenaltyFlow() {
  const [rlvp, setRlvp] = useState(true)
  let pathScore = 0

  return (
    <div>
      <div className="toggle-row">
        <label className="switch" onClick={() => setRlvp(v => !v)}>
          <span className={`track ${rlvp ? 'on' : ''}`}><span className="knob" /></span>
          {rlvp ? 'RLVP · reward outcome + penalize path' : 'RLVR · reward the outcome only'}
        </label>
      </div>

      <div className="flow">
        {STEPS.map((s, i) => {
          const active = rlvp && (s.kind === 'pen' || s.kind === 'dis')
          if (active && s.kind === 'pen') pathScore -= 1
          if (active && s.kind === 'dis') pathScore += 1
          const cls = s.kind === 'good' ? 'good' : active && s.kind === 'pen' ? 'bad' : active && s.kind === 'dis' ? 'good' : ''
          return (
            <React.Fragment key={i}>
              <div style={{ flex: 1, minWidth: 128 }}>
                <div className={`node ${cls}`}>
                  {s.act}
                  <div className="sub">{s.sub}</div>
                </div>
                <div style={{ minHeight: 42, textAlign: 'center' }}>
                  {(s.kind === 'pen' || s.kind === 'dis') && (
                    rlvp
                      ? <div className={`annot ${s.kind}`}>{s.tag}</div>
                      : <div className="annot" style={{ color: 'var(--gray)' }}>? invisible</div>
                  )}
                  {s.kind === 'good' && <div className="annot dis">{s.tag}</div>}
                </div>
              </div>
              {i < STEPS.length - 1 && <div className="arrowc">→</div>}
            </React.Fragment>
          )
        })}
      </div>

      {rlvp && (
        <div className="card" style={{ padding: '12px 16px', marginTop: 6, background: 'var(--bg-soft)', display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 800 }}>Verifiable rule engine</span>
          <span className="kicker">a deterministic predicate over (state before the action, action) — checkable the instant it fires, independent of the outcome.</span>
        </div>
      )}

      <div className="grid two" style={{ marginTop: 18 }}>
        <div className="note">
          {rlvp ? (
            <><b>Penalize the path.</b> The gradient lowers the probability of the specific offending
            action (over-calling) in its context, and raises the compliant one (authenticating first).
            These rules are <em>outcome-neutral</em> — breaking them does not change whether the dispute
            is resolved — so this is signal the outcome reward can never provide.</>
          ) : (
            <><b>The path is invisible.</b> Outcome-only training sees one terminal reward. Over-calling
            and acting without authentication go unpenalized — in fact, doing more of the forbidden
            thing often <em>raises</em> the resolution rate.</>
          )}
        </div>
        <div className="stat card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div className="lbl">Path score this episode</div>
          <div className={`num ${rlvp ? (pathScore < 0 ? 'r' : 'g') : ''}`} style={{ color: rlvp ? undefined : 'var(--gray)' }}>
            {rlvp ? (pathScore > 0 ? `+${pathScore}` : pathScore) : '—'}
          </div>
          <div className="kicker">{rlvp ? 'net of one −λ over-call and one +β precondition' : 'no path channel'}</div>
        </div>
      </div>
    </div>
  )
}
