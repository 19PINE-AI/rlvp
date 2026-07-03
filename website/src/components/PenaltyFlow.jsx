import React, { useState } from 'react'

// A real-world phone-agent episode. A rule-checker looks at every action.
const STEPS = [
  { act: 'call user', sub: 'first attempt', kind: 'ok' },
  { act: 'call again', sub: 'no answer yet', kind: 'pen', tag: 'penalty · called too soon' },
  { act: 'authenticate', sub: 'collect DOB + last 4', kind: 'dis', tag: 'reward · verified identity first' },
  { act: 'resolve dispute', sub: 'update the account', kind: 'good', tag: 'result: resolved ✓' },
]

export default function PenaltyFlow() {
  const [rlvp, setRlvp] = useState(true)
  let penalties = 0, rewards = 0

  return (
    <div>
      <div className="toggle-row">
        <label className="switch" onClick={() => setRlvp(v => !v)}>
          <span className={`track ${rlvp ? 'on' : ''}`}><span className="knob" /></span>
          {rlvp ? 'Grade the path — reward the result AND check every step' : 'Score the result only'}
        </label>
      </div>

      <div className="flow">
        {STEPS.map((s, i) => {
          const active = rlvp && (s.kind === 'pen' || s.kind === 'dis')
          if (active && s.kind === 'pen') penalties += 1
          if (active && s.kind === 'dis') rewards += 1
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
                      : <div className="annot" style={{ color: 'var(--gray)' }}>? not checked</div>
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
          <span style={{ fontWeight: 800 }}>The rule-checker</span>
          <span className="kicker">a fixed set of yes/no rules over what the agent just did — checked the instant it acts, no matter how the call ends.</span>
        </div>
      )}

      <div className="grid two" style={{ marginTop: 18 }}>
        <div className="note">
          {rlvp ? (
            <><b>Grade the path.</b> The agent is nudged away from the exact bad step (calling too
            soon) in the exact situation it happened, and toward the good one (verifying identity
            first). These rules don't change whether the dispute gets resolved — so they're feedback a
            plain success score can never give.</>
          ) : (
            <><b>The path is invisible.</b> Scoring only the result, the agent sees one number at the
            end. Calling too soon and acting without authorization go unnoticed — and doing more of the
            forbidden thing can even <em>raise</em> the success rate.</>
          )}
        </div>
        <div className="stat card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div className="lbl">This call, the rule-checker fired</div>
          <div className="num" style={{ fontSize: 26, color: rlvp ? 'var(--ink)' : 'var(--gray)' }}>
            {rlvp ? <>
              <span style={{ color: 'var(--red)' }}>{penalties} penalty</span>
              <span style={{ color: 'var(--ink-soft)', fontWeight: 600, fontSize: 20 }}> · </span>
              <span style={{ color: 'var(--green)' }}>{rewards} reward</span>
            </> : '—'}
          </div>
          <div className="kicker">{rlvp ? 'one bad step flagged, one good step credited' : 'no path checking'}</div>
        </div>
      </div>
    </div>
  )
}
