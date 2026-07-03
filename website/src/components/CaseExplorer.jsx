import React, { useState } from 'react'
import data from '../data/paperData.json'

const DOMAINS = [
  ['airline', 'Airline customer service', 'Qwen3-4B · τ²-bench airline'],
  ['fileops', 'File operations (sysadmin)', 'Qwen3-1.7B · FileOps'],
]

function highlightAction(a) {
  // pull the tool name (first token before "{") for a subtle highlight
  const m = a.match(/^([a-z_]+)\s*(\{[\s\S]*)?$/i)
  if (!m) return a
  return (<><span className="k">{m[1]}</span>{m[2] ? ' ' + m[2] : ''}</>)
}

function CaseDetail({ c }) {
  const violByTurn = {}
  for (const v of c.violations) (violByTurn[v.turn] ||= []).push(v)
  return (
    <div className="card" style={{ padding: 24 }}>
      <div className="verdict">
        <span className={`tag ${c.success ? 'clean' : 'viol'}`}>{c.success ? 'TASK SOLVED' : 'TASK FAILED'}</span>
        {c.violations.length > 0
          ? <span className="tag viol">{c.violations.length} path violation{c.violations.length > 1 ? 's' : ''}</span>
          : <span className="tag clean">path clean</span>}
        <span className="big" style={{ color: 'var(--ink-soft)', fontWeight: 600 }}>{c.steps.length} actions</span>
      </div>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>Goal</div>
      <p style={{ color: 'var(--ink-soft)' }}>{c.goal}</p>

      <div style={{ fontWeight: 700, margin: '16px 0 6px' }}>Trajectory</div>
      <div className="transcript">
        {c.steps.map((s, i) => {
          const turn = i + 1
          const flags = violByTurn[turn]
          return (
            <div className={`step ${flags ? 'flag' : ''}`} key={i}>
              <div className="n" title={flags ? 'rule violated at this step' : undefined}>{turn}</div>
              <div>
                <div className="act">{highlightAction(s.action)}</div>
                {s.result && <div className="res">↳ {s.result}</div>}
                {flags && flags.map((v, k) => (
                  <div className="flagbox" key={k}>
                    <b>Rule violated — {v.name}.</b> {v.desc}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>

      <div className="note" style={{ marginTop: 16 }}>
        {c.violations.length > 0 ? (
          <>Result-only training is blind to these violations — it only checks whether the task was
          solved. Grading the path puts a <b>penalty</b> on the flagged step, teaching the agent to
          avoid it while it is still pushed to solve the task.</>
        ) : (
          <>A clean run: the agent did the required checks before acting. This is the behavior that
          grading the path makes reliable.</>
        )}
      </div>
    </div>
  )
}

export default function CaseExplorer() {
  const [dom, setDom] = useState('airline')
  const cases = data.cases[dom]
  const [sel, setSel] = useState(0)
  const c = cases[Math.min(sel, cases.length - 1)]

  return (
    <section className="section" id="cases">
      <div className="wrap">
        <div className="eyebrow">Interactive</div>
        <h2>Navigate the real evaluation cases</h2>
        <p className="section-lead">
          These are actual runs from the evaluation — the agent's real actions, the tool results it
          saw, and the exact steps where a rule was broken.
        </p>

        <div className="tabs" style={{ margin: '22px 0' }}>
          {DOMAINS.map(([id, label, sub]) => (
            <button key={id} className={`tab ${dom === id ? 'active' : ''}`}
              onClick={() => { setDom(id); setSel(0) }}>
              {label}
            </button>
          ))}
        </div>

        <div className="explorer">
          <div>
            <div className="kicker" style={{ marginBottom: 8 }}>
              {DOMAINS.find(d => d[0] === dom)[2]} · {cases.length} cases
            </div>
            <div className="caselist">
              {cases.map((cc, i) => (
                <button key={cc.id} className={`caseitem ${i === sel ? 'active' : ''}`} onClick={() => setSel(i)}>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                    <span className={`tag ${cc.violations.length ? 'viol' : 'clean'}`} style={{ fontSize: 11 }}>
                      {cc.violations.length ? `${cc.violations.length} viol` : 'clean'}
                    </span>
                    <span className="kicker">case {i + 1}</span>
                  </div>
                  <div className="g">{cc.goal}</div>
                </button>
              ))}
            </div>
          </div>
          <CaseDetail c={c} />
        </div>
      </div>
    </section>
  )
}
