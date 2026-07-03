import React, { useEffect, useState } from 'react'

// A looping "graded call" scene: an agent takes steps, and each step gets a plain-language
// verdict as it happens. Shows the core idea — grade the path, not just the result — with no jargon.
const STEPS = [
  { ico: '📞', lbl: 'Call the customer', verdict: null },
  { ico: '📞', lbl: 'Call again — still no answer', verdict: { ok: false, text: 'not allowed · called too soon' } },
  { ico: '🔒', lbl: 'Verify identity first', verdict: { ok: true, text: 'good move · did the required check' } },
  { ico: '🎉', lbl: 'Resolve the billing dispute', verdict: { ok: true, text: 'result · task solved' } },
]

const prefersReduced = () =>
  typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false

export default function HeroScene() {
  const reduce = prefersReduced()
  const [phase, setPhase] = useState(reduce ? STEPS.length + 1 : 0)

  useEffect(() => {
    if (reduce) return
    const id = setInterval(() => setPhase(p => (p + 1) % (STEPS.length + 2)), 1350)
    return () => clearInterval(id)
  }, [reduce])

  return (
    <div className="hs-card" aria-hidden="true">
      <div className="hs-head">
        <span className="badge2"><span className="dot-live" />live agent</span>
        <span>Grade the path, not just the result</span>
      </div>

      {STEPS.map((s, i) => {
        const show = phase > i
        const vk = s.verdict ? (s.verdict.ok ? 'ok' : 'bad') : ''
        return (
          <div key={i} className={`hs-row ${show ? 'show' : ''} ${show ? vk : ''}`}>
            <span className="ico">{s.ico}</span>
            <span className="lbl">{s.lbl}</span>
            {s.verdict && (
              <span className={`hs-badge ${vk}`}>{s.verdict.ok ? '✓' : '✕'} {s.verdict.text}</span>
            )}
          </div>
        )
      })}

      <div className="hs-foot">
        <b>Score only the result</b> and the ✕ step goes unnoticed. <b>Grade the path</b> and the
        agent learns to skip it — while still being pushed to solve the task.
      </div>
    </div>
  )
}
