import React from 'react'
import data from '../data/paperData.json'

const M = data.metrics

function Bar({ pct, color }) {
  return (
    <div className="barcmp"><div className="track2">
      <div className="fill" style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: color }} />
    </div></div>
  )
}

function Setting({ items }) {
  return (
    <div className="card" style={{ padding: '16px 20px', marginBottom: 22 }}>
      <div className="eyebrow" style={{ color: 'var(--ink-soft)' }}>Evaluation setting</div>
      <div className="grid three" style={{ gap: 12 }}>
        {items.map(([k, v]) => (
          <div key={k}>
            <div className="kicker" style={{ fontSize: 13 }}>{k}</div>
            <div style={{ fontWeight: 700 }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function Constraints() {
  const fo = M.sysadmin.fileops, cs = M.sysadmin.csops
  const rows = [['File operations', fo], ['Customer-service ops', cs]]
  return (
    <div>
      <Setting items={[
        ['Domain', 'Sysadmin + customer-service tasks'],
        ['Held-out tasks', '30 per domain, 8 tries each'],
        ['Model', 'Qwen3-4B · 5 training seeds'],
        ['Baseline', 'Result-only training (rules in the prompt)'],
        ['Ours', 'Reward the result + grade the path'],
        ['Metric', 'violation-free rate = episodes that broke no rule'],
      ]} />
      <div className="cmp">
        <div className="col base">
          <div style={{ marginBottom: 10 }}><span className="tag base">RESULT-ONLY · baseline</span></div>
          {rows.map(([name, d]) => (
            <div key={name} style={{ marginBottom: 14 }}>
              <div className="metric-line" style={{ borderBottom: 'none', paddingBottom: 2 }}>
                <span>{name} · violation-free</span><span className="v r">{Math.round(d.baseline.clean * 100)}%</span>
              </div>
              <Bar pct={d.baseline.clean * 100} color="var(--gray)" />
              <div className="kicker">{d.baseline.violPer100} violations / 100 calls
                {Object.keys(d.baseline.perRule).length > 0 &&
                  ` · ${Object.entries(d.baseline.perRule).map(([r, n]) => `${r} (${n})`).join(', ')}`}</div>
            </div>
          ))}
        </div>
        <div className="col ours">
          <div style={{ marginBottom: 10 }}><span className="tag ours">GRADE THE PATH · ours</span></div>
          {rows.map(([name, d]) => (
            <div key={name} style={{ marginBottom: 14 }}>
              <div className="metric-line" style={{ borderBottom: 'none', paddingBottom: 2 }}>
                <span>{name} · violation-free</span><span className="v g">{Math.round(d.ours.clean * 100)}%</span>
              </div>
              <Bar pct={d.ours.clean * 100} color="var(--green)" />
              <div className="kicker">{d.ours.violPer100} violations / 100 calls · task success held at {Math.round(d.ours.pass * 100)}%</div>
            </div>
          ))}
        </div>
      </div>
      <p className="kicker" style={{ marginTop: 16 }}>
        Grading the path drives violation-free episodes to ~100% with no drop in task success —
        something result-only training never reaches, no matter how long it trains.
      </p>
    </div>
  )
}

function Harm() {
  const t = M.terminalbench
  return (
    <div>
      <Setting items={[
        ['Benchmark', 'TerminalBench (real shell containers)'],
        ['Model', 'Qwen3-4B · 5 training seeds'],
        ['Signal', 'penalty on destructive commands'],
        ['Task success', 'at the floor (isolates the harm question)'],
        ['Baseline', 'Result-only training'],
        ['Metric', 'destructive commands per episode'],
      ]} />
      <div className="grid three">
        <div className="stat card">
          <div className="num r">{t.baseline.viol}</div>
          <div className="lbl">baseline harmful actions / ep <span className="kicker">±{t.baseline.violStd}</span></div>
        </div>
        <div className="stat card" style={{ background: '#f2fbf5', border: '1px solid #bfe6cd' }}>
          <div className="num g">{t.ours.viol}</div>
          <div className="lbl">ours harmful actions / ep <span className="kicker">±{t.ours.violStd}</span></div>
        </div>
        <div className="stat card">
          <div className="num">≈6×</div>
          <div className="lbl">fewer harmful actions at equal success</div>
        </div>
      </div>
      <div className="cmp" style={{ marginTop: 18 }}>
        <div className="col base">
          <div className="metric-line"><span>Harmful actions / ep</span><span className="v r">{t.baseline.viol}</span></div>
          <Bar pct={t.baseline.viol / t.baseline.viol * 100} color="var(--red)" />
          <div className="metric-line"><span>Productive commands / ep</span><span className="v">≈{t.baseline.productive}</span></div>
          <div className="metric-line"><span>Task success</span><span className="v">{Math.round(t.baseline.success * 100)}%</span></div>
        </div>
        <div className="col ours">
          <div className="metric-line"><span>Harmful actions / ep</span><span className="v g">{t.ours.viol}</span></div>
          <Bar pct={t.ours.viol / t.baseline.viol * 100} color="var(--green)" />
          <div className="metric-line"><span>Productive commands / ep</span><span className="v g">≈{t.ours.productive}</span></div>
          <div className="metric-line"><span>Task success</span><span className="v">{Math.round(t.ours.success * 100)}%</span></div>
        </div>
      </div>
      <p className="kicker" style={{ marginTop: 16 }}>
        Not passivity: the penalized policy issues <em>more</em> productive commands, just cleaner ones.
        The per-seed harm distributions do not overlap.
      </p>
    </div>
  )
}

function Lean() {
  const dd = M.deadUpdates
  return (
    <div>
      <Setting items={[
        ['Task', 'miniF2F algebra theorem proving'],
        ['Model sizes', '4B and 30B'],
        ['Seeds', '5 training runs per method'],
        ['Ours', 'progress reward toward the proof'],
        ['Metric', 'training rounds to master the task · failed runs'],
        ['Optimizers', 'Muon (fast) / AdamW (stable)'],
      ]} />
      <div className="card" style={{ padding: 6, overflowX: 'auto' }}>
        <table className="tbl">
          <thead><tr>
            <th>size</th><th>method</th><th>rounds to master</th><th>learning curve (AUC)</th><th>final success</th><th>failed runs</th>
          </tr></thead>
          <tbody>
            {M.lean.map((r, i) => (
              <tr key={i} className={r.ours ? 'ours' : ''}>
                <td>{r.scale}</td>
                <td>{r.arm.replace('aligned potential', 'progress reward').replace('outcome-only', 'result-only')}{r.ours && ' ✓'}</td>
                <td className="num">{r.iters} ± {r.std}</td>
                <td className="num">{r.auc.toFixed(2)}</td>
                <td className="num">{r.final.toFixed(2)}</td>
                <td className="num">{r.diverged}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 20, fontWeight: 700 }}>Wasted training rounds
        <span className="kicker" style={{ fontWeight: 500 }}> — rounds where every attempt failed, so nothing was learned</span>
      </div>
      <div className="grid three" style={{ marginTop: 10 }}>
        <div className="stat card"><div className="num r">{dd.outcome}%</div><div className="lbl">result-only training</div></div>
        <div className="stat card"><div className="num" style={{ color: 'var(--gray)' }}>{dd.dapo}%</div><div className="lbl">DAPO <span className="kicker">(at 5.6× the sampling cost)</span></div></div>
        <div className="stat card" style={{ background: '#f2fbf5', border: '1px solid #bfe6cd' }}><div className="num g">{dd.potential}%</div><div className="lbl">progress reward (ours)</div></div>
      </div>
      <p className="kicker" style={{ marginTop: 16 }}>
        The progress reward is the only method that is both fast and reliable at both sizes:
        ~1.4–1.6× faster to master the task, and 0 of 5 runs failed — where result-only training
        loses 1–3 runs to instability.
      </p>
    </div>
  )
}

const SUBSECTIONS = [
  ['results-constraints', 'Deployable constraints',
    'Sysadmin + customer-service tasks — the penalty drives violation-free episodes to ~100% at no cost to task success.',
    Constraints],
  ['results-harm', 'TerminalBench harm',
    'A real shell benchmark — the harm penalty cuts destructive actions roughly sixfold at equal task success.',
    Harm],
  ['results-lean', 'Sample efficiency (Lean)',
    'miniF2F theorem proving — the progress reward masters the task faster and never destabilizes.',
    Lean],
]

function SubHead({ i, title, blurb }) {
  return (
    <div style={{ display: 'flex', gap: 14, alignItems: 'baseline', marginBottom: 16 }}>
      <span style={{
        flex: 'none', width: 34, height: 34, borderRadius: 9, background: 'var(--ink)', color: '#fff',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 15,
      }}>{i}</span>
      <div>
        <h3 style={{ margin: 0, fontSize: 24 }}>{title}</h3>
        <p style={{ margin: '4px 0 0', color: 'var(--ink-soft)' }}>{blurb}</p>
      </div>
    </div>
  )
}

export default function Results() {
  return (
    <section className="section" id="results" style={{ background: 'var(--bg-soft)' }}>
      <div className="wrap">
        <div className="eyebrow">Experiments</div>
        <h2>Main results — baseline vs. ours</h2>
        <p className="section-lead">
          Three experiments, all shown below. Every number is from the paper's real result dumps —
          each one gives the setup and how the usual result-only training compares to grading the
          path.
        </p>
        {SUBSECTIONS.map(([id, title, blurb, Body], idx) => (
          <div key={id} id={id} style={{ marginTop: idx === 0 ? 40 : 56, scrollMarginTop: 72 }}>
            <SubHead i={idx + 1} title={title} blurb={blurb} />
            <Body />
          </div>
        ))}
      </div>
    </section>
  )
}
