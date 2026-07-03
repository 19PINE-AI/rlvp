import React, { useState } from 'react'
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
        ['Held-out tasks', '30 per domain, k=8, temp 0.7'],
        ['Policy', 'Qwen3-4B · 5 seeds'],
        ['Baseline', 'Outcome-only GRPO (rules in prompt)'],
        ['Ours', 'Reward outcome + penalize path'],
        ['Metric', 'clean rate = violation-free episodes'],
      ]} />
      <div className="cmp">
        <div className="col base">
          <div style={{ marginBottom: 10 }}><span className="tag base">BASELINE · outcome-only</span></div>
          {rows.map(([name, d]) => (
            <div key={name} style={{ marginBottom: 14 }}>
              <div className="metric-line" style={{ borderBottom: 'none', paddingBottom: 2 }}>
                <span>{name} · clean rate</span><span className="v r">{Math.round(d.baseline.clean * 100)}%</span>
              </div>
              <Bar pct={d.baseline.clean * 100} color="var(--gray)" />
              <div className="kicker">{d.baseline.violPer100} violations / 100 calls
                {Object.keys(d.baseline.perRule).length > 0 &&
                  ` · ${Object.entries(d.baseline.perRule).map(([r, n]) => `${r} (${n})`).join(', ')}`}</div>
            </div>
          ))}
        </div>
        <div className="col ours">
          <div style={{ marginBottom: 10 }}><span className="tag ours">OURS · penalize the path</span></div>
          {rows.map(([name, d]) => (
            <div key={name} style={{ marginBottom: 14 }}>
              <div className="metric-line" style={{ borderBottom: 'none', paddingBottom: 2 }}>
                <span>{name} · clean rate</span><span className="v g">{Math.round(d.ours.clean * 100)}%</span>
              </div>
              <Bar pct={d.ours.clean * 100} color="var(--green)" />
              <div className="kicker">{d.ours.violPer100} violations / 100 calls · task success held at {Math.round(d.ours.pass * 100)}%</div>
            </div>
          ))}
        </div>
      </div>
      <p className="kicker" style={{ marginTop: 16 }}>
        The penalty drives violation-free episodes from near-baseline to ~100% at no cost to task
        success — a deployability constraint outcome-only GRPO never reaches at any budget.
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
        ['Policy', 'Qwen3-4B · 5 seeds'],
        ['Signal', 'penalty on destructive actions'],
        ['Task success', 'at the floor (isolates the harm axis)'],
        ['Baseline', 'Outcome-only GRPO'],
        ['Metric', 'destructive actions per episode'],
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
        ['Scales', '4B and 30B (Qwen3-30B-A3B)'],
        ['Seeds', '5 per arm'],
        ['Ours', 'aligned potential (falling goal count)'],
        ['Metric', 'iterations to 0.9 success · divergences'],
        ['Optimizers', 'Muon (fast) / AdamW (stable)'],
      ]} />
      <div className="card" style={{ padding: 6, overflowX: 'auto' }}>
        <table className="tbl">
          <thead><tr>
            <th>scale</th><th>arm</th><th>iters → 0.9</th><th>AUC</th><th>final</th><th>diverged</th>
          </tr></thead>
          <tbody>
            {M.lean.map((r, i) => (
              <tr key={i} className={r.ours ? 'ours' : ''}>
                <td>{r.scale}</td>
                <td>{r.arm}{r.ours && ' ✓'}</td>
                <td className="num">{r.iters} ± {r.std}</td>
                <td className="num">{r.auc.toFixed(2)}</td>
                <td className="num">{r.final.toFixed(2)}</td>
                <td className="num">{r.diverged}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid three" style={{ marginTop: 18 }}>
        <div className="stat card"><div className="num r">{dd.outcome}%</div><div className="lbl">dead updates · outcome-only</div></div>
        <div className="stat card"><div className="num" style={{ color: 'var(--gray)' }}>{dd.dapo}%</div><div className="lbl">dead updates · DAPO <span className="kicker">(5.6× sampling)</span></div></div>
        <div className="stat card" style={{ background: '#f2fbf5', border: '1px solid #bfe6cd' }}><div className="num g">{dd.potential}%</div><div className="lbl">dead updates · aligned potential</div></div>
      </div>
      <p className="kicker" style={{ marginTop: 16 }}>
        The aligned potential is the only arm simultaneously fast and reliable at both scales:
        ~1.4–1.6× faster to mastery under a matched optimizer, and 0/5 divergences where outcome-only
        loses 1–3 seeds.
      </p>
    </div>
  )
}

const TABS = [
  ['constraints', 'Deployable constraints', Constraints],
  ['harm', 'TerminalBench harm', Harm],
  ['lean', 'Sample efficiency (Lean)', Lean],
]

export default function Results() {
  const [tab, setTab] = useState('constraints')
  const Active = TABS.find(t => t[0] === tab)[2]
  return (
    <section className="section" id="results" style={{ background: 'var(--bg-soft)' }}>
      <div className="wrap">
        <div className="eyebrow">Experiments</div>
        <h2>Main results — baseline vs. ours</h2>
        <p className="section-lead">
          Every number below is from the paper's real result dumps. Switch experiments to see the
          setting and how outcome-only training compares to the two-channel recipe.
        </p>
        <div className="tabs" style={{ marginTop: 24 }}>
          {TABS.map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <Active />
      </div>
    </section>
  )
}
