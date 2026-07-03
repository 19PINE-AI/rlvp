import React from 'react'
import data from './data/paperData.json'
import GroupRollout from './components/GroupRollout.jsx'
import PenaltyFlow from './components/PenaltyFlow.jsx'
import PotentialFlow from './components/PotentialFlow.jsx'
import Results from './components/Results.jsx'
import CaseExplorer from './components/CaseExplorer.jsx'

function Nav() {
  const links = [
    ['mechanism', 'Mechanism'],
    ['penalty', 'Penalize the path'],
    ['potential', 'Reward progress'],
    ['results', 'Results'],
    ['cases', 'Evaluation cases'],
  ]
  return (
    <div className="nav">
      <div className="wrap nav-inner">
        <a href="#top" className="brand">RL<span>V</span>P</a>
        <nav className="links">
          {links.map(([id, label]) => <a key={id} href={`#${id}`}>{label}</a>)}
          <a href="https://github.com/19PINE-AI/rlvp" target="_blank" rel="noreferrer">GitHub ↗</a>
        </nav>
      </div>
    </div>
  )
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="wrap">
        <div className="eyebrow">Reinforcement Learning from Verifiable Penalties</div>
        <h1>Penalize the Path,<br />Reward the Outcome</h1>
        <div className="authors">{data.meta.authors.join('  ·  ')}</div>
        <p className="thesis">{data.meta.tagline}</p>
        <div className="pill-row">
          <span className="pill green">−λ&nbsp; penalty on a verified bad move</span>
          <span className="pill green">+β&nbsp; credit for verified progress</span>
          <span className="pill blue">one channel, two uses</span>
        </div>
        <div className="pill-row">
          <a className="btn primary" href="#mechanism">See how it works</a>
          <a className="btn" href="#cases">Explore real evaluation cases</a>
        </div>
      </div>
    </header>
  )
}

function MechanismIntro() {
  return (
    <section className="section" id="mechanism">
      <div className="wrap">
        <div className="eyebrow">The core idea</div>
        <h2>Why a per-action path signal creates gradient the outcome cannot</h2>
        <p className="section-lead">
          Group-relative RL (GRPO) turns a group of rollouts into gradient only through their
          reward <em>variance</em>. When every rollout in a group fails, that variance is zero and the
          update is dead. A verifiable signal attached to each <em>action</em> restores it. Toggle the
          channel below to see the effect.
        </p>
        <div className="card" style={{ padding: 26, marginTop: 26 }}>
          <GroupRollout />
        </div>
      </div>
    </section>
  )
}

function App() {
  return (
    <>
      <Nav />
      <Hero />
      <MechanismIntro />

      <section className="section" id="penalty">
        <div className="wrap">
          <div className="eyebrow">Use #1 · Deployability</div>
          <h2>Penalize the path</h2>
          <p className="section-lead">
            A deterministic rule engine checks every action. A <strong>violation</strong> attaches a
            penalty <span style={{ color: 'var(--red)', fontWeight: 700 }}>−λ</span>; performing the
            required precondition attaches a discharge <span style={{ color: 'var(--green)', fontWeight: 700 }}>+β</span>.
            The penalty teaches outcome-neutral constraints the reward is blind to.
          </p>
          <div className="card" style={{ padding: 26, marginTop: 24 }}>
            <PenaltyFlow />
          </div>
        </div>
      </section>

      <section className="section" id="potential">
        <div className="wrap">
          <div className="eyebrow">Use #2 · Sample efficiency</div>
          <h2>Reward verified progress</h2>
          <p className="section-lead">
            The <em>same</em> +β credit, paid for verifiable progress instead of compliance, becomes a
            dense <strong>potential</strong>. On a real miniF2F proof the kernel verifies each tactic and
            the remaining obligations fall — every drop pays +β, densifying a reward the outcome leaves
            silent until the very end.
          </p>
          <div className="card" style={{ padding: 26, marginTop: 24 }}>
            <PotentialFlow />
          </div>
        </div>
      </section>

      <Results />
      <CaseExplorer />

      <footer className="foot" id="paper">
        <div className="wrap">
          <div className="grid two" style={{ alignItems: 'start' }}>
            <div>
              <h3 style={{ color: '#fff' }}>RLVP</h3>
              <p>Penalize the Path, Reward the Outcome.</p>
              <p style={{ fontSize: 14 }}>{data.meta.authors.join(' · ')}</p>
              <p style={{ marginTop: 16 }}>
                <a href="https://github.com/19PINE-AI/rlvp" target="_blank" rel="noreferrer">Code ↗</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="https://01.me/research/rlvp" target="_blank" rel="noreferrer">Website ↗</a>
              </p>
            </div>
            <div>
              <div style={{ fontWeight: 700, color: '#fff', marginBottom: 8 }}>Cite</div>
              <div className="cite">{`@article{rlvp2026,
  title  = {RLVP: Penalize the Path, Reward the Outcome},
  author = {Li, Bojie and Shi, Noah},
  year   = {2026}
}`}</div>
              <p style={{ fontSize: 12.5, marginTop: 14 }}>
                All numbers and evaluation transcripts on this page are extracted from the
                paper's real results dumps.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </>
  )
}

export default App
