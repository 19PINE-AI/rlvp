import React from 'react'
import data from './data/paperData.json'
import HeroScene from './components/HeroScene.jsx'
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
          <a href="https://arxiv.org/abs/2607.07435" target="_blank" rel="noreferrer">Paper ↗</a>
          <a href="https://github.com/19PINE-AI/rlvp" target="_blank" rel="noreferrer">GitHub ↗</a>
        </nav>
      </div>
    </div>
  )
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="wrap hero-grid">
        <div>
          <div className="eyebrow">Teaching real-world AI agents to act well</div>
          <h1>Penalize the Path,<br />Reward the Outcome</h1>
          <div className="authors">{data.meta.authors.join('  ·  ')}</div>
          <p className="thesis">
            An AI agent that fixes code, places phone calls, or resolves support tickets can't just
            get the final result right — it has to behave correctly at <em>every step</em>. RLVP
            grades the <strong>path</strong> the agent takes, not only the outcome, so it learns
            faster and stays safe to deploy.
          </p>
          <div className="pill-row">
            <span className="pill green">Grade every step, not just the result</span>
            <span className="pill blue">Learns faster — even from failed tries</span>
            <span className="pill green">Safer to deploy — respects the rules</span>
          </div>
          <div className="pill-row">
            <a className="btn primary" href="#mechanism">See how it works</a>
            <a className="btn" href="#cases">Explore real evaluation cases</a>
            <a className="btn" href="https://arxiv.org/abs/2607.07435" target="_blank" rel="noreferrer">Read the paper (arXiv) ↗</a>
          </div>
        </div>
        <HeroScene />
      </div>
    </header>
  )
}

function MechanismIntro() {
  return (
    <section className="section" id="mechanism">
      <div className="wrap">
        <div className="eyebrow">The core idea</div>
        <h2>How the agent learns — even when every try fails</h2>
        <p className="section-lead">
          Today's method trains an agent from one number: did it succeed? When several attempts at a
          task <em>all</em> fail, they all score the same — zero — so nothing is learned from them.
          Grading each <em>step</em> breaks the tie: some failed attempts still did good things,
          others did bad things. Flip the switch to see the effect.
        </p>
        <p className="kicker" style={{ maxWidth: 660, marginTop: -4 }}>
          <em>For RL readers:</em> a group-relative advantage is a within-group reward variance, which
          is zero on an all-fail group; a per-action signal restores it.
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
          <div className="eyebrow">Use #1 · Behaving safely</div>
          <h2>Penalize the path</h2>
          <p className="section-lead">
            A simple rule-checker watches every action. Break a rule — call too soon, act without
            authorization — and that step gets a{' '}
            <strong style={{ color: 'var(--red)' }}>penalty</strong>; do the required thing first and
            it gets a <strong style={{ color: 'var(--green)' }}>reward</strong>. This teaches the
            do's and don'ts that a plain success-or-fail score can't see.
          </p>
          <div className="card" style={{ padding: 26, marginTop: 24 }}>
            <PenaltyFlow />
          </div>
        </div>
      </section>

      <section className="section" id="potential">
        <div className="wrap">
          <div className="eyebrow">Use #2 · Learning faster</div>
          <h2>Reward real progress</h2>
          <p className="section-lead">
            The same idea has a second use. Instead of only flagging bad steps, give a small{' '}
            <strong style={{ color: 'var(--green)' }}>reward each time the agent gets closer</strong>{' '}
            to the goal. On a real math-proof task, every verified step toward the finish earns a
            little reward — so the agent gets useful feedback long before it ever completes a proof,
            and learns from far fewer tries.
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
                <a href="https://arxiv.org/abs/2607.07435" target="_blank" rel="noreferrer">Paper (arXiv) ↗</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="https://github.com/19PINE-AI/rlvp" target="_blank" rel="noreferrer">Code ↗</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="https://01.me/research/rlvp" target="_blank" rel="noreferrer">Website ↗</a>
              </p>
            </div>
            <div>
              <div style={{ fontWeight: 700, color: '#fff', marginBottom: 8 }}>Cite</div>
              <div className="cite">{`@article{li2026rlvp,
  title   = {RLVP: Penalize the Path, Reward the Outcome},
  author  = {Li, Bojie and Shi, Noah},
  journal = {arXiv preprint arXiv:2607.07435},
  year    = {2026}
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
