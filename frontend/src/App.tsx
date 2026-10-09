import { useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { Btn, Card } from './components/ui'
import Analytics from './pages/Analytics'
import Compute from './pages/Compute'
import Corpus from './pages/Corpus'
import Dashboard from './pages/Dashboard'
import DBExplorer from './pages/DBExplorer'
import Leaderboard from './pages/Leaderboard'
import LiveConsole from './pages/LiveConsole'
import NewExperiment from './pages/NewExperiment'
import Settings from './pages/Settings'

const NAV = [['/', 'Dashboard'], ['/new', 'New Run'], ['/live', 'Live Console'], ['/analytics', 'Analytics'], ['/db', 'DB Explorer'],
  ['/corpus', 'Corpus'], ['/leaderboard', 'Leaderboard'], ['/compute', 'GPU Nodes'], ['/settings', 'Settings']] as const

const CONSENT_KEY = 'rt.consent.v1'
const readConsent = () => { try { return localStorage.getItem(CONSENT_KEY) === '1' } catch { return false } }

function Consent({ onOk }: { onOk: () => void }) {
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Responsible use">
      <Card trim title="Insert coin — responsible use">
        <div className="stack mono">
          <p>This is a security-evaluation tool. Use it only on models and apps you own or are explicitly authorized to test, in an isolated environment.</p>
          <p>Default objectives are benign canaries. Benchmark datasets are supplied by you. The attack corpus stays on this machine and leaves only inside a scoped, provenance-stamped vendor disclosure.</p>
          <p>Check each provider’s usage policy or red-team programme before pointing a BYOK key at it.</p>
          <Btn onClick={() => { try { localStorage.setItem(CONSENT_KEY, '1') } catch { /* private mode */ } onOk() }}>I understand — start</Btn>
        </div>
      </Card>
    </div>
  )
}

export default function App() {
  const [ok, setOk] = useState(readConsent())
  return (
    <BrowserRouter>
      {!ok && <Consent onOk={() => setOk(true)} />}
      <div className="app">
        <nav className="side" aria-label="Primary">
          <div className="brand">RED-TEAM<br />ARCADE</div>
          <div className="nav">{NAV.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}</div>
          <div className="muted mono" style={{ marginTop: 'auto', fontSize: 12 }}>OSS guardrail-robustness benchmark · built on PyRIT</div>
        </nav>
        <main className="main" id="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new" element={<NewExperiment />} />
            <Route path="/live" element={<LiveConsole />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/db" element={<DBExplorer />} />
            <Route path="/corpus" element={<Corpus />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/compute" element={<Compute />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
