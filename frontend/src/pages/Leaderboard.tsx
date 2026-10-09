import { Card, State, pct } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { useState } from 'react'

export default function Leaderboard() {
  const [cipher, setCipher] = useState('')
  const [strategy, setStrategy] = useState('')
  const meta = useGet('/api/meta')
  const q = new URLSearchParams({ ...(cipher && { cipher }), ...(strategy && { strategy }) }).toString()
  const lb = useGet<any[]>(`/api/analytics/leaderboard${q ? `?${q}` : ''}`)
  return (
    <div>
      <h1>Leaderboard</h1>
      <p className="muted mono">Most robust first. Attack-success-rate (ASR) with 95% Wilson CI. Aggregates only — corpus payloads never appear here.</p>
      <Card><div className="row">
        <label style={{ margin: 0 }}>Cipher<select value={cipher} onChange={(e) => setCipher(e.target.value)}><option value="">all</option>{meta.data?.ciphers.map((c: any) => <option key={c.name}>{c.name}</option>)}</select></label>
        <label style={{ margin: 0 }}>Strategy<select value={strategy} onChange={(e) => setStrategy(e.target.value)}><option value="">all</option>{meta.data?.strategies.map((c: any) => <option key={c.name}>{c.name}</option>)}</select></label>
      </div></Card>
      <State loading={lb.loading} error={lb.error} empty={!lb.data?.length}>
        {lb.data?.map((r, i) => (
          <Card key={r.model} trim={i === 0}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div><h2 style={{ margin: 0 }}>#{i + 1} {r.model}</h2><span className="muted mono">{r.attempts} attempts</span></div>
              <div style={{ textAlign: 'right' }}><div className="kpi">{pct(r.asr)}</div><div className="mono muted">ASR [{pct(r.asr_ci_low)} – {pct(r.asr_ci_high)}] · evasion {pct(r.filter_evasion_rate)}</div></div>
            </div>
            <div className="bar" aria-hidden><i style={{ width: `${r.asr * 100}%` }} /></div>
          </Card>))}
      </State>
    </div>
  )
}
