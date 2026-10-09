import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Btn, Card, Kpi, State, StatusPill, pct } from '../components/ui'
import { useGet } from '../hooks/useApi'

export default function Dashboard() {
  const o = useGet('/api/analytics/overview')
  const runs = useGet<any[]>('/api/runs')
  const d = o.data
  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h1>Dashboard</h1>
        <Link to="/new"><Btn>+ New attempt</Btn></Link>
      </div>
      <State loading={o.loading} error={o.error}>
        {d && (<>
          <div className="grid">
            <Kpi label="Bypass rate" value={d.bypass_rate * 100} fmt={(n) => `${n.toFixed(1)}%`} />
            <Kpi label="Attempts" value={d.total_attempts} />
            <Kpi label="Spend (USD)" value={d.spend_usd} fmt={(n) => `$${n.toFixed(4)}`} />
            <Kpi label="Active runs" value={d.active_runs} />
          </div>
          <Card title="Bypass rate by model">
            {d.per_model.length === 0 ? <span className="muted">No attempts yet. <Link to="/new">Start a run ›</Link></span> : (
              <div style={{ height: 240 }} role="img" aria-label="Bar chart of bypass rate per model">
                <ResponsiveContainer><BarChart data={d.per_model.map((m: any) => ({ ...m, rate: +(m.bypass_rate * 100).toFixed(1) }))}>
                  <CartesianGrid stroke="var(--border)" /><XAxis dataKey="model" stroke="var(--muted)" /><YAxis unit="%" stroke="var(--muted)" />
                  <Tooltip contentStyle={{ background: 'var(--surface)', border: '2px solid var(--secondary)' }} />
                  <Bar dataKey="rate" fill="#ffda14" /></BarChart></ResponsiveContainer>
              </div>)}
          </Card>
          <Card title="Recent runs">
            <div className="stack mono">{(runs.data ?? []).slice(0, 8).map((r) => (
              <div key={r.id} className="row"><StatusPill s={r.status} /><Link to={`/live?run=${r.id}`}>#{r.id} {r.name}</Link>
                <span className="muted">{r.dataset_name} · seed {r.seed} · spent ${r.spent_usd.toFixed(4)}</span></div>))}
              {!runs.data?.length && <span className="muted">—</span>}</div>
          </Card>
          <div className="muted mono">Filter evasion {pct(d.filter_evasion_rate)} · confirmed {d.confirmed}/{d.total_attempts}</div>
        </>)}
      </State>
    </div>
  )
}
