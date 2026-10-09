import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { AX, COLORS, ChartCard, DataDetails, TIP, colorOf, heat, pctTick, type Status } from '../components/charts'
import { cipherName } from '../lib/format'
import { Btn, Card, State, pct } from '../components/ui'
import { useGet } from '../hooks/useApi'

type F = Record<string, string>
const TABS = [['crypto', 'Crypto → bypass'], ['cost', 'Cost & scaling'], ['models', 'Models & defenses'], ['matrix', 'Strategy × cipher'], ['rigor', 'Rigor & standards']] as const

function qs(f: F, extra: F = {}) {
  const p = new URLSearchParams()
  Object.entries({ ...f, ...extra }).forEach(([k, v]) => v && p.set(k, v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

export default function Analytics() {
  const [sp, setSp] = useSearchParams()
  const tab = (TABS.find(([k]) => k === sp.get('tab'))?.[0] ?? 'crypto') as (typeof TABS)[number][0]
  const setTab = (k: string) => setSp({ tab: k })
  const [f, setF] = useState<F>({})
  const opts = useGet('/api/analytics/filters')
  const set = (k: string, v: string) => setF((s) => ({ ...s, [k]: v }))
  const o = opts.data
  const sel = (k: string, label: string, vals: string[]) => (
    <label style={{ margin: 0 }}>{label}<select value={f[k] ?? ''} onChange={(e) => set(k, e.target.value)}><option value="">all</option>{vals.map((v) => <option key={v}>{v}</option>)}</select></label>
  )
  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h1>Analytics</h1>
        <div className="row">
          <Btn variant="alt" onClick={() => window.open(`/api/analytics/report${qs(f, { format: 'html' })}`, '_blank')}>Robustness report (HTML)</Btn>
          <Btn variant="ghost" onClick={() => window.open(`/api/analytics/report${qs(f)}`, '_blank')}>JSON</Btn>
        </div>
      </div>
      <Card trim>
        <div className="row" role="search" aria-label="Analytics filters">
          {sel('model', 'Model', o?.models ?? [])}{sel('cipher', 'Cipher', o?.ciphers ?? [])}{sel('strategy', 'Strategy', o?.strategies ?? [])}{sel('defense', 'Defense', o?.defenses ?? [])}
          <label style={{ margin: 0 }}>Run<select value={f.run_id ?? ''} onChange={(e) => set('run_id', e.target.value)}><option value="">all</option>{(o?.runs ?? []).map((r: any) => <option key={r.run_id} value={r.run_id}>#{r.run_id} {r.name} ({r.n})</option>)}</select></label>
          <label style={{ margin: 0 }}>From<input type="date" value={f.date_from ?? ''} min={o?.date_min} max={o?.date_max} onChange={(e) => set('date_from', e.target.value)} /></label>
          <label style={{ margin: 0 }}>To<input type="date" value={f.date_to ?? ''} min={o?.date_min} max={o?.date_max} onChange={(e) => set('date_to', e.target.value)} /></label>
          <Btn variant="ghost" onClick={() => setF({})}>Reset</Btn>
        </div>
      </Card>
      <div className="row" role="tablist" style={{ marginBottom: 16 }}>
        {TABS.map(([k, label]) => <button key={k} role="tab" aria-selected={tab === k} className={`btn ${tab === k ? '' : 'ghost'}`} onClick={() => setTab(k)}>{label}</button>)}
      </div>
      {tab === 'crypto' && <CryptoTab f={f} />}
      {tab === 'cost' && <CostTab f={f} />}
      {tab === 'models' && <ModelsTab f={f} />}
      {tab === 'matrix' && <MatrixTab f={f} />}
      {tab === 'rigor' && <RigorTab f={f} />}
    </div>
  )
}

const st = (q: { loading: boolean; error: string | null; data: any }, empty = false): Status => ({ loading: q.loading && !q.data, error: q.error, empty: !!q.data && empty })
const P = (x: number | null | undefined) => (x == null ? null : `${(x * 100).toFixed(1)}%`)

function CryptoTab({ f }: { f: F }) {
  const [metric, setMetric] = useState('avalanche_score')
  const c = useGet(`/api/analytics/correlation${qs(f, { metric })}`, [metric, JSON.stringify(f)])
  const mt = useGet<any[]>(`/api/analytics/metric-table${qs(f)}`, [JSON.stringify(f)])
  const d = c.data
  const ciphers: string[] = d ? [...new Set<string>(d.by_cipher.map((x: any) => x.cipher))] : []
  return (
    <State loading={c.loading && !d} error={c.error}>
      {d && d.n === 0 && <Card title="No data for this metric / filter"><label style={{ margin: 0 }}>X axis<select value={metric} onChange={(e) => setMetric(e.target.value)}>{['avalanche_score', 'shannon_entropy', 'diffusion_score', 'confusion_chi2'].map((m) => <option key={m}>{m}</option>)}</select></label><p className="muted mono">Pick another metric or reset the filters.</p></Card>}
      {d && d.n > 0 && (<>
        <Card trim title="Headline: crypto property vs bypass rate">
          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ margin: 0 }}>X axis<select value={metric} onChange={(e) => setMetric(e.target.value)}>{['avalanche_score', 'shannon_entropy', 'diffusion_score', 'confusion_chi2'].map((m) => <option key={m}>{m}</option>)}</select></label>
            <div className="mono">n={d.n} · Spearman ρ = <b className="kpi" style={{ fontSize: 28 }}>{d.spearman ? d.spearman.rho.toFixed(3) : 'n/a'}</b>{d.spearman && <span className="muted"> (p={d.spearman.p.toExponential(1)})</span>}{d.logistic && <span className="muted"> · logit slope {d.logistic.coef.toExponential(2)}</span>}</div>
          </div>
          <div style={{ height: 340 }} role="img" aria-label={`Bypass rate by cipher against ${metric} with fitted logistic curve`}>
            <ResponsiveContainer><ScatterChart margin={{ top: 8, right: 16 }}>
              <CartesianGrid stroke="var(--border)" /><XAxis type="number" dataKey="x" name={metric} domain={['auto', 'auto']} {...AX} />
              <YAxis type="number" dataKey="y" name="bypass" domain={[0, 1]} tickFormatter={pctTick} {...AX} /><ZAxis range={[110, 110]} /><Tooltip {...TIP} formatter={(v: any) => (typeof v === 'number' ? v.toFixed(3) : v)} /><Legend />
              {ciphers.map((cp) => <Scatter key={cp} isAnimationActive={false} name={cp} data={d.by_cipher.filter((x: any) => x.cipher === cp).map((x: any) => ({ x: x.x, y: x.bypass_rate }))} fill={colorOf(ciphers, cp)} />)}
              {d.curve.length > 0 && <Scatter name="logistic fit" isAnimationActive={false} data={d.curve.map((p: any) => ({ x: p.x, y: p.p }))} line={{ stroke: '#fff', strokeDasharray: '6 4', strokeWidth: 2 }} shape={() => <g />} legendType="plainline" fill="#fff" />}
            </ScatterChart></ResponsiveContainer>
          </div>
          <DataDetails caption="Bypass rate per cipher against the selected property" table={{ headers: ['Cipher', metric, 'Bypass rate', 'Attempts'], rows: d.by_cipher.map((x: any) => [cipherName(x.cipher), +x.x.toFixed(4), P(x.bypass_rate), x.n]) }} />
        </Card>
        <ChartCard title="Bypass rate per cipher (95% confidence interval)" height={280} label="Bar chart of bypass rate per cipher with confidence intervals" table={{ headers: ['Cipher', 'Bypass rate', 'Low', 'High', 'Attempts'], rows: d.by_cipher.map((x: any) => [cipherName(x.cipher), P(x.bypass_rate), P(x.ci_low), P(x.ci_high), x.n]) }}>
          <ResponsiveContainer><BarChart data={d.by_cipher.map((x: any) => ({ ...x, err: [x.bypass_rate - x.ci_low, x.ci_high - x.bypass_rate] }))}>
            <CartesianGrid stroke="var(--border)" /><XAxis dataKey="cipher" {...AX} /><YAxis domain={[0, 1]} tickFormatter={pctTick} {...AX} /><Tooltip {...TIP} formatter={(v: any) => (typeof v === 'number' ? pct(v) : v)} />
            <Bar dataKey="bypass_rate" isAnimationActive={false}>{d.by_cipher.map((x: any) => <Cell key={x.cipher} fill={colorOf(ciphers, x.cipher)} />)}<ErrorBar dataKey="err" stroke="#fff" /></Bar>
          </BarChart></ResponsiveContainer>
        </ChartCard>
        <Card title="All four crypto metrics vs bypass">
          <State loading={mt.loading && !mt.data} error={mt.error}>
            <table><thead><tr><th>Metric</th><th>n</th><th>Spearman ρ</th><th>p</th><th>Logit slope</th></tr></thead>
              <tbody>{mt.data?.map((m) => <tr key={m.metric}><td>{m.metric}</td><td>{m.n}</td><td>{m.spearman ? m.spearman.rho.toFixed(3) : '—'}</td><td>{m.spearman ? m.spearman.p.toExponential(1) : '—'}</td><td>{m.logistic ? m.logistic.coef.toExponential(2) : '—'}</td></tr>)}</tbody></table>
          </State>
        </Card>
      </>)}
    </State>
  )
}

function CostTab({ f }: { f: F }) {
  const sc = useGet(`/api/analytics/scaling${qs(f)}`, [JSON.stringify(f)])
  const pa = useGet<any[]>(`/api/analytics/pareto${qs(f)}`, [JSON.stringify(f)])
  const pts: any[] = sc.data?.points ?? []
  const ciphers: string[] = useMemo(() => [...new Set<string>(pts.map((p: any) => p.cipher))], [pts])
  const line = (key: 'enc_vs_plaintext' | 'dec_vs_ciphertext', xk: string, cp: string) => {
    const fit = sc.data?.fits?.[cp]?.[key]
    const xs = pts.filter((p: any) => p.cipher === cp).map((p: any) => p[xk])
    if (!fit || fit.slope == null || !xs.length) return null
    const lo = Math.min(...xs), hi = Math.max(...xs)
    return [lo, hi].map((x) => ({ x, y: fit.slope * x + fit.intercept }))
  }
  const panel = (title: string, xk: string, yk: string, key: 'enc_vs_plaintext' | 'dec_vs_ciphertext', xlabel: string) => (
    <ChartCard title={title} label={title} height={300} status={st(sc, !pts.length)} note="Dashed lines are per-cipher best-fit lines. Fit quality (R²) is in the table below."
      table={{ headers: ['Cipher', 'Slope (ms per byte)', 'R²', 'Samples'], rows: ciphers.map((cp) => { const fit = sc.data?.fits?.[cp]?.[key]; return [cipherName(cp), fit?.slope == null ? null : +fit.slope.toExponential(2), fit?.r2 == null ? null : +fit.r2.toFixed(3), pts.filter((p: any) => p.cipher === cp).length] }) }}>
      <ResponsiveContainer><ScatterChart margin={{ right: 16, bottom: 16 }}><CartesianGrid stroke="var(--border)" />
        <XAxis type="number" dataKey="x" name={xlabel} label={{ value: xlabel, position: 'bottom', fill: 'var(--muted)', fontSize: 11 }} {...AX} /><YAxis type="number" dataKey="y" name="ms" unit="ms" {...AX} /><Tooltip {...TIP} /><Legend />
        {ciphers.map((cp) => <Scatter key={cp} isAnimationActive={false} name={cipherName(cp)} data={pts.filter((p: any) => p.cipher === cp).map((p: any) => ({ x: p[xk], y: p[yk] }))} fill={colorOf(ciphers, cp)} />)}
        {ciphers.map((cp) => { const l = line(key, xk, cp); return l ? <Scatter key={`${cp}-fit`} isAnimationActive={false} legendType="none" data={l} line={{ stroke: colorOf(ciphers, cp), strokeDasharray: '5 4' }} shape={() => <g />} fill="none" /> : null })}
      </ScatterChart></ResponsiveContainer>
    </ChartCard>
  )
  return (
    <>
      <ChartCard title="Cost of a bypass: success rate vs time per attempt" height={320} status={st(pa, !pa.data?.length)} label="Scatter of bypass rate versus mean cost in milliseconds per cipher; frontier ciphers are highlighted"
        note="Cost = encode time + decode time + model reply time. Yellow = no other cipher is both cheaper and more successful (the Pareto frontier)."
        table={{ headers: ['Cipher', 'Bypass rate', 'Mean cost (ms)', 'Cost (USD)', 'On frontier', 'Attempts'], rows: (pa.data ?? []).map((p) => [cipherName(p.cipher), P(p.bypass_rate), +p.mean_cost_ms.toFixed(2), +p.mean_cost_usd.toFixed(6), p.frontier ? 'yes' : '', p.n]) }}>
        <ResponsiveContainer><ScatterChart margin={{ right: 16 }}><CartesianGrid stroke="var(--border)" /><XAxis type="number" dataKey="mean_cost_ms" name="cost" unit="ms" {...AX} /><YAxis type="number" dataKey="bypass_rate" name="bypass" domain={[0, 1]} tickFormatter={pctTick} {...AX} /><ZAxis dataKey="n" range={[90, 360]} />
          <Tooltip {...TIP} formatter={(v: any, n: any) => (n === 'bypass' ? pct(v) : typeof v === 'number' ? v.toFixed(2) : v)} />
          <Scatter isAnimationActive={false} data={pa.data ?? []}>{(pa.data ?? []).map((p) => <Cell key={p.cipher} fill={p.frontier ? '#ffda14' : '#6b66b0'} stroke={p.frontier ? '#fff' : 'none'} />)}</Scatter></ScatterChart></ResponsiveContainer>
      </ChartCard>
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(420px,1fr))' }}>
        {panel('Prompt size vs encode time', 'plaintext_bytes', 'enc_ms', 'enc_vs_plaintext', 'prompt size (bytes)')}
        {panel('Encoded size vs decode time', 'ciphertext_bytes', 'dec_ms', 'dec_vs_ciphertext', 'encoded size (bytes)')}
      </div>
      <ChartCard title="How much longer the encoded prompt is" height={240} status={st(sc, !pts.length)} label="Bar chart of mean payload expansion per cipher" note="1.0 means the same length as the original prompt; about 1.33 is typical for Base64."
        table={{ headers: ['Cipher', 'Encoded length ÷ original length'], rows: Object.entries(sc.data?.fits ?? {}).map(([c, v]: any) => [cipherName(c), +v.expansion.toFixed(3)]) }}>
        <ResponsiveContainer><BarChart data={Object.entries(sc.data?.fits ?? {}).map(([cipher, v]: any) => ({ cipher: cipherName(cipher), expansion: v.expansion }))}><CartesianGrid stroke="var(--border)" /><XAxis dataKey="cipher" {...AX} /><YAxis {...AX} /><Tooltip {...TIP} /><Bar dataKey="expansion" fill="#b9a8ff" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartCard>
    </>
  )
}

function ModelsTab({ f }: { f: F }) {
  const ms = useGet<any[]>(`/api/analytics/model-size${qs(f)}`, [JSON.stringify(f)])
  const lb = useGet<any[]>(`/api/analytics/leaderboard${qs(f)}`, [JSON.stringify(f)])
  const ab = useGet<any[]>(`/api/analytics/defense-ablation${qs(f)}`, [JSON.stringify(f)])
  const sized = (ms.data ?? []).filter((m) => m.size_b != null)
  const abl = ab.data ?? []
  const scope = abl[0]?.scope
  return (
    <>
      <ChartCard title="Does a bigger model resist better? Bypass rate vs model size (billions of parameters)" height={300} status={st(ms, !ms.data?.length)} label="Chart of bypass rate against model parameter count"
        note={sized.length < (ms.data?.length ?? 0) ? 'Models whose size is not in their name (for example cloud APIs) appear in the table but not on the chart.' : 'Small edge models on the left, larger or cloud models on the right.'}
        table={{ headers: ['Model', 'Size (B)', 'Attempts', 'Bypass rate'], rows: (ms.data ?? []).map((m) => [m.model, m.size_b, m.n, P(m.rate)]) }}>
        <ResponsiveContainer><ScatterChart margin={{ right: 16 }}><CartesianGrid stroke="var(--border)" /><XAxis type="number" dataKey="size_b" scale="log" domain={['auto', 'auto']} name="params" unit="B" {...AX} /><YAxis type="number" dataKey="rate" domain={[0, 1]} tickFormatter={pctTick} {...AX} /><ZAxis dataKey="n" range={[80, 320]} /><Tooltip {...TIP} />
          <Scatter name="bypass" isAnimationActive={false} data={sized} fill="#ffda14" line={{ stroke: '#ffda14' }} /></ScatterChart></ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Models ranked by how often they were bypassed (most robust first)" height={Math.max(220, (lb.data?.length ?? 1) * 54)} status={st(lb, !lb.data?.length)} label="Horizontal bar chart of attack success rate per model"
        table={{ headers: ['Model', 'Attempts', 'Success rate', 'Low', 'High'], rows: (lb.data ?? []).map((r) => [r.model, r.attempts, P(r.asr), P(r.asr_ci_low), P(r.asr_ci_high)]) }}>
        <ResponsiveContainer><BarChart layout="vertical" data={(lb.data ?? []).map((r) => ({ ...r, err: [r.asr - r.asr_ci_low, r.asr_ci_high - r.asr] }))}><CartesianGrid stroke="var(--border)" /><XAxis type="number" domain={[0, 1]} tickFormatter={pctTick} {...AX} /><YAxis type="category" dataKey="model" width={150} {...AX} /><Tooltip {...TIP} formatter={(v: any) => (typeof v === 'number' ? pct(v) : v)} />
          <Bar dataKey="asr" fill="#4502ff" isAnimationActive={false}><ErrorBar dataKey="err" stroke="#fff" /></Bar></BarChart></ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Do the defenses help? Bypass rate with each defense switched on" height={280} status={st(ab, !abl.length)} label="Bar chart of bypass rate per defense compared to no defense"
        note={scope ? `Compared only on matching cells (same model, dataset, cipher and strategy) so the difference is fair.${scope.defense_filter_ignored ? ' Your defense filter is ignored here.' : ''}` : undefined}
        table={{ headers: ['Defense', 'Bypass rate', 'Same cells without it', 'Change (pp)', '95% range (pp)', 'Cells'], rows: abl.map((r) => [r.defense, P(r.rate), P(r.baseline_rate), r.delta_vs_none == null ? null : +(r.delta_vs_none * 100).toFixed(1), r.delta_ci_low == null ? null : `${(r.delta_ci_low * 100).toFixed(1)} to ${(r.delta_ci_high * 100).toFixed(1)}`, r.matched_cells]) }}>
        <ResponsiveContainer><BarChart data={abl}><CartesianGrid stroke="var(--border)" /><XAxis dataKey="defense" {...AX} /><YAxis domain={[0, 1]} tickFormatter={pctTick} {...AX} /><Tooltip {...TIP} formatter={(v: any) => (typeof v === 'number' ? pct(v) : v)} />
          <Bar dataKey="rate" isAnimationActive={false}>{abl.map((r, i) => <Cell key={r.defense} fill={r.defense === 'none' ? '#dc2626' : COLORS[(i + 1) % COLORS.length]} />)}</Bar></BarChart></ResponsiveContainer>
      </ChartCard>
      <Card title="Defense effect (matched cells only)">
        <State loading={ab.loading && !ab.data} error={ab.error} empty={abl.length <= 1}>
          <table><thead><tr><th>Defense</th><th>Bypass rate</th><th>Same cells without it</th><th>Change</th><th>95% range</th><th>Cells</th></tr></thead><tbody>{abl.filter((r) => r.defense !== 'none').map((r) => <tr key={r.defense}><td>{r.defense}</td><td>{pct(r.rate)}</td><td>{pct(r.baseline_rate)}</td>
            <td className={r.delta_vs_none < 0 ? 'ok' : r.delta_vs_none > 0 ? 'bad' : ''}>{r.delta_vs_none == null ? (r.note ?? '—') : `${r.delta_vs_none > 0 ? '+' : ''}${(r.delta_vs_none * 100).toFixed(1)} pp`}</td>
            <td>{r.delta_ci_low == null ? '—' : `${(r.delta_ci_low * 100).toFixed(1)} to ${(r.delta_ci_high * 100).toFixed(1)} pp`}</td><td>{r.matched_cells}</td></tr>)}</tbody></table>
          <p className="muted mono">A negative change means the defense reduced bypasses. Small samples give wide ranges, so treat them as a guide.</p>
        </State>
      </Card>
    </>
  )
}

function MatrixTab({ f }: { f: F }) {
  const h = useGet<any[]>(`/api/analytics/heatmap${qs(f)}`, [JSON.stringify(f)])
  const ciphers = [...new Set((h.data ?? []).map((x) => x.cipher))], strats = [...new Set((h.data ?? []).map((x) => x.strategy))]
  const cell = (c: string, s: string) => (h.data ?? []).find((x) => x.cipher === c && x.strategy === s)
  return (
    <State loading={h.loading && !h.data} error={h.error} empty={!h.data?.length}>
      <Card trim title="Bypass rate: cipher × strategy">
        <div className="tablewrap"><table aria-label="Heatmap of bypass rate by cipher and strategy"><thead><tr><th>cipher \ strategy</th>{strats.map((s) => <th key={s}>{s}</th>)}</tr></thead>
          <tbody>{ciphers.map((c) => <tr key={c}><td><b>{c}</b></td>{strats.map((s) => { const x = cell(c, s); return <td key={s} title={x ? `${x.k}/${x.n} · CI ${pct(x.ci_low)}–${pct(x.ci_high)}` : ''} style={{ background: x ? heat(x.rate) : 'transparent', color: '#fff', textAlign: 'center', fontFamily: 'var(--pixel)', fontSize: 22 }}>{x ? `${pct(x.rate)}` : '—'}<div style={{ fontSize: 11, fontFamily: 'var(--mono)' }}>{x ? `n=${x.n}` : ''}</div></td> })}</tr>)}</tbody></table></div>
        <p className="muted mono">Green = robust, red = frequently bypassed. Hover a cell for the confidence interval.</p>
      </Card>
    </State>
  )
}

function RigorTab({ f }: { f: F }) {
  const ja = useGet<any[]>(`/api/analytics/judge-agreement${qs(f)}`, [JSON.stringify(f)])
  const std = useGet<any[]>(`/api/analytics/standards${qs(f)}`, [JSON.stringify(f)])
  const tl = useGet<any[]>(`/api/analytics/timeline${qs(f)}`, [JSON.stringify(f)])
  const cellv = (x: any) => (x?.agreement == null ? '—' : `${pct(x.agreement)}${x.kappa != null ? ` · κ ${x.kappa.toFixed(2)}` : ''} (n=${x.n})`)
  return (
    <>
      <ChartCard title="Bypass rate for each run, in order" height={260} status={st(tl, !tl.data?.length)} label="Line chart of bypass rate per run over time"
        table={{ headers: ['Run', 'Name', 'Started', 'Attempts', 'Bypass rate'], rows: (tl.data ?? []).map((r) => [r.run_id, r.name, r.created_at, r.n, P(r.rate)]) }}>
        <ResponsiveContainer><LineChart data={tl.data ?? []}><CartesianGrid stroke="var(--border)" /><XAxis dataKey="run_id" {...AX} /><YAxis domain={[0, 1]} tickFormatter={pctTick} {...AX} /><Tooltip {...TIP} formatter={(v: any) => (typeof v === 'number' && v <= 1 ? pct(v) : v)} /><Line dataKey="rate" stroke="#ffda14" strokeWidth={3} dot={{ fill: '#ffda14' }} isAnimationActive={false} /></LineChart></ResponsiveContainer>
      </ChartCard>
      <Card title="Do the three checks agree? Canary match, judge model, Llama Guard">
        <State loading={ja.loading && !ja.data} error={ja.error} empty={!ja.data?.length}>
          <table><thead><tr><th>Strategy</th><th>n</th><th>Canary ↔ judge</th><th>Canary ↔ Guard</th><th>Judge ↔ Guard</th></tr></thead>
            <tbody>{ja.data?.map((r) => <tr key={r.strategy}><td>{r.strategy}</td><td>{r.n}</td><td>{cellv(r.regex_judge)}</td><td>{cellv(r.regex_guard)}</td><td>{cellv(r.judge_guard)}</td></tr>)}</tbody></table>
          <p className="muted mono">A dash means that check was not turned on for those runs, so no agreement is shown. κ near 1 = strong agreement, near 0 = no better than chance. Turn on a judge and Llama Guard in New Run to fill this in.</p>
        </State>
      </Card>
      <Card title="OWASP LLM Top 10 and MITRE ATLAS">
        <State loading={std.loading && !std.data} error={std.error} empty={!std.data?.length}>
          <table><thead><tr><th>OWASP</th><th>ATLAS</th><th>Attempts</th><th>Bypass rate (95% range)</th></tr></thead><tbody>{std.data?.map((r, i) => <tr key={i}><td>{r.owasp}</td><td>{r.atlas}</td><td>{r.attempts}</td><td>{pct(r.bypass_rate)} [{pct(r.ci_low)} – {pct(r.ci_high)}]</td></tr>)}</tbody></table>
        </State>
      </Card>
    </>
  )
}
