import { type ColumnDef } from '@tanstack/react-table'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Btn, Card, DataTable, Drawer, KV, Pill, State } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { post } from '../lib/api'
import { cell, header } from '../lib/format'
import { connectLive } from '../lib/ws'

const TABS = ['runs', 'attempts', 'payloads', 'responses', 'judgments', 'defenses', 'disclosures']
const PAGE = 50

export default function DBExplorer() {
  const [tab, setTab] = useState('attempts')
  const [q, setQ] = useState('')
  const [qDeb, setQDeb] = useState('')
  const [eligible, setEligible] = useState(false)
  const [page, setPage] = useState(0)
  const [sort, setSort] = useState<{ id: string; desc: boolean }>({ id: 'id', desc: true })
  const [sel, setSel] = useState<any | null>(null)
  const [tick, setTick] = useState(0)
  const [live, setLive] = useState(0)
  const [redact, setRedact] = useState(true)
  const [note, setNote] = useState<string | null>(null)
  const [sql, setSql] = useState('select cipher_type, count(*) n, avg(shannon_entropy) h from payloads group by 1')
  const [sqlOut, setSqlOut] = useState<any>(null)
  const [sqlErr, setSqlErr] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => { window.clearTimeout(timer.current); timer.current = window.setTimeout(() => { setQDeb(q); setPage(0) }, 250) }, [q])
  const params = `limit=${PAGE}&offset=${page * PAGE}&sort=${sort.id}&desc=${sort.desc}&q=${encodeURIComponent(qDeb)}${eligible && tab === 'judgments' ? '&eligible=true' : ''}`
  const d = useGet<{ total: number; rows: any[] }>(`/api/explorer/${tab}?${params}`, [tick])
  useEffect(() => connectLive(null, (e) => { if (e.type === 'attempt') { setLive((n) => n + 1); setTick((t) => (t + 1) % 1e9) } }), []) // live row-count ticker
  const pages = Math.max(1, Math.ceil((d.data?.total ?? 0) / PAGE))
  useEffect(() => { if (d.data && page >= pages) setPage(pages - 1) }, [d.data, page, pages])

  const cols = useMemo<ColumnDef<any>[]>(() => Object.keys(d.data?.rows[0] ?? {}).map((k) => ({
    accessorKey: k, enableSorting: false,
    header: () => <span onClick={() => { setSort((s) => ({ id: k, desc: s.id === k ? !s.desc : true })); setPage(0) }} aria-sort={sort.id === k ? (sort.desc ? 'descending' : 'ascending') : 'none'} title={`Sort by ${k}`}>{header(k)}{sort.id === k ? (sort.desc ? ' ▼' : ' ▲') : ''}</span>,
    cell: (c: any) => cell(k, c.getValue()) })), [d.data, sort])

  const exportUrl = (format: 'csv' | 'json') => `/api/explorer/${tab}/export?format=${format}&redact=${redact}&q=${encodeURIComponent(qDeb)}${eligible && tab === 'judgments' ? '&eligible=true' : ''}`
  const promote = async (attemptId: number) => {
    try { const r = await post('/api/corpus/promote', { source_attempt_id: attemptId }); setNote(`Attempt #${attemptId} → corpus (${r.status}, ${r.target_model})`) } catch (e: any) { setNote(`Promote failed: ${e.message}`) }
  }
  const runSql = async () => { setSqlErr(null); try { setSqlOut(await post('/api/explorer-sql', { sql })) } catch (e: any) { setSqlErr(e.message) } }
  const canPromote = (tab === 'judgments' || tab === 'attempts') && sel && (tab === 'attempts' || (sel.filter_evaded && sel.semantic_complied))
  const promoteId = tab === 'judgments' ? sel?.attempt_id : sel?.id

  return (
    <div>
      <h1>Database explorer</h1>
      <div className="row" role="tablist" style={{ marginBottom: 12 }}>{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={`btn ${tab === t ? '' : 'ghost'}`} onClick={() => { setTab(t); setSel(null); setPage(0); setSort({ id: 'id', desc: true }) }}>{t}</button>)}</div>
      <Card>
        <div className="row">
          <input aria-label="Search rows" placeholder="full-text search…" value={q} onChange={(e) => setQ(e.target.value)} style={{ maxWidth: 320 }} />
          {tab === 'judgments' && <label className="row" style={{ margin: 0, textTransform: 'none' }}><input type="checkbox" checked={eligible} onChange={(e) => { setEligible(e.target.checked); setPage(0) }} /> Promote-eligible only</label>}
          <span className="mono" aria-live="polite">{d.data?.total ?? 0} rows</span>
          {live > 0 && <Pill tone="blue">+{live} live</Pill>}
          <label className="row" style={{ margin: 0, textTransform: 'none' }}><input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} /> Redact payloads/responses in export</label>
          <a className="btn ghost" href={exportUrl('csv')} download>CSV</a><a className="btn ghost" href={exportUrl('json')} download>JSON</a>
        </div>
        {note && <p className="mono" role="status">{note}</p>}
      </Card>
      <State loading={d.loading && !d.data} error={d.error} empty={!d.data?.rows.length}>
        <DataTable rows={d.data?.rows ?? []} columns={cols} onRow={setSel} />
      </State>
      {(d.data?.total ?? 0) > 0 && (
        <div className="row" style={{ margin: '12px 0' }}>
          <Btn variant="ghost" disabled={page === 0} onClick={() => setPage(0)}>« First</Btn>
          <Btn variant="ghost" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>‹ Prev</Btn>
          <span className="mono">page {page + 1} / {pages}</span>
          <Btn variant="ghost" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>Next ›</Btn>
          <Btn variant="ghost" disabled={page + 1 >= pages} onClick={() => setPage(pages - 1)}>Last »</Btn>
        </div>)}
      {sel && <Drawer title={`${tab} #${sel.id ?? ''}`} onClose={() => setSel(null)}>
        {canPromote && <div className="row" style={{ marginBottom: 12 }}><Btn variant="alt" onClick={() => promote(promoteId)}>Promote to corpus ›</Btn><span className="muted mono">needs filter_evaded ∧ semantic_complied</span></div>}
        <KV data={sel} /></Drawer>}
      <Card title="Read-only SQL (SELECT only)">
        <textarea rows={3} aria-label="SQL" value={sql} onChange={(e) => setSql(e.target.value)} />
        <div className="row" style={{ margin: '8px 0' }}><Btn onClick={runSql}>Run</Btn>{sqlErr && <span className="bad" role="alert">{sqlErr}</span>}</div>
        {sqlOut && <div className="tablewrap"><table><thead><tr>{sqlOut.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead><tbody>{sqlOut.rows.map((r: any[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>}
      </Card>
    </div>
  )
}
