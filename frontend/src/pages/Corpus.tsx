import { type ColumnDef } from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import { Btn, Card, DataTable, Drawer, Field, KV, Pill, State, StatusPill, download } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { cell } from '../lib/format'
import { del, post } from '../lib/api'

export default function Corpus() {
  const models = useGet<any[]>('/api/corpus/models')
  const [model, setModel] = useState<string | null>(null)
  const [status, setStatus] = useState('')
  const items = useGet<any[]>(model ? `/api/corpus?target_model=${encodeURIComponent(model)}${status ? `&status=${status}` : ''}` : null, [status])
  const [sel, setSel] = useState<string[]>([])
  const [open, setOpen] = useState<any | null>(null)
  const [wiz, setWiz] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const refresh = () => { models.reload(); items.reload() }
  const cols = useMemo<ColumnDef<any>[]>(() => [
    { accessorKey: 'status', header: 'Status', cell: (c) => <StatusPill s={String(c.getValue())} /> },
    { accessorKey: 'strategy_name', header: 'Strategy' }, { accessorKey: 'cipher_type', header: 'Cipher', cell: (c) => cell('cipher_type', c.getValue()) }, { accessorKey: 'objective_id', header: 'Objective', cell: (c) => cell('objective_id', c.getValue()) },
    { accessorKey: 'last_verified_at', header: 'Last checked', cell: (c) => cell('last_verified_at', c.getValue()) },
    { accessorKey: 'tags', header: 'Tags' },
    { id: 'act', header: '', enableSorting: false, cell: (c) => <Btn variant="ghost" onClick={async (e) => { e.stopPropagation(); setBusy(c.row.original.id); await post(`/api/corpus/${c.row.original.id}/verify`); setBusy(null); refresh() }}>{busy === c.row.original.id ? '…' : 'Re-verify'}</Btn> },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [busy, model, status])

  if (!model) return (
    <div>
      <h1>Attack corpus</h1>
      <p className="muted mono">Local-only, partitioned per target model. A jailbreak on one model is never assumed to work on another.</p>
      <State loading={models.loading} error={models.error} empty={!models.data?.length}>
        <div className="grid">{models.data?.map((m) => (
          <button key={m.target_model} onClick={() => setModel(m.target_model)} className="card trim" style={{ textAlign: 'left', cursor: 'pointer', color: 'inherit', font: 'inherit' }} aria-label={`Open ${m.target_model}`}>
            <h2>{m.target_model}</h2><div className="mono muted">{m.target_provider}</div>
            <div className="row" style={{ margin: '12px 0' }}><Pill tone="red">{m.working} working</Pill><Pill tone="green">{m.patched} patched</Pill><Pill>{m.unverified} unverified</Pill></div>
            <div className="mono" style={{ fontSize: 12 }}>strategies: {m.strategies_that_work}<br />ciphers: {m.ciphers_that_work}<br />verified: {String(m.last_verified ?? 'never').slice(0, 16)}</div>
          </button>))}</div>
      </State>
    </div>)

  return (
    <div>
      <div className="row"><Btn variant="ghost" onClick={() => { setModel(null); setSel([]); models.reload() }}>‹ All models</Btn><h1 style={{ margin: 0 }}>{model}</h1></div>
      <Card><div className="row">
        <label style={{ margin: 0 }}>Status<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="">all</option><option>working</option><option>patched</option><option>unverified</option></select></label>
        <Btn variant="alt" onClick={async () => { setBusy('all'); await post(`/api/corpus/verify-all?target_model=${encodeURIComponent(model)}`); setBusy(null); refresh() }}>{busy === 'all' ? 'Sweeping…' : 'Batch re-verify model'}</Btn>
        <Btn disabled={!sel.length} onClick={() => setWiz(true)}>Build vendor report › ({sel.length})</Btn>
      </div></Card>
      <State loading={items.loading && !items.data} error={items.error} empty={!items.data?.length}>
        <DataTable rows={items.data ?? []} columns={cols} onRow={setOpen} selectable idKey="id" selected={sel} onSelect={setSel} />
      </State>
      {open && <Drawer title={`${open.strategy_name} × ${open.cipher_type}`} onClose={() => setOpen(null)}>
        <div className="stack"><div className="row"><StatusPill s={open.status} /><Btn variant="danger" onClick={async () => { await del(`/api/corpus/${open.id}`); setOpen(null); refresh() }}>Delete</Btn></div>
          <KV data={{ plaintext_prompt: open.plaintext_prompt, ciphertext_payload: open.ciphertext_payload, decode_instruction: open.decode_instruction, metrics: open.metrics_json, judge_rationale: open.judge_rationale, repro: { seed: open.seed, config: open.repro_config_json }, tags: open.tags, disclosure_id: open.disclosure_id }} /></div></Drawer>}
      {wiz && <Wizard model={model} ids={sel} onClose={() => { setWiz(false); refresh() }} />}
    </div>)
}

function Wizard({ model, ids, onClose }: { model: string; ids: string[]; onClose: () => void }) {
  const [step, setStep] = useState(1)
  const [f, setF] = useState({ vendor_name: '', vendor_contact: '', hold_days: 90, notes: '' })
  const [report, setReport] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  const build = async () => { setErr(null); try { setReport(await post('/api/corpus/disclosure', { item_ids: ids, ...f })); setStep(4) } catch (e: any) { setErr(e.message) } }
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Vendor report wizard"><div className="card trim" style={{ maxHeight: '90vh', overflow: 'auto', width: 'min(720px,100%)' }}>
      <h2>Vendor report · step {step}/4</h2>
      {step === 1 && <div className="stack"><p className="mono">{ids.length} item(s) for <b>{model}</b>. One report = one model + one vendor. Nothing is transmitted automatically.</p><Btn onClick={() => setStep(2)}>Next ›</Btn></div>}
      {step === 2 && <div className="stack"><Field label="Vendor name"><input value={f.vendor_name} onChange={(e) => setF({ ...f, vendor_name: e.target.value })} /></Field>
        <Field label="Vendor security contact"><input value={f.vendor_contact} onChange={(e) => setF({ ...f, vendor_contact: e.target.value })} /></Field>
        <Field label="Hold before publishing (days)"><input type="number" value={f.hold_days} onChange={(e) => setF({ ...f, hold_days: Number(e.target.value) })} /></Field>
        <Btn onClick={() => setStep(3)} disabled={!f.vendor_name}>Next ›</Btn></div>}
      {step === 3 && <div className="stack"><p className="mono">Generating writes a signed report and an audit row to telemetry.disclosures, and stamps each item with a disclosure id.</p>{err && <p className="bad" role="alert">{err}</p>}<Btn onClick={build}>Generate &amp; sign ▶</Btn></div>}
      {step === 4 && report && <div className="stack"><p className="ok mono">Signed. sha256 {report.sha256.slice(0, 16)}… · hold until {report.report.provenance.hold_until.slice(0, 10)}</p>
        <iframe title="Report preview" sandbox="" srcDoc={report.html} style={{ width: '100%', height: 320, background: '#fff', border: '2px solid var(--border)' }} />
        <div className="row"><Btn onClick={() => download(`disclosure-${report.disclosure_id}.json`, JSON.stringify({ report: report.report, sha256: report.sha256, hmac_sha256: report.hmac_sha256 }, null, 1), 'application/json')}>Signed JSON</Btn>
          <Btn variant="alt" onClick={() => download(`disclosure-${report.disclosure_id}.html`, report.html, 'text/html')}>HTML</Btn></div></div>}
      <div style={{ marginTop: 16 }}><Btn variant="ghost" onClick={onClose}>Close</Btn></div>
    </div></div>)
}
