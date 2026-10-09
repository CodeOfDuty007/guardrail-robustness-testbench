import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Btn, Card, Chips, Field, State, download } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { post } from '../lib/api'

const OWASP = ['', 'LLM01:Prompt Injection', 'LLM02:Sensitive Information Disclosure', 'LLM07:System Prompt Leakage']
const ATLAS = ['', 'AML.T0051', 'AML.T0054', 'AML.T0056']

export default function NewExperiment() {
  const meta = useGet('/api/meta')
  const provs = useGet<any[]>('/api/providers')
  const nav = useNavigate()
  const [f, setF] = useState<any>({
    name: 'canary-sweep', engine: 'native', attacker_provider_id: '', target_provider_id: '', judge_provider_id: '', guard_provider_id: '',
    dataset: 'canary', ciphers: ['STANDARD_BASE64', 'CUSTOM_BASE64', 'AES_128_GCM'], strategies: ['DirectCipherWrap'], defenses: [],
    max_iterations: 3, seed: 1337, budget_usd: '', owasp_tag: '', atlas_tag: '', limit: '',
  })
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = (k: string, v: unknown) => setF((s: any) => ({ ...s, [k]: v }))
  const body = useMemo(() => {
    const n = (v: any) => (v === '' || v == null ? null : Number(v))
    return { ...f, attacker_provider_id: n(f.attacker_provider_id), target_provider_id: n(f.target_provider_id), judge_provider_id: n(f.judge_provider_id),
      guard_provider_id: n(f.guard_provider_id), budget_usd: n(f.budget_usd), limit: n(f.limit), seed: Number(f.seed), max_iterations: Number(f.max_iterations),
      owasp_tag: f.owasp_tag || null, atlas_tag: f.atlas_tag || null }
  }, [f])
  const ProviderSelect = ({ k, label }: { k: string; label: string }) => (
    <Field label={label}><select value={f[k]} onChange={(e) => set(k, e.target.value)}>
      <option value="">{k === 'target_provider_id' ? 'Simulated mock target (no compute)' : '— none —'}</option>
      {(provs.data ?? []).map((p) => <option key={p.id} value={p.id}>{p.label} [{p.kind}]</option>)}</select></Field>)
  const launch = async () => {
    setBusy(true); setErr(null)
    try { const { id } = await post('/api/runs', body); nav(`/live?run=${id}`) } catch (e: any) { setErr(e.message) } finally { setBusy(false) }
  }
  const yaml = () => download(`${f.name}.json`, JSON.stringify(body, null, 2), 'application/json')
  const m = meta.data
  return (
    <div>
      <h1>New experiment</h1>
      <State loading={meta.loading} error={meta.error}>{m && (<>
        <Card trim title="1 · Arena roles">
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))' }}>
            <ProviderSelect k="attacker_provider_id" label="Attacker" /><ProviderSelect k="target_provider_id" label="Target" />
            <ProviderSelect k="judge_provider_id" label="Judge LLM" /><ProviderSelect k="guard_provider_id" label="Llama Guard 3" />
          </div>
          <p className="muted mono">Add GPU models on the <a href="/compute">GPU Nodes</a> page; they appear here. Without a target the built-in simulation runs.</p>
        </Card>
        <Card title="1b · Engine">
          <div className="chips">{[['native', 'Native loop (defenses, BYO datasets)'], ['pyrit', 'PyRIT executors (Crescendo · TAP · PAIR)']].map(([v, l]) => (
            <label key={v} className="chip" style={{ marginBottom: 0 }}><input type="radio" name="engine" checked={f.engine === v} onChange={() => set('engine', v)} />{l}</label>))}</div>
          <p className="muted mono">PyRIT engine: {m.pyrit?.available ? `PyRIT ${m.pyrit.version} ready` : 'not installed'} · canary objectives only · no defenses. Without an attacker model a scripted attacker is used (mock mode).</p>
        </Card>
        <Card title="2 · Ciphers"><Chips label="Ciphers" value={f.ciphers} onChange={(v) => set('ciphers', v)} options={m.ciphers.map((c: any) => ({ value: c.name, hint: c.label }))} /></Card>
        <Card title="3 · Strategies"><Chips label="Strategies" value={f.strategies} onChange={(v) => set('strategies', v)} options={m.strategies.map((s: any) => ({ value: s.name, hint: s.adaptive ? 'adaptive' : 'static', label: s.name + (s.adaptive ? ' ⟳' : '') }))} />
          <h3 style={{ marginTop: 16 }}>Defenses (ablation)</h3><Chips label="Defenses" value={f.defenses} onChange={(v) => set('defenses', v)} options={m.defenses.map((d: string) => ({ value: d }))} /></Card>
        <Card title="4 · Objectives">
          <div className="chips">{m.datasets.map((d: any) => (
            <label key={d.name} className="chip" style={{ opacity: d.available ? 1 : .5, marginBottom: 0 }}>
              <input type="radio" name="ds" disabled={!d.available} checked={f.dataset === d.name} onChange={() => set('dataset', d.name)} />
              {d.name} {d.available ? `(${d.count})` : '— BYO'}</label>))}</div>
          <p className="muted mono">Canary objectives are benign. JBB / HarmBench / StrongReject must be supplied by you in <code>datasets/</code>.</p>
        </Card>
        <Card title="5 · Limits & tags">
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))' }}>
            <Field label="Run name"><input value={f.name} onChange={(e) => set('name', e.target.value)} /></Field>
            <Field label="Seed"><input type="number" value={f.seed} onChange={(e) => set('seed', e.target.value)} /></Field>
            <Field label="Max iterations"><input type="number" min={1} value={f.max_iterations} onChange={(e) => set('max_iterations', e.target.value)} /></Field>
            <Field label="Budget cap (USD)"><input type="number" step="0.01" placeholder="none" value={f.budget_usd} onChange={(e) => set('budget_usd', e.target.value)} /></Field>
            <Field label="Objective limit"><input type="number" placeholder="all" value={f.limit} onChange={(e) => set('limit', e.target.value)} /></Field>
            <Field label="OWASP tag"><select value={f.owasp_tag} onChange={(e) => set('owasp_tag', e.target.value)}>{OWASP.map((o) => <option key={o}>{o}</option>)}</select></Field>
            <Field label="ATLAS tag"><select value={f.atlas_tag} onChange={(e) => set('atlas_tag', e.target.value)}>{ATLAS.map((o) => <option key={o}>{o}</option>)}</select></Field>
          </div>
        </Card>
        <Card title="6 · Review & launch">
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(body, null, 1)}</pre>
          {err && <p className="bad" role="alert">{err}</p>}
          <div className="row" style={{ marginTop: 16 }}>
            <Btn onClick={launch} disabled={busy || !f.ciphers.length || !f.strategies.length}>{busy ? 'Launching…' : 'Start ▶'}</Btn>
            <Btn variant="ghost" onClick={yaml}>Download config</Btn>
          </div>
        </Card></>)}
      </State>
    </div>
  )
}
