import { useState } from 'react'
import { Btn, Card, Field, Pill, State } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { del, post, put } from '../lib/api'

const KINDS = ['openai', 'anthropic', 'gemini', 'groq', 'azureopenai', 'ollama', 'vllm', 'openai_compat', 'mock']

export default function Settings() {
  const provs = useGet<any[]>('/api/providers')
  const [f, setF] = useState({ label: '', kind: 'openai', model_name: '', base_url: '', role_hint: 'target', cost_per_1k_prompt: 0, cost_per_1k_completion: 0 })
  const [keys, setKeys] = useState<Record<number, string>>({})
  const [remember, setRemember] = useState(false)
  const [health, setHealth] = useState<Record<number, any>>({})
  const [err, setErr] = useState<string | null>(null)
  const add = async () => {
    setErr(null)
    try { await post('/api/providers', { ...f, base_url: f.base_url || null }); setF({ ...f, label: '', model_name: '' }); provs.reload() } catch (e: any) { setErr(e.message) }
  }
  const saveKey = async (id: number) => { await put(`/api/providers/${id}/key`, { key: keys[id], remember }); setKeys((k) => ({ ...k, [id]: '' })); provs.reload() }
  const validate = async (id: number) => {
    setHealth((h) => ({ ...h, [id]: { busy: true } }))
    const r = await post(`/api/providers/${id}/validate`)
    setHealth((h) => ({ ...h, [id]: r }))
  }
  return (
    <div>
      <h1>Settings · providers (BYOK)</h1>
      <Card trim title="Key handling">
        <p className="mono">Keys live in server <b>process memory</b> only and are never written to either database, logs, exports or reports. Tick “remember” to store in the OS keyring instead.</p>
        <label className="row" style={{ textTransform: 'none', fontSize: 18 }}><input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} /> Remember keys on this machine (OS keyring)</label>
      </Card>
      <Card title="Add provider">
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))' }}>
          <Field label="Label"><input value={f.label} onChange={(e) => setF({ ...f, label: e.target.value })} /></Field>
          <Field label="Kind"><select value={f.kind} onChange={(e) => setF({ ...f, kind: e.target.value })}>{KINDS.map((k) => <option key={k}>{k}</option>)}</select></Field>
          <Field label="Model name"><input value={f.model_name} placeholder="gpt-4o / llama3:8b" onChange={(e) => setF({ ...f, model_name: e.target.value })} /></Field>
          <Field label="Base URL (optional)"><input value={f.base_url} placeholder="http://gpu:11434" onChange={(e) => setF({ ...f, base_url: e.target.value })} /></Field>
          <Field label="USD / 1k prompt tokens"><input type="number" step="0.0001" min={0} value={f.cost_per_1k_prompt} onChange={(e) => setF({ ...f, cost_per_1k_prompt: Number(e.target.value) })} /></Field>
          <Field label="USD / 1k completion tokens"><input type="number" step="0.0001" min={0} value={f.cost_per_1k_completion} onChange={(e) => setF({ ...f, cost_per_1k_completion: Number(e.target.value) })} /></Field>
          <Field label="Role"><select value={f.role_hint} onChange={(e) => setF({ ...f, role_hint: e.target.value })}>{['target', 'attacker', 'judge'].map((k) => <option key={k}>{k}</option>)}</select></Field>
        </div>
        {err && <p className="bad" role="alert">{err}</p>}
        <Btn onClick={add} disabled={!f.label || !f.model_name}>Add ▶</Btn>
      </Card>
      <State loading={provs.loading} error={provs.error} empty={!provs.data?.length}>
        {provs.data?.map((p) => (
          <Card key={p.id}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div><b>{p.label}</b> <Pill tone="blue">{p.kind}</Pill> <span className="mono muted">{p.model_name} {p.base_url ?? ''}</span></div>
              <div className="row">{p.has_key ? <Pill tone="green">key set</Pill> : <Pill>no key</Pill>}
                {health[p.id] && !health[p.id].busy && <Pill tone={health[p.id].ok ? 'green' : 'red'}>{health[p.id].ok ? `ok ${health[p.id].latency_ms?.toFixed(0)}ms` : `fail ${health[p.id].error}`}</Pill>}
                <Btn variant="ghost" onClick={() => validate(p.id)}>Validate</Btn>
                <Btn variant="danger" onClick={async () => { await del(`/api/providers/${p.id}`); provs.reload() }}>Delete</Btn></div>
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <input type="password" autoComplete="off" aria-label={`API key for ${p.label}`} placeholder="paste API key (never shown again)" value={keys[p.id] ?? ''} onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })} style={{ maxWidth: 420 }} />
              <Btn onClick={() => saveKey(p.id)} disabled={!keys[p.id]}>Save key</Btn>
            </div>
          </Card>))}
      </State>
    </div>
  )
}
