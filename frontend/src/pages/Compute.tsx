import { useState } from 'react'
import { Btn, Card, Field, Pill, State, StatusPill } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { del, post } from '../lib/api'

export default function Compute() {
  const nodes = useGet<any[]>('/api/compute/nodes')
  const [f, setF] = useState({ label: '', base_url: 'http://localhost:11434', kind: 'ollama', token: '' })
  const [pick, setPick] = useState<Record<number, string[]>>({})
  const [role, setRole] = useState('target')
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const add = async () => {
    setBusy(true); setMsg(null)
    try { await post('/api/compute/nodes', { ...f, token: f.token || null }); setF({ ...f, label: '', token: '' }); nodes.reload() } catch (e: any) { setMsg(e.message) } finally { setBusy(false) }
  }
  const adopt = async (id: number) => {
    try { const r = await fetch(`/api/compute/nodes/${id}/adopt?role=${role}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pick[id] ?? []) }); if (!r.ok) throw new Error(await r.text())
      setMsg(`Registered ${(pick[id] ?? []).length} model(s) as ${role} providers — pick them in New Run.`); setPick((p) => ({ ...p, [id]: [] })) } catch (e: any) { setMsg(e.message) }
  }
  return (
    <div>
      <h1>GPU nodes</h1>
      <Card trim title="Connect a GPU host">
        <p className="mono">Any host serving an OpenAI-compatible API works: Ollama, vLLM, llama.cpp. Start one with <code>compute/docker-compose.gpu.yml</code> on the GPU box, then reach it via VPN/Tailscale or <code>ssh -L 11434:localhost:11434 gpu</code> and add <code>http://localhost:11434</code> here.</p>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))' }}>
          <Field label="Label"><input value={f.label} placeholder="lab-4090" onChange={(e) => setF({ ...f, label: e.target.value })} /></Field>
          <Field label="Base URL"><input value={f.base_url} onChange={(e) => setF({ ...f, base_url: e.target.value })} /></Field>
          <Field label="Server"><select value={f.kind} onChange={(e) => setF({ ...f, kind: e.target.value })}><option>ollama</option><option>vllm</option><option>openai_compat</option></select></Field>
          <Field label="Bearer token (optional, memory only)"><input type="password" autoComplete="off" value={f.token} onChange={(e) => setF({ ...f, token: e.target.value })} /></Field>
        </div>
        <div className="row" style={{ marginTop: 12 }}><Btn onClick={add} disabled={busy || !f.label || !f.base_url}>{busy ? 'Probing…' : 'Connect & probe ▶'}</Btn>{msg && <span role="status" className="mono">{msg}</span>}</div>
      </Card>
      <State loading={nodes.loading} error={nodes.error} empty={!nodes.data?.length}>
        {nodes.data?.map((n) => (
          <Card key={n.id}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div><b>{n.label}</b> <StatusPill s={n.status} /> <Pill tone="blue">{n.kind}</Pill> <span className="mono muted">{n.base_url} {n.latency_ms ? `· ${n.latency_ms.toFixed(0)}ms` : ''}</span></div>
              <div className="row"><Btn variant="ghost" onClick={async () => { await post(`/api/compute/nodes/${n.id}/probe`); nodes.reload() }}>Re-probe</Btn><Btn variant="danger" onClick={async () => { await del(`/api/compute/nodes/${n.id}`); nodes.reload() }}>Remove</Btn></div>
            </div>
            {n.gpu_info && <div className="mono muted">GPU/VRAM: {n.gpu_info}</div>}
            {n.models.length > 0 && (<div style={{ marginTop: 12 }}>
              <div className="chips">{n.models.map((m: string) => (
                <label key={m} className="chip" style={{ marginBottom: 0 }}><input type="checkbox" checked={(pick[n.id] ?? []).includes(m)}
                  onChange={(e) => setPick((p) => ({ ...p, [n.id]: e.target.checked ? [...(p[n.id] ?? []), m] : (p[n.id] ?? []).filter((x) => x !== m) }))} />{m}</label>))}</div>
              <div className="row" style={{ marginTop: 8 }}>
                <select style={{ width: 160 }} aria-label="Role" value={role} onChange={(e) => setRole(e.target.value)}><option>target</option><option>attacker</option><option>judge</option></select>
                <Btn onClick={() => adopt(n.id)} disabled={!(pick[n.id] ?? []).length}>Use as provider ›</Btn></div></div>)}
            {n.status === 'offline' && <p className="bad mono">Unreachable. Check the tunnel/VPN and that the server is listening on 0.0.0.0 (Ollama: <code>OLLAMA_HOST=0.0.0.0</code>).</p>}
          </Card>))}
      </State>
    </div>
  )
}
