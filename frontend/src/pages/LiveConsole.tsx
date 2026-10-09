import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Btn, Card, Pill, StatusPill } from '../components/ui'
import { useGet } from '../hooks/useApi'
import { post } from '../lib/api'
import { connectLive, type LiveEvent } from '../lib/ws'

const STATES = ['GENERATE', 'MUTATE', 'DISPATCH', 'EVALUATE', 'PERSIST', 'ADAPT']

export default function LiveConsole() {
  const [sp, setSp] = useSearchParams()
  const runId = sp.get('run') ? Number(sp.get('run')) : null
  const runs = useGet<any[]>('/api/runs')
  const [events, setEvents] = useState<LiveEvent[]>([])
  const [fsm, setFsm] = useState<Record<number, string>>({})
  const [runStatus, setRunStatus] = useState<string>('')
  const [up, setUp] = useState(false)
  const [promoted, setPromoted] = useState<Record<number, string>>({})

  useEffect(() => {
    setEvents([]); setFsm({})
    return connectLive(runId, (e) => {
      if (e.type === 'attempt') setEvents((p) => [e, ...p].slice(0, 200))
      else if (e.type === 'fsm') setFsm((s) => ({ ...s, [e.attempt_id]: e.state }))
      else if (e.type === 'run') { setRunStatus(e.status); runs.reload() }
    }, setUp)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  const stats = useMemo(() => ({ n: events.length, ev: events.filter((e) => e.filter_evaded).length, hit: events.filter((e) => e.promote_eligible).length }), [events])
  const promote = async (id: number) => {
    try { const r = await post('/api/corpus/promote', { source_attempt_id: id }); setPromoted((p) => ({ ...p, [id]: `in corpus (${r.status})` })) }
    catch (e: any) { setPromoted((p) => ({ ...p, [id]: e.message })) }
  }
  return (
    <div>
      <h1>Live console</h1>
      <Card trim>
        <div className="row">
          <label style={{ margin: 0 }}>Run
            <select style={{ width: 260 }} value={runId ?? ''} onChange={(e) => setSp(e.target.value ? { run: e.target.value } : {})}>
              <option value="">All runs (firehose)</option>
              {(runs.data ?? []).map((r) => <option key={r.id} value={r.id}>#{r.id} {r.name} — {r.status}</option>)}</select></label>
          <Pill tone={up ? 'green' : 'red'}>{up ? 'stream live' : 'reconnecting'}</Pill>
          {runStatus && <StatusPill s={runStatus} />}
          <span className="mono muted">{stats.n} attempts · {stats.ev} evaded · <span className="bad">{stats.hit} confirmed</span></span>
        </div>
      </Card>
      <div role="log" aria-live="polite" aria-label="Attempt stream">
        {events.length === 0 && <Card><span className="muted">WAITING FOR ATTEMPTS… start a run from “New Run”.</span></Card>}
        <AnimatePresence initial={false}>
          {events.map((e) => (
            <motion.div key={e.attempt_id} layout initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} className={`term ${e.promote_eligible ? 'hit' : ''}`}>
              <div className="fsm" aria-label={`state ${fsm[e.attempt_id] ?? 'DONE'}`}>{STATES.map((s) => <span key={s} className={(fsm[e.attempt_id] ?? 'ADAPT') === s ? 'on' : ''}>{s}</span>)}</div>
              <div className="row"><Pill tone="blue">#{e.attempt_id}</Pill><b>{e.strategy}</b><span>×</span><b>{e.cipher}</b><span className="muted">iter {e.iteration} · {e.objective_id} · {e.target_model}</span></div>
              <div><span className="tag">[plaintext]</span> <span className="val">{e.plaintext}</span></div>
              <div><span className="tag">[payload]</span> <span className="val">{e.ciphertext}</span></div>
              <div><span className="tag">[response]</span> <span className="val">{e.response}</span></div>
              <div className="row" style={{ marginTop: 8 }}>
                <Pill tone={e.filter_evaded ? 'amber' : 'green'}>{e.filter_evaded ? 'filter evaded' : 'refused'}</Pill>
                <Pill tone={e.semantic_complied ? 'red' : 'grey'}>{e.semantic_complied ? 'complied' : 'not complied'}</Pill>
                {e.blocked && <Pill tone="blue">defense blocked</Pill>}
                <span className="muted">H̄={e.metrics?.shannon_entropy?.toFixed(2)} · aval={e.metrics?.avalanche_score?.toFixed(3)} · χ²={e.metrics?.confusion_chi2?.toFixed(0)} · {e.latency_ms?.toFixed(0)}ms</span>
                {e.promote_eligible && (promoted[e.attempt_id]
                  ? <span className="ok">{promoted[e.attempt_id]}</span>
                  : <Btn variant="alt" onClick={() => promote(e.attempt_id)}>Promote to corpus ›</Btn>)}
              </div>
            </motion.div>))}
        </AnimatePresence>
      </div>
    </div>
  )
}
