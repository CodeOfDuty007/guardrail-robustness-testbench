import { type ReactNode } from 'react'
import { StatusPill } from '../components/ui'

import { headerLabel } from './headers'
export const header = headerLabel

const CIPHERS: Record<string, string> = {
  STANDARD_BASE64: 'Base64', CUSTOM_BASE64: 'Custom Base64', XOR_MASK: 'XOR mask', AES_128_GCM: 'AES-128-GCM', ROT_N: 'ROT-N (Caesar)', NONE: 'None (plain text)',
}
const OBJECTIVES: Record<string, string> = {
  reveal_planted_secret: 'Reveal the planted secret', emit_sentinel_string: 'Output the marker string', leak_system_prompt: 'Leak the system prompt',
}
const STATUS: Record<string, string> = { success: 'Bypassed', failed: 'Held', timeout: 'Timed out', running: 'Running', done: 'Finished', error: 'Error', pending: 'Waiting' }
export const cipherName = (c: string) => CIPHERS[c] ?? c
export const objectiveName = (o: string) => OBJECTIVES[o] ?? o.replace(/_/g, ' ')
export const statusName = (s: string) => STATUS[s] ?? s

export function fmtDate(v: string): string {
  const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(v) ? v : `${v}Z`)
  if (isNaN(d.getTime())) return v
  return d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/** Human-friendly rendering of a single cell. The raw value is kept in the tooltip. */
export function cell(key: string, v: unknown): ReactNode {
  if (v == null || v === '') return <span className="muted">—</span>
  if (typeof v === 'boolean') return v ? <span className="ok">Yes</span> : <span className="muted">No</span>
  if (key === 'status' && typeof v === 'string') return <StatusPill s={v} label={statusName(v)} />
  if ((key === 'cipher_type' || key === 'cipher') && typeof v === 'string') return <span title={v}>{cipherName(v)}</span>
  if (key === 'objective_id' && typeof v === 'string') return <span title={v}>{objectiveName(v)}</span>
  if (/(_at|^generated_at|^hold_until)$/.test(key) && typeof v === 'string') return <span title={v}>{fmtDate(v)}</span>
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v)
  return <span title={s}>{s}</span>
}
