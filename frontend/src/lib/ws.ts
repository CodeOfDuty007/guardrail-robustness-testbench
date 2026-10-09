export type LiveEvent = { type: 'attempt' | 'fsm' | 'run'; run_id: number; ts: string; [k: string]: any }

export function connectLive(runId: number | null, onEvent: (e: LiveEvent) => void, onState?: (up: boolean) => void) {
  let ws: WebSocket | null = null
  let closed = false
  let retry = 500
  const open = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws/live${runId ? `?run_id=${runId}` : ''}`)
    ws.onopen = () => { retry = 500; onState?.(true) }
    ws.onmessage = (m) => onEvent(JSON.parse(m.data))
    ws.onclose = () => { onState?.(false); if (!closed) setTimeout(open, (retry = Math.min(retry * 2, 8000))) }
  }
  open()
  return () => { closed = true; ws?.close() }
}
