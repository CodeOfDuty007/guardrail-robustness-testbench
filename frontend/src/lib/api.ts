export async function api<T = any>(path: string, opts: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, ...rest } = opts
  const r = await fetch(path, {
    ...rest,
    headers: { ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}), ...rest.headers },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try { const d = JSON.parse(t).detail; msg = typeof d === 'string' ? d : JSON.stringify(d) } catch { /* plain text */ }
    throw new Error(msg || r.statusText)
  }
  const ct = r.headers.get('content-type') || ''
  return (ct.includes('json') ? r.json() : r.text()) as Promise<T>
}
export const get = <T = any>(p: string) => api<T>(p)
export const post = <T = any>(p: string, json?: unknown) => api<T>(p, { method: 'POST', json: json ?? {} })
export const put = <T = any>(p: string, json: unknown) => api<T>(p, { method: 'PUT', json })
export const patch = <T = any>(p: string, json: unknown) => api<T>(p, { method: 'PATCH', json })
export const del = <T = any>(p: string) => api<T>(p, { method: 'DELETE' })
