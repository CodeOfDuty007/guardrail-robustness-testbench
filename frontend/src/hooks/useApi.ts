import { useCallback, useEffect, useRef, useState } from 'react'
import { get } from '../lib/api'

export function useGet<T = any>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(!!path)
  const seq = useRef(0)
  const lastPath = useRef<string | null>(null)
  const reload = useCallback(() => {
    if (!path) return
    const id = ++seq.current
    if (lastPath.current !== path) { setData(null); lastPath.current = path } // never show another query's rows
    setLoading(true)
    get<T>(path)
      .then((d) => { if (id === seq.current) { setData(d); setError(null) } })
      .catch((e) => { if (id === seq.current) setError(String(e.message || e)) })
      .finally(() => { if (id === seq.current) setLoading(false) })
  }, [path])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reload, [reload, ...deps])
  return { data, error, loading, reload }
}
