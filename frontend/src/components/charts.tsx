import { type ReactNode } from 'react'

export const COLORS = ['#ffda14', '#b9a8ff', '#4ade80', '#fb7185', '#38bdf8', '#fb923c', '#e879f9', '#f5f5ff']
export const TIP = { contentStyle: { background: 'var(--surface)', border: '2px solid var(--secondary)', fontFamily: 'JetBrains Mono', fontSize: 12 }, labelStyle: { color: 'var(--secondary)' } }
export const AX = { stroke: 'var(--muted)', tick: { fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--muted)' } }
export const pctTick = (v: number) => `${Math.round(v * 100)}%`
export const colorOf = (names: string[], n: string) => COLORS[Math.max(0, names.indexOf(n)) % COLORS.length]

export type Status = { loading?: boolean; error?: string | null; empty?: boolean }
export type TableData = { headers: string[]; rows: (string | number | null)[][] }

/** The numbers behind a chart, readable without seeing it (screen readers, copy/paste). */
export function DataDetails({ table, caption }: { table: TableData; caption: string }) {
  if (!table.rows.length) return null
  return (
    <details style={{ marginTop: 8 }}>
      <summary className="mono" style={{ cursor: 'pointer' }}>View the data as a table ({table.rows.length} rows)</summary>
      <div className="tablewrap" style={{ maxHeight: 260, overflow: 'auto', marginTop: 8 }}>
        <table><caption className="sr">{caption}</caption>
          <thead><tr>{table.headers.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
          <tbody>{table.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c ?? '—'}</td>)}</tr>)}</tbody></table>
      </div>
    </details>
  )
}

/** One chart panel with its own loading / error / empty handling, so one failing request never hides the rest. */
export function ChartCard({ title, note, height = 320, label, status, table, children }: {
  title: string; note?: string; height?: number; label: string; status?: Status; table?: TableData; children: ReactNode
}) {
  return (
    <section className="card" aria-labelledby={undefined}>
      <h3>{title}</h3>
      {status?.error ? <p className="bad" role="alert">Could not load this chart: {status.error}</p>
        : status?.loading ? <p className="muted" role="status">Loading…</p>
        : status?.empty ? <p className="muted">No data for the current filters.</p>
        : (<>
          <div style={{ height }} role="img" aria-label={label}>{children}</div>
          {table && <DataDetails table={table} caption={title} />}
        </>)}
      {note && <p className="muted mono" style={{ fontSize: 13, marginBottom: 0 }}>{note}</p>}
    </section>
  )
}

/** Heat colour: 0% = success green → 100% = danger red, via the arcade palette. */
export function heat(rate: number): string {
  const r = Math.round(22 + (220 - 22) * rate), g = Math.round(163 - (163 - 38) * rate), b = Math.round(74 - (74 - 38) * rate)
  return `rgb(${r},${g},${b})`
}
