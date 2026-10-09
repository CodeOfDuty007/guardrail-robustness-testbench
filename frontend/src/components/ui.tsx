import { animate, motion } from 'framer-motion'
import { useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { headerLabel } from '../lib/headers'
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type ColumnDef, type SortingState } from '@tanstack/react-table'

export const Card = ({ children, trim, title, className = '' }: { children: ReactNode; trim?: boolean; title?: string; className?: string }) => (
  <section className={`card ${trim ? 'trim' : ''} ${className}`}>{title && <h3>{title}</h3>}{children}</section>
)

export const Btn = ({ variant, ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'alt' | 'ghost' | 'danger' }) => (
  <button type="button" {...p} className={`btn ${variant ?? ''} ${p.className ?? ''}`} />
)

export const Pill = ({ tone = 'grey', children }: { tone?: 'red' | 'amber' | 'green' | 'grey' | 'blue'; children: ReactNode }) => (
  <span className={`pill ${tone}`}>{children}</span>
)

const STATUS_TONE: Record<string, 'red' | 'amber' | 'green' | 'grey' | 'blue'> = {
  working: 'red', patched: 'green', unverified: 'grey', success: 'red', failed: 'green', timeout: 'amber',
  running: 'blue', done: 'green', error: 'red', pending: 'grey', online: 'green', offline: 'red', unknown: 'grey',
}
export const StatusPill = ({ s, label }: { s: string; label?: string }) => <Pill tone={STATUS_TONE[s] ?? 'grey'}>{label ?? s}</Pill>

export function Kpi({ label, value, fmt = (n: number) => Math.round(n).toLocaleString() }: { label: string; value: number; fmt?: (n: number) => string }) {
  const [shown, setShown] = useState(0)
  const prev = useRef(0)
  useEffect(() => {
    const c = animate(prev.current, value, { duration: 0.7, onUpdate: setShown })
    prev.current = value
    return () => c.stop()
  }, [value])
  return (
    <Card trim>
      <div className="kpi" aria-label={`${label}: ${fmt(value)}`}>{fmt(shown)}</div>
      <div className="kpi-label">{label}</div>
    </Card>
  )
}

export function State({ loading, error, empty, children }: { loading?: boolean; error?: string | null; empty?: boolean; children: ReactNode }) {
  if (error) return <Card><span className="bad" role="alert">ERROR: {error}</span><div className="muted mono">Is the backend running? <code>cd backend && uv run uvicorn app.main:app</code></div></Card>
  if (loading) return <div className="muted" role="status">LOADING…</div>
  if (empty) return <Card><span className="muted">NO DATA YET — launch an experiment first.</span></Card>
  return <>{children}</>
}

export function Chips({ options, value, onChange, label }: { options: { value: string; label?: string; hint?: string }[]; value: string[]; onChange: (v: string[]) => void; label: string }) {
  return (
    <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
      <legend className="sr">{label}</legend>
      <div className="chips">
        {options.map((o) => (
          <label key={o.value} className="chip" title={o.hint} style={{ marginBottom: 0, color: 'inherit' }}>
            <input type="checkbox" checked={value.includes(o.value)}
              onChange={(e) => onChange(e.target.checked ? [...value, o.value] : value.filter((v) => v !== o.value))} />
            {o.label ?? o.value}
          </label>
        ))}
      </div>
    </fieldset>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return <div><label>{label}{children}</label></div>
}

export function DataTable<T extends object>({ rows, columns, onRow, selectable, selected, onSelect, idKey }: {
  rows: T[]; columns: ColumnDef<T, any>[]; onRow?: (r: T) => void
  selectable?: boolean; selected?: string[]; onSelect?: (ids: string[]) => void; idKey?: keyof T
}) {
  const [sorting, setSorting] = useState<SortingState>([])
  const cols: ColumnDef<T, any>[] = selectable && idKey ? [{
    id: '_sel', header: '', enableSorting: false,
    cell: ({ row }) => {
      const id = String(row.original[idKey])
      return <input type="checkbox" aria-label={`select ${id}`} checked={selected?.includes(id)} onClick={(e) => e.stopPropagation()}
        onChange={(e) => onSelect?.(e.target.checked ? [...(selected ?? []), id] : (selected ?? []).filter((x) => x !== id))} />
    },
  }, ...columns] : columns
  const t = useReactTable({ data: rows, columns: cols, state: { sorting }, onSortingChange: setSorting, getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel() })
  return (
    <div className="tablewrap">
      <table>
        <thead>{t.getHeaderGroups().map((g) => (
          <tr key={g.id}>{g.headers.map((h) => (
            <th key={h.id} onClick={h.column.getToggleSortingHandler()} aria-sort={h.column.getIsSorted() === 'asc' ? 'ascending' : h.column.getIsSorted() === 'desc' ? 'descending' : 'none'}>
              {flexRender(h.column.columnDef.header, h.getContext())}{{ asc: ' ▲', desc: ' ▼' }[h.column.getIsSorted() as string] ?? ''}
            </th>))}</tr>))}
        </thead>
        <tbody>{t.getRowModel().rows.map((r) => (
          <tr key={r.id} onClick={() => onRow?.(r.original)} style={{ cursor: onRow ? 'pointer' : undefined }}>
            {r.getVisibleCells().map((c) => <td key={c.id}>{flexRender(c.column.columnDef.cell, c.getContext())}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  )
}

export function Drawer({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => { const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h) }, [onClose])
  return (
    <motion.aside className="drawer" role="dialog" aria-label={title} initial={{ x: 80, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}><h2>{title}</h2><Btn variant="ghost" onClick={onClose}>Close ✕</Btn></div>
      {children}
    </motion.aside>
  )
}

export const KV = ({ data }: { data: Record<string, unknown> }) => (
  <div className="stack">{Object.entries(data).map(([k, v]) => (
    <div key={k}><div className="muted" style={{ fontSize: 18 }} title={k}>{headerLabel(k)}</div>
      <pre className="mono" style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{typeof v === 'object' ? JSON.stringify(v, null, 1) : String(v)}</pre></div>))}</div>
)

export function download(name: string, text: string, type = 'text/plain') {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}
export const pct = (n: number | null | undefined) => (n == null ? '—' : `${(n * 100).toFixed(1)}%`)
