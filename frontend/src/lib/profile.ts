/** Pure data shaping for the Robustness Profile tab (no React, no DOM) — unit-tested in profile.test.ts. */

export type Rate = { n: number; k: number; rate: number; ci_low: number; ci_high: number }
export type FamilyMeta = Rate & {
  family: string; category: string | null; channel: string | null; channels: string[]
  citation: string | null; owasp_tag: string | null; atlas_tag: string | null; atlas_parent: string | null
}
export type HeatmapResp = { models: string[]; families: FamilyMeta[]; cells: (Rate & { family: string; model: string })[]; attacker_refused: number }
export type RadarResp = { families: string[]; models: { model: string; families: Record<string, Rate> }[]; attacker_refused: number }
export type CompoundingFamily = {
  family: string; category: string | null; matched_cells: number; bare: Rate; ciphered: Rate
  delta: number | null; delta_ci_low: number | null; delta_ci_high: number | null; note: string | null
  by_cipher: { cipher: string; matched_cells: number; bare: Rate; ciphered: Rate; delta: number | null }[]
}

export type HeatRow = { meta: FamilyMeta; cells: (Rate | null)[] }

/** Families × models grid; a (family, model) pair never tested is null (rendered as "—", never 0%). */
export function pivotHeatmap(h: HeatmapResp | null | undefined): { models: string[]; rows: HeatRow[] } {
  if (!h) return { models: [], rows: [] }
  const idx = new Map(h.cells.map((c) => [`${c.family}\u0000${c.model}`, c]))
  return {
    models: h.models,
    rows: h.families.map((meta) => ({ meta, cells: h.models.map((m) => idx.get(`${meta.family}\u0000${m}`) ?? null) })),
  }
}

/**
 * Recharts rows for a radar overlay. Model names such as `qwen2.5:7b` contain dots, which recharts would read as
 * nested paths if used as a dataKey, so each model gets an index key (`m0`, `m1`, …). Only families tested on EVERY
 * overlaid model become axes — a missing cell is never drawn as 0; the rest are reported in `excluded`.
 */
export function radarData(r: RadarResp | null | undefined, only?: string[]) {
  const models = (r?.models ?? []).filter((m) => !only?.length || only.includes(m.model))
  const series = models.map((m, i) => ({ key: `m${i}`, model: m.model }))
  const all = r?.families ?? []
  const axes = models.length ? all.filter((f) => models.every((m) => m.families[f])) : []
  const excluded = all.filter((f) => !axes.includes(f))
  const rows = axes.map((family) => {
    const row: Record<string, string | number> = { family }
    models.forEach((m, i) => { row[`m${i}`] = m.families[family].rate })
    return row
  })
  return { series, axes, excluded, rows }
}

const err = (x: Rate): [number, number] => [x.rate - x.ci_low, x.ci_high - x.rate]

/** Bare vs ciphered bars for families that have matched cells; the others are listed with the reason. */
export function compoundingBars(fams: CompoundingFamily[] | null | undefined) {
  const list = fams ?? []
  return {
    bars: list.filter((f) => f.matched_cells > 0).map((f) => ({
      family: f.family, bare: f.bare.rate, ciphered: f.ciphered.rate, bareErr: err(f.bare), ciphErr: err(f.ciphered),
      bareN: f.bare.n, ciphN: f.ciphered.n, delta: f.delta, deltaLo: f.delta_ci_low, deltaHi: f.delta_ci_high, cells: f.matched_cells,
    })),
    unmatched: list.filter((f) => f.matched_cells === 0).map((f) => ({ family: f.family, note: f.note ?? 'no matched cells' })),
  }
}

/** "12.5% (95% CI 4.0%–30.1%, 3/24)" — the heatmap tooltip / accessible label. */
export function rateLabel(x: Rate): string {
  const p = (v: number) => `${(v * 100).toFixed(1)}%`
  return `${p(x.rate)} (95% CI ${p(x.ci_low)}–${p(x.ci_high)}, ${x.k}/${x.n})`
}
