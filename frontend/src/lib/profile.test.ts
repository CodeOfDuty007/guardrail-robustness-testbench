import { describe, expect, it } from 'vitest'
import { compoundingBars, pivotHeatmap, radarData, rateLabel, type CompoundingFamily, type HeatmapResp, type RadarResp } from './profile'

const R = (k: number, n: number) => ({ k, n, rate: k / n, ci_low: Math.max(0, k / n - 0.1), ci_high: Math.min(1, k / n + 0.1) })
const meta = (family: string) => ({ ...R(1, 4), family, category: 'jailbreak', channel: 'direct', channels: ['direct'], citation: null, owasp_tag: null, atlas_tag: null, atlas_parent: null })

describe('pivotHeatmap', () => {
  it('builds a families × models grid and leaves untested pairs null (not 0)', () => {
    const h: HeatmapResp = {
      models: ['qwen2.5:7b', 'mock-weak-1b'], families: [meta('persuasion'), meta('many_shot')], attacker_refused: 0,
      cells: [{ family: 'persuasion', model: 'qwen2.5:7b', ...R(1, 2) }, { family: 'persuasion', model: 'mock-weak-1b', ...R(2, 2) },
        { family: 'many_shot', model: 'mock-weak-1b', ...R(0, 3) }],
    }
    const { models, rows } = pivotHeatmap(h)
    expect(models).toEqual(['qwen2.5:7b', 'mock-weak-1b'])
    expect(rows.map((r) => r.meta.family)).toEqual(['persuasion', 'many_shot'])
    expect(rows[0].cells.map((c) => c?.rate)).toEqual([0.5, 1])
    expect(rows[1].cells[0]).toBeNull()
    expect(rows[1].cells[1]?.n).toBe(3)
    expect(pivotHeatmap(null)).toEqual({ models: [], rows: [] })
  })
})

describe('radarData', () => {
  const r: RadarResp = {
    families: ['a', 'b', 'c', 'd'], attacker_refused: 0,
    models: [
      { model: 'qwen2.5:7b', families: { a: R(1, 4), b: R(2, 4), c: R(0, 4), d: R(4, 4) } },
      { model: 'llama3:8b', families: { a: R(3, 4), b: R(1, 4), c: R(1, 4) } },
    ],
  }
  it('keys series by index (dotted model names) and keeps only families every model was tested on', () => {
    const d = radarData(r)
    expect(d.series).toEqual([{ key: 'm0', model: 'qwen2.5:7b' }, { key: 'm1', model: 'llama3:8b' }])
    expect(d.axes).toEqual(['a', 'b', 'c'])
    expect(d.excluded).toEqual(['d'])
    expect(d.rows[0]).toEqual({ family: 'a', m0: 0.25, m1: 0.75 })
    expect(Object.keys(d.rows[0])).not.toContain('qwen2.5:7b')
  })
  it('restricting to one model brings its own families back', () => {
    const d = radarData(r, ['qwen2.5:7b'])
    expect(d.axes).toEqual(['a', 'b', 'c', 'd'])
    expect(d.excluded).toEqual([])
    expect(radarData(undefined)).toEqual({ series: [], axes: [], excluded: [], rows: [] })
  })
})

describe('compoundingBars', () => {
  it('plots matched families with CI error bars and lists the rest with their reason', () => {
    const fam = (family: string, cells: number, note: string | null): CompoundingFamily => ({
      family, category: 'jailbreak', matched_cells: cells, bare: R(1, 10), ciphered: R(5, 10), delta: cells ? 0.4 : null,
      delta_ci_low: cells ? 0.05 : null, delta_ci_high: cells ? 0.75 : null, note, by_cipher: [],
    })
    const { bars, unmatched } = compoundingBars([fam('persuasion', 6, null), fam('roleplay_persona', 0, 'never run with a cipher')])
    expect(bars).toHaveLength(1)
    expect(bars[0]).toMatchObject({ family: 'persuasion', bare: 0.1, ciphered: 0.5, bareN: 10, cells: 6, delta: 0.4, deltaLo: 0.05, deltaHi: 0.75 })
    expect(bars[0].bareErr[0]).toBeCloseTo(0.1)
    expect(bars[0].ciphErr[1]).toBeCloseTo(0.1)
    expect(bars[0].ciText).toBe('5.0 to 75.0')
    expect(unmatched).toEqual([{ family: 'roleplay_persona', note: 'never run with a cipher' }])
  })
  it('prefers n_cells and shows the reason when the CI is null (fewer than 2 cells)', () => {
    const one: CompoundingFamily = {
      family: 'many_shot', category: 'jailbreak', matched_cells: 1, n_cells: 1, bare: R(0, 1), ciphered: R(1, 1), delta: 1,
      delta_ci_low: null, delta_ci_high: null, delta_note: 'only 1 matched cell: no confidence interval', note: null, by_cipher: [],
    }
    const { bars } = compoundingBars([one])
    expect(bars[0].cells).toBe(1)
    expect(bars[0].ciText).toBe('only 1 matched cell: no confidence interval')
    expect(bars[0].deltaLo).toBeNull()
  })
})

it('rateLabel shows rate, CI and k/n', () => {
  expect(rateLabel({ k: 3, n: 24, rate: 0.125, ci_low: 0.04, ci_high: 0.301 })).toBe('12.5% (95% CI 4.0%–30.1%, 3/24)')
})
