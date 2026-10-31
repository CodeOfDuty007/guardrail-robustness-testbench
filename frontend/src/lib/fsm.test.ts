import { describe, expect, it } from 'vitest'
import { fsmSteps, pushState } from './fsm'

const cls = (h: string[] | undefined) => Object.fromEntries(fsmSteps(h).map((s) => [s.state, s.cls]))

describe('fsmSteps', () => {
  it('marks passed states done and the current one on', () => {
    expect(cls(['GENERATE', 'MUTATE', 'DISPATCH'])).toMatchObject({ GENERATE: 'done', MUTATE: 'done', DISPATCH: 'on', EVALUATE: '' })
  })
  it('a retried attempt ends on ADAPT and a confirmed one on DONE, never both', () => {
    const base = ['GENERATE', 'MUTATE', 'DISPATCH', 'EVALUATE', 'PERSIST']
    expect(cls([...base, 'ADAPT'])).toMatchObject({ PERSIST: 'done', ADAPT: 'on', DONE: '' })
    expect(cls([...base, 'DONE'])).toMatchObject({ PERSIST: 'done', DONE: 'on', ADAPT: '' })
  })
  it('flags states the engine never reported (PyRIT path) as skipped, not missing', () => {
    expect(cls(['DISPATCH', 'ADAPT'])).toMatchObject({ GENERATE: 'skip', MUTATE: 'skip', DISPATCH: 'done', EVALUATE: 'skip', ADAPT: 'on' })
  })
  it('shows nothing lit when no event has arrived', () => {
    expect(fsmSteps(undefined).every((s) => s.cls === '')).toBe(true)
  })
})

describe('pushState', () => {
  it('appends new states and ignores an immediate repeat', () => {
    expect(pushState(undefined, 'GENERATE')).toEqual(['GENERATE'])
    expect(pushState(['GENERATE'], 'GENERATE')).toEqual(['GENERATE'])
    expect(pushState(['GENERATE'], 'MUTATE')).toEqual(['GENERATE', 'MUTATE'])
  })
})
