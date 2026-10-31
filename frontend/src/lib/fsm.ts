// Live Console attempt FSM: GENERATE → MUTATE → DISPATCH → EVALUATE → PERSIST → ADAPT (retry) | DONE (confirmed bypass).
// The server emits one event per state, so the console keeps the history and shows every step an attempt passed through.
export const FSM_STATES = ['GENERATE', 'MUTATE', 'DISPATCH', 'EVALUATE', 'PERSIST', 'ADAPT', 'DONE'] as const

// on = current state, done = visited, skip = passed without an event (engine-managed, e.g. the PyRIT path
// only reports DISPATCH and the outcome), '' = not reached yet.
export type StepClass = 'on' | 'done' | 'skip' | ''

export function fsmSteps(history: string[] | undefined): { state: string; cls: StepClass }[] {
  const last = history && history.length ? history[history.length - 1] : undefined
  const lastIdx = last ? FSM_STATES.indexOf(last as (typeof FSM_STATES)[number]) : -1
  return FSM_STATES.map((state, i) => {
    let cls: StepClass = ''
    if (state === last) cls = 'on'
    else if (history?.includes(state)) cls = 'done'
    else if (lastIdx > i && state !== 'ADAPT') cls = 'skip' // ADAPT and DONE are alternative outcomes, never "skipped"
    return { state, cls }
  })
}

export function pushState(prev: string[] | undefined, state: string): string[] {
  const p = prev ?? []
  return p[p.length - 1] === state ? p : [...p, state]
}
