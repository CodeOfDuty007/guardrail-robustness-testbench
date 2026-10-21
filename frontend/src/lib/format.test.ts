import { describe, expect, it } from 'vitest'
import { cipherName, objectiveName, statusName } from './format'

describe('format helpers', () => {
  it('maps known codes to labels and passes unknown through', () => {
    expect(cipherName('STANDARD_BASE64')).toBe('Base64')
    expect(cipherName('MYSTERY')).toBe('MYSTERY')
    expect(objectiveName('some_new_goal')).toBe('some new goal')
    expect(statusName('success')).toBe('Bypassed')
  })
})
