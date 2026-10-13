import { describe, it, expect } from 'vitest'
import { matchExpected } from './matchExpected'

describe('matchExpected', () => {
  it('correctly matches ACCEPT scenarios', () => {
    expect(matchExpected('ACCEPT', 'ACCEPT')).toBe(true)
    expect(matchExpected('ACCEPT', 'REJECT_SPOOF')).toBe(false)
    expect(matchExpected('ACCEPT', null)).toBe(false)
  })

  it('correctly matches REJECT_ANY against all REJECT_* variants', () => {
    expect(matchExpected('REJECT_ANY', 'REJECT_SPOOF')).toBe(true)
    expect(matchExpected('REJECT_ANY', 'REJECT_NOT_ADDRESSED')).toBe(true)
    expect(matchExpected('REJECT_ANY', 'REJECT_LOW_QUALITY')).toBe(true)
    expect(matchExpected('REJECT_ANY', 'ACCEPT')).toBe(false)
    expect(matchExpected('REJECT_ANY', null)).toBe(false)
  })

  it('correctly matches specific REJECT labels', () => {
    expect(matchExpected('REJECT_SPOOF', 'REJECT_SPOOF')).toBe(true)
    expect(matchExpected('REJECT_SPOOF', 'REJECT_NOT_ADDRESSED')).toBe(false)
    expect(matchExpected('REJECT_NOT_ADDRESSED', 'REJECT_NOT_ADDRESSED')).toBe(true)
    expect(matchExpected('REJECT_AMBIGUOUS', 'REJECT_AMBIGUOUS')).toBe(true)
  })

  it('correctly handles the NONE (silence) scenario', () => {
    expect(matchExpected('NONE', null)).toBe(true)
    expect(matchExpected('NONE', undefined)).toBe(true)
    expect(matchExpected('NONE', 'NONE')).toBe(true)
    expect(matchExpected('NONE', 'IDLE')).toBe(true)
    expect(matchExpected('NONE', 'ACCEPT')).toBe(false)
    expect(matchExpected('NONE', 'REJECT_SPOOF')).toBe(false)
  })
})
