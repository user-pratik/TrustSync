import { describe, it, expect } from 'vitest'
import { fmtScore, fmtAngle, fmtFps, fmtTime, clamp01 } from './format'

describe('format utilities', () => {
  it('formats score with fixed 2 decimal places and handles null/NaN', () => {
    expect(fmtScore(0.723)).toBe('0.72')
    expect(fmtScore(0.728)).toBe('0.73')
    expect(fmtScore(0)).toBe('0.00')
    expect(fmtScore(1)).toBe('1.00')
    expect(fmtScore(null)).toBe('--')
    expect(fmtScore(NaN)).toBe('--')
  })

  it('formats angles with degree symbol and handles null/NaN', () => {
    expect(fmtAngle(12.4)).toBe('12°')
    expect(fmtAngle(12.8)).toBe('13°')
    expect(fmtAngle(-15.2)).toBe('-15°')
    expect(fmtAngle(null)).toBe('--')
    expect(fmtAngle(NaN)).toBe('--')
  })

  it('formats FPS as integer and handles null/NaN', () => {
    expect(fmtFps(24.2)).toBe('24')
    expect(fmtFps(23.8)).toBe('24')
    expect(fmtFps(null)).toBe('--')
    expect(fmtFps(NaN)).toBe('--')
  })

  it('formats time in mm:ss including 75s as 01:15', () => {
    expect(fmtTime(75)).toBe('01:15')
    expect(fmtTime(0)).toBe('00:00')
    expect(fmtTime(9)).toBe('00:09')
    expect(fmtTime(60)).toBe('01:00')
    expect(fmtTime(3599)).toBe('59:59')
    expect(fmtTime(-5)).toBe('00:00')
    expect(fmtTime(NaN)).toBe('00:00')
  })

  it('clamps values to [0, 1]', () => {
    expect(clamp01(-0.5)).toBe(0)
    expect(clamp01(0.45)).toBe(0.45)
    expect(clamp01(1.5)).toBe(1)
    expect(clamp01(NaN)).toBe(0)
  })
})
