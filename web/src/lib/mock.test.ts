import { describe, it, expect } from 'vitest'
import { mockFrame } from './mock'
import type { Decision } from '../types'

describe('mockFrame generator', () => {
  it('satisfies cycle, final frame, and bounds constraints over k in 0..979', () => {
    const finalDecisions: Decision[] = []
    const seenDecisions = new Set<Decision>()

    for (let k = 0; k < 980; k++) {
      const f = mockFrame(k)
      seenDecisions.add(f.decision)

      if (f.utterance_final) {
        finalDecisions.push(f.decision)
      }

      // Every score field in [0, 1]
      expect(f.liveness).toBeGreaterThanOrEqual(0)
      expect(f.liveness).toBeLessThanOrEqual(1)
      expect(f.attribution_conf).toBeGreaterThanOrEqual(0)
      expect(f.attribution_conf).toBeLessThanOrEqual(1)
      expect(f.addressee).toBeGreaterThanOrEqual(0)
      expect(f.addressee).toBeLessThanOrEqual(1)

      // Bbox checks
      for (const face of f.faces) {
        const [x1, y1, x2, y2] = face.bbox
        expect(x1).toBeGreaterThanOrEqual(0)
        expect(y1).toBeGreaterThanOrEqual(0)
        expect(x2).toBeLessThanOrEqual(1)
        expect(y2).toBeLessThanOrEqual(1)
        expect(x1).toBeLessThan(x2)
        expect(y1).toBeLessThan(y2)
      }
    }

    // Exactly 7 frames have utterance_final
    expect(finalDecisions.length).toBe(7)
    expect(finalDecisions).toEqual([
      'ACCEPT',
      'REJECT_SPOOF',
      'ACCEPT',
      'REJECT_NOT_ADDRESSED',
      'REJECT_NO_FACE',
      'REJECT_AMBIGUOUS',
      'REJECT_LOW_QUALITY',
    ])

    // All 8 Decision values appear at least once
    const allDecisions: Decision[] = [
      'IDLE',
      'LISTENING',
      'ACCEPT',
      'REJECT_SPOOF',
      'REJECT_AMBIGUOUS',
      'REJECT_NOT_ADDRESSED',
      'REJECT_LOW_QUALITY',
      'REJECT_NO_FACE',
    ]
    for (const d of allDecisions) {
      expect(seenDecisions.has(d)).toBe(true)
    }

    // Deterministic
    expect(mockFrame(42)).toEqual(mockFrame(42))

    // Cycle behavior: mockFrame(980) has same values as mockFrame(0) (except t and frame_idx)
    const f0 = mockFrame(0)
    const f980 = mockFrame(980)
    expect(f980.decision).toEqual(f0.decision)
    expect(f980.speech_active).toEqual(f0.speech_active)
    expect(f980.liveness).toEqual(f0.liveness)
    expect(f980.attribution_conf).toEqual(f0.attribution_conf)
    expect(f980.addressee).toEqual(f0.addressee)
    expect(f980.faces.length).toEqual(f0.faces.length)
  })
})
