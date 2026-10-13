import { describe, it, expect } from 'vitest'
import { DECISION_META, toneClasses, type Tone } from './decision'
import type { Decision } from '../types'

describe('decision metadata and tones', () => {
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

  it('provides a DECISION_META entry with non-empty label and an icon for every Decision', () => {
    for (const dec of allDecisions) {
      const meta = DECISION_META[dec]
      expect(meta).toBeDefined()
      expect(typeof meta.label).toBe('string')
      expect(meta.label.length).toBeGreaterThan(0)
      expect(meta.icon).toBeDefined()
      expect(meta.tone).toBeDefined()
      expect(meta.short.length).toBeGreaterThan(0)
    }
  })

  it('toneClasses returns different strings per tone', () => {
    const tones: Tone[] = ['accept', 'reject', 'listen', 'idle']
    const classes = tones.map((t) => toneClasses(t))

    // Ensure all 4 class strings are unique
    const uniqueClasses = new Set(classes)
    expect(uniqueClasses.size).toBe(4)

    for (const cls of classes) {
      expect(typeof cls).toBe('string')
      expect(cls.length).toBeGreaterThan(0)
    }
  })
})
