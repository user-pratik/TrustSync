import { describe, it, expect } from 'vitest'
import {
  headline,
  formatPct,
  scenarioRows,
  type ResultsFile,
  type MetricsDict,
} from './results'

function makeMockMetrics(acc: number): MetricsDict {
  return {
    decision_accuracy: acc,
    far_spoof: 0.0,
    far_all_reject: 0.0,
    frr: 0.0,
    liveness: { precision: 1.0, recall: 1.0, f1: 1.0 },
    attribution_accuracy: 1.0,
    addressee_accuracy: 1.0,
    per_scenario: {
      replay_spoof: { n: 10, accuracy: 1.0, accept_rate: 0.0 },
      live_single: { n: 10, accuracy: 0.9, accept_rate: 0.9 },
    },
  }
}

describe('results utilities', () => {
  it('formats percentages accurately (0, 1, 0.9166)', () => {
    expect(formatPct(0)).toBe('0.0%')
    expect(formatPct(1)).toBe('100.0%')
    expect(formatPct(0.9166)).toBe('91.7%')
    expect(formatPct(null)).toBe('--')
    expect(formatPct(NaN)).toBe('--')
  })

  it('selects headline metrics in correct priority order: loso_learned.pooled > loso_rule.pooled > current', () => {
    const mockFile: ResultsFile = {
      generated_at: '2026-10-13T00:00:00Z',
      dataset: { n_clips: 20, per_scenario_counts: {}, subjects: [] },
      config: {},
      current: makeMockMetrics(0.8),
      loso_rule: {
        folds: {},
        pooled: makeMockMetrics(0.9),
      },
      loso_learned: {
        folds: {},
        pooled: makeMockMetrics(0.95),
      },
    }

    // 1. Picks loso_learned.pooled when present
    expect(headline(mockFile).decision_accuracy).toBe(0.95)

    // 2. Picks loso_rule.pooled when loso_learned is missing
    const withoutLearned: ResultsFile = { ...mockFile, loso_learned: undefined }
    expect(headline(withoutLearned).decision_accuracy).toBe(0.9)

    // 3. Picks current when both loso are missing
    const currentOnly: ResultsFile = { ...withoutLearned, loso_rule: undefined }
    expect(headline(currentOnly).decision_accuracy).toBe(0.8)
  })

  it('sorts scenario rows alphabetically by scenario name', () => {
    const metrics = makeMockMetrics(0.9)
    const rows = scenarioRows(metrics)
    expect(rows.length).toBe(2)
    expect(rows[0].scenario).toBe('live_single')
    expect(rows[1].scenario).toBe('replay_spoof')
  })
})
