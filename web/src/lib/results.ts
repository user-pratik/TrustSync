export interface MetricsDict {
  n_clips?: number
  decision_accuracy: number
  far_spoof: number
  far_all_reject: number
  frr: number
  false_trigger_rate?: number
  missed_utterance_rate?: number
  liveness: { precision: number; recall: number; f1: number }
  attribution_accuracy: number
  addressee_accuracy: number
  reason_accuracy?: number
  per_scenario?: Record<
    string,
    { n: number; accuracy: number; accept_rate: number }
  >
  confusion?: Record<string, Record<string, number>>
}

export interface AblationRow {
  name: string
  accuracy: number
  far_all_reject: number
  frr: number
  liveness_f1: number
}

export interface ResultsFile {
  generated_at: string
  dataset: {
    n_clips: number
    per_scenario_counts: Record<string, number>
    subjects: string[]
  }
  config: Record<string, unknown>
  current: MetricsDict
  calibrated?: MetricsDict
  loso_rule?: {
    folds: Record<string, MetricsDict>
    pooled: MetricsDict
  }
  loso_learned?: {
    folds: Record<string, MetricsDict>
    pooled: MetricsDict
  }
  learned_kind?: string
  ablation?: AblationRow[]
  av_lag?: {
    median: number | null
    p10: number | null
    p90: number | null
    n: number
    saturated_fraction: number
  }
  fps?: {
    mean: number
    p5: number
    source: string
    machine: string
  } | null
  figures?: Array<{ name: string; path: string; caption: string }>
  notes?: string[]
}

export function headline(results: ResultsFile): MetricsDict {
  if (results.loso_learned?.pooled) {
    return results.loso_learned.pooled
  }
  if (results.loso_rule?.pooled) {
    return results.loso_rule.pooled
  }
  return results.current
}

export function formatPct(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) {
    return '--'
  }
  return `${(x * 100).toFixed(1)}%`
}

export interface ScenarioRow {
  scenario: string
  n: number
  accuracy: number
  accept_rate: number
}

export function scenarioRows(metrics: MetricsDict): ScenarioRow[] {
  if (!metrics.per_scenario) return []

  const rows: ScenarioRow[] = Object.entries(metrics.per_scenario).map(
    ([scenario, data]) => ({
      scenario,
      n: data.n,
      accuracy: data.accuracy,
      accept_rate: data.accept_rate,
    }),
  )

  return rows.sort((a, b) => a.scenario.localeCompare(b.scenario))
}
