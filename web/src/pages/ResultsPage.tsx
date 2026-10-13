import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { getResults } from '../lib/api'
import { isMockRequested } from '../lib/useTrustStream'
import {
  headline,
  formatPct,
  scenarioRows,
  type ResultsFile,
} from '../lib/results'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Skeleton } from '../components/Skeleton'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { BarChart3, Database } from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import clsx from 'clsx'

export const ResultsPage: React.FC = () => {
  const isMock = isMockRequested()

  const [loading, setLoading] = useState<boolean>(true)
  const [errorStatus, setErrorStatus] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [data, setData] = useState<ResultsFile | null>(null)
  const [isSampleData, setIsSampleData] = useState<boolean>(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    setErrorStatus(null)
    setErrorMessage(null)

    if (isMock) {
      try {
        const res = await fetch('/mock-results.json')
        if (!res.ok) throw new Error('Failed to load sample data')
        const json = await res.json()
        setData(json)
        setIsSampleData(true)
      } catch (err) {
        setErrorStatus(500)
        setErrorMessage(err instanceof Error ? err.message : 'Error')
      } finally {
        setLoading(false)
      }
      return
    }

    const res = await getResults()
    if (res.ok) {
      setData(res.data as unknown as ResultsFile)
      setIsSampleData(false)
    } else {
      setErrorStatus(res.status)
      setErrorMessage(res.message)
    }
    setLoading(false)
  }, [isMock])

  useEffect(() => {
    loadData()
  }, [loadData])

  const hl = useMemo(() => (data ? headline(data) : null), [data])
  const scRows = useMemo(() => (hl ? scenarioRows(hl) : []), [hl])

  if (loading) {
    return (
      <div className="flex flex-col gap-6 animate-pulse">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-card" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-[300px] rounded-card" />
          <Skeleton className="h-[300px] rounded-card" />
        </div>
      </div>
    )
  }

  if (errorStatus === 404) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <EmptyState
          icon={BarChart3}
          title="No evaluation yet"
          hint="Run python -m trustsync.eval.run_eval."
        />
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <ErrorState message={errorMessage} onRetry={loadData} />
      </div>
    )
  }

  if (!data || !hl) return null

  // Confusion matrix preparation
  const confusionData = hl.confusion ?? {}
  const expectedLabels = Object.keys(confusionData)
  const allPreds = new Set<string>()
  expectedLabels.forEach((exp) => {
    Object.keys(confusionData[exp] ?? {}).forEach((p) => allPreds.add(p))
  })
  const predLabels = Array.from(allPreds).sort()

  const maxConfusionCount = Math.max(
    1,
    ...expectedLabels.flatMap((exp) =>
      Object.values(confusionData[exp] ?? {}),
    ),
  )

  return (
    <div className="flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-text">
            Evaluation & Benchmark Results
          </h1>
          <p className="text-xs text-muted mt-0.5">
            Leave-one-subject-out (LOSO) performance and ablation analysis
          </p>
        </div>

        {isSampleData && <Chip tone="info">Sample data</Chip>}
      </div>

      {/* KPI Row (6 Cards) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Decision Accuracy</span>
          <span className="font-mono text-2xl font-bold tabular-nums text-text my-1">
            {formatPct(hl.decision_accuracy)}
          </span>
          <span className="text-[10px] text-muted">Accept / reject agreement</span>
        </Card>

        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Spoof FAR</span>
          <span
            className={clsx(
              'font-mono text-2xl font-bold tabular-nums my-1',
              hl.far_spoof === 0 ? 'text-accept' : 'text-reject',
            )}
          >
            {formatPct(hl.far_spoof)}
          </span>
          <span className="text-[10px] text-muted">Replay false accepts</span>
        </Card>

        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Liveness F1</span>
          <span className="font-mono text-2xl font-bold tabular-nums text-text my-1">
            {formatPct(hl.liveness?.f1)}
          </span>
          <span className="text-[10px] text-muted">Precision & recall</span>
        </Card>

        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Attribution Acc</span>
          <span className="font-mono text-2xl font-bold tabular-nums text-text my-1">
            {formatPct(hl.attribution_accuracy)}
          </span>
          <span className="text-[10px] text-muted">Speaker identification</span>
        </Card>

        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Addressee Acc</span>
          <span className="font-mono text-2xl font-bold tabular-nums text-text my-1">
            {formatPct(hl.addressee_accuracy)}
          </span>
          <span className="text-[10px] text-muted">Looking at robot</span>
        </Card>

        <Card className="flex flex-col justify-between">
          <span className="text-xs font-medium text-muted">Pipeline FPS</span>
          <span className="font-mono text-2xl font-bold tabular-nums text-text my-1">
            {data.fps ? Math.round(data.fps.mean).toString() : '--'}
          </span>
          <span className="text-[10px] text-muted">Real-time throughput</span>
        </Card>
      </div>

      {/* Row 2: Ablation and Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Ablation Card */}
        <Card title="Ablation Analysis">
          <div className="sr-only">
            Ablation analysis comparing models B0 to A5 showing accuracy and false accept rate.
          </div>
          <div className="h-[240px] w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.ablation ?? []}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <CartesianGrid stroke="rgba(255, 255, 255, 0.06)" vertical={false} />
                <XAxis dataKey="name" stroke="#8B98A9" fontSize={12} tickLine={false} />
                <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} stroke="#8B98A9" fontSize={11} tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="rounded-lg border border-white/[0.08] bg-elev/95 p-2.5 text-xs font-mono shadow-glass backdrop-blur-xl">
                          <div className="font-bold text-text mb-1">{payload[0]?.payload?.name}</div>
                          <div className="text-accept">Accuracy: {formatPct(Number(payload[0]?.value))}</div>
                          <div className="text-reject">FAR: {formatPct(Number(payload[1]?.value))}</div>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="accuracy" name="Accuracy" fill="#2DD4BF" radius={[4, 4, 0, 0]} />
                <Bar dataKey="far_all_reject" name="False Accept Rate" fill="#F87171" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-[11px] text-muted leading-relaxed">
            B0 is the naive voice interface (accepts all utterances without visual verification).
          </p>
        </Card>

        {/* Confusion Card */}
        <Card title="Confusion Matrix">
          <div className="sr-only">
            Confusion matrix displaying counts of predicted decisions for each expected label.
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/[0.08] text-[10px] uppercase tracking-wider text-muted font-semibold">
                  <th className="py-2 text-left">Expected</th>
                  {predLabels.map((p) => (
                    <th key={p} className="py-2 text-center">{p}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {expectedLabels.map((exp) => (
                  <tr key={exp}>
                    <td className="py-2.5 font-mono text-[11px] text-muted text-left whitespace-nowrap">
                      {exp}
                    </td>
                    {predLabels.map((p) => {
                      const count = confusionData[exp]?.[p] ?? 0
                      const intensity = count > 0 ? Math.min(1, count / maxConfusionCount) : 0
                      const isDiagonal = exp === p || (exp === 'REJECT_ANY' && p.startsWith('REJECT_'))
                      return (
                        <td
                          key={p}
                          className="py-2.5 text-center font-mono tabular-nums text-xs"
                          style={{
                            backgroundColor:
                              count > 0
                                ? isDiagonal
                                  ? `rgba(45, 212, 191, ${0.1 + intensity * 0.3})`
                                  : `rgba(248, 113, 113, ${0.1 + intensity * 0.3})`
                                : undefined,
                          }}
                        >
                          {count > 0 ? count : '·'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Row 3: Per-scenario and LOSO */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Per-scenario Card */}
        <Card title="Per-Scenario Accuracy">
          <div className="flex flex-col gap-3 py-1">
            {scRows.map((row) => (
              <div key={row.scenario} className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-text capitalize">
                    {row.scenario.replace(/_/g, ' ')}{' '}
                    <span className="text-[10px] text-muted font-mono">(n={row.n})</span>
                  </span>
                  <span className="font-mono tabular-nums text-muted">
                    {formatPct(row.accuracy)}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-white/[0.08] overflow-hidden">
                  <div
                    className="h-full bg-accept transition-all duration-300"
                    style={{ width: `${row.accuracy * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Leave-One-Subject-Out Card */}
        <Card title="Leave-One-Subject-Out (LOSO)">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-white/[0.08] text-[10px] uppercase tracking-wider text-muted font-semibold">
                <tr>
                  <th className="py-2">Held-Out Fold</th>
                  <th className="py-2 text-right">Accuracy</th>
                  <th className="py-2 text-right">FAR</th>
                  <th className="py-2 text-right">Liveness F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {Object.entries(data.loso_learned?.folds ?? data.loso_rule?.folds ?? {}).map(
                  ([sub, m]) => (
                    <tr key={sub}>
                      <td className="py-2.5 font-mono text-muted">{sub}</td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-text">
                        {formatPct(m.decision_accuracy)}
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-text">
                        {formatPct(m.far_all_reject)}
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-text">
                        {formatPct(m.liveness?.f1)}
                      </td>
                    </tr>
                  ),
                )}
                {/* Pooled Row */}
                <tr className="font-semibold bg-white/[0.02]">
                  <td className="py-2.5 font-mono text-accept">Pooled LOSO</td>
                  <td className="py-2.5 text-right font-mono tabular-nums text-accept">
                    {formatPct(hl.decision_accuracy)}
                  </td>
                  <td className="py-2.5 text-right font-mono tabular-nums text-accept">
                    {formatPct(hl.far_all_reject)}
                  </td>
                  <td className="py-2.5 text-right font-mono tabular-nums text-accept">
                    {formatPct(hl.liveness?.f1)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Row 4: Dataset & Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset Card */}
        <Card title="Dataset" icon={Database}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted">Total Recorded Sessions</span>
              <span className="font-mono font-bold text-text">{data.dataset?.n_clips ?? 0}</span>
            </div>

            <div className="flex flex-wrap gap-1.5 pt-1">
              {Object.entries(data.dataset?.per_scenario_counts ?? {}).map(([sc, count]) => (
                <Chip key={sc} tone="idle" className="text-[10px]">
                  {sc}: {count}
                </Chip>
              ))}
            </div>

            <div className="text-xs text-muted pt-2 border-t border-white/[0.04]">
              Subjects: {data.dataset?.subjects?.join(', ') || 'None'}
            </div>
          </div>
        </Card>

        {/* Notes Card */}
        <Card title="Metadata & Latency">
          <div className="flex flex-col gap-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted">Generated At</span>
              <span className="font-mono text-text">{data.generated_at}</span>
            </div>

            {data.av_lag && (
              <div className="flex items-center justify-between">
                <span className="text-muted">A/V Lag Median</span>
                <span className="font-mono text-text">
                  {data.av_lag.median !== null ? `${(data.av_lag.median * 1000).toFixed(1)} ms` : '--'}
                  {' '}(sat: {(data.av_lag.saturated_fraction * 100).toFixed(0)}%)
                </span>
              </div>
            )}

            <div className="pt-2 border-t border-white/[0.04] text-[11px] text-muted">
              {data.notes?.map((n, i) => (
                <p key={i}>• {n}</p>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
