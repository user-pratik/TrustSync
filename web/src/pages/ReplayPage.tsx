import React, { useState, useEffect, useMemo } from 'react'
import { getSessions, startSource, type SessionInfo } from '../lib/api'
import { useTrustStream } from '../lib/useTrustStream'
import { matchExpected } from '../lib/matchExpected'
import { Card } from '../components/Card'
import { Chip, type ChipTone } from '../components/Chip'
import { EmptyState } from '../components/EmptyState'
import { VideoStage } from '../components/VideoStage'
import { DecisionCard } from '../components/DecisionCard'
import { TrustGauges } from '../components/TrustGauges'
import { Play, CheckCircle2, XCircle, Film } from 'lucide-react'
import clsx from 'clsx'

export const ReplayPage: React.FC = () => {
  const stream = useTrustStream(false)

  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [scenarioFilter, setScenarioFilter] = useState<string>('all')
  const [elapsed, setElapsed] = useState<number>(0)
  const [runStartTime, setRunStartTime] = useState<number | null>(null)

  // Fetch available recorded sessions
  useEffect(() => {
    let active = true
    getSessions().then((res) => {
      if (active && res.ok && res.data.length > 0) {
        setSessions(res.data)
        setSelectedId(res.data[0].id)
      }
    })
    return () => {
      active = false
    }
  }, [])

  const selectedSession = useMemo(
    () => sessions.find((s) => s.id === selectedId),
    [sessions, selectedId],
  )

  const distinctScenarios = useMemo(() => {
    const s = new Set<string>()
    for (const item of sessions) {
      if (item.scenario) s.add(item.scenario)
    }
    return ['all', ...Array.from(s)]
  }, [sessions])

  const filteredSessions = useMemo(() => {
    if (scenarioFilter === 'all') return sessions
    return sessions.filter((s) => s.scenario === scenarioFilter)
  }, [sessions, scenarioFilter])

  // Track elapsed time while running
  useEffect(() => {
    if (stream.status?.running && runStartTime !== null) {
      const interval = setInterval(() => {
        setElapsed((Date.now() - runStartTime) / 1000)
      }, 100)
      return () => clearInterval(interval)
    } else {
      if (!stream.status?.running) {
        setRunStartTime(null)
      }
    }
  }, [stream.status?.running, runStartTime])

  const handleRunClip = async () => {
    if (!selectedId) return
    setRunStartTime(Date.now())
    setElapsed(0)
    await startSource({ type: 'session', id: selectedId })
  }

  // Determine latest utterance decision
  const latestUtterance = stream.events.length > 0 ? stream.events[0] : null
  const utteranceDecision = latestUtterance?.decision ?? null

  const isMatched = selectedSession
    ? matchExpected(selectedSession.expected, utteranceDecision)
    : false

  const getExpectedTone = (expected: string): ChipTone => {
    if (expected === 'ACCEPT') return 'accept'
    if (expected.startsWith('REJECT_')) return 'reject'
    return 'idle'
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left Column: Sessions List (4 cols) */}
      <aside className="col-span-12 lg:col-span-4 flex flex-col gap-4">
        {/* Scenario Filter Chips */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
          {distinctScenarios.map((sc) => (
            <button
              key={sc}
              type="button"
              onClick={() => setScenarioFilter(sc)}
              className={clsx(
                'whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-medium transition-all focus-visible:outline-2 focus-visible:outline-accept',
                scenarioFilter === sc
                  ? 'border-accept/40 bg-accept/12 text-accept'
                  : 'border-white/[0.08] bg-white/[0.04] text-muted hover:text-text',
              )}
            >
              {sc}
            </button>
          ))}
        </div>

        {filteredSessions.length === 0 ? (
          <Card>
            <EmptyState
              icon={Film}
              title="No recorded clips yet"
              hint="Run tools/record_session.py or tools/make_demo_session.py."
            />
          </Card>
        ) : (
          <div className="flex flex-col gap-2.5 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
            {filteredSessions.map((s) => {
              const isSelected = s.id === selectedId
              return (
                <div
                  key={s.id}
                  onClick={() => setSelectedId(s.id)}
                  className={clsx(
                    'cursor-pointer rounded-card border p-3.5 transition-all text-left backdrop-blur-md',
                    isSelected
                      ? 'border-accept/40 bg-white/[0.08] shadow-glass'
                      : 'border-white/[0.08] bg-white/[0.04] hover:bg-white/[0.06]',
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-semibold text-text truncate">
                      {s.id}
                    </span>
                    <Chip tone={getExpectedTone(s.expected)} className="text-[10px] py-0">
                      {s.expected}
                    </Chip>
                  </div>

                  <div className="mt-2 flex items-center justify-between text-xs text-muted">
                    <span className="text-[11px] capitalize">{s.scenario.replace(/_/g, ' ')}</span>
                    <span className="text-[11px]">{s.subjects?.join(', ') || 'No subject'}</span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </aside>

      {/* Right Column: Player & Analysis (8 cols) */}
      <section className="col-span-12 lg:col-span-8 flex flex-col gap-4">
        {/* Result Banner above stage */}
        {selectedSession && latestUtterance && (
          <div
            className={clsx(
              'flex items-center justify-between rounded-xl border p-3 backdrop-blur-xl transition-all',
              isMatched
                ? 'border-accept/30 bg-accept/10 text-accept'
                : 'border-reject/30 bg-reject/10 text-reject',
            )}
          >
            <div className="flex items-center gap-2">
              {isMatched ? (
                <CheckCircle2 className="h-5 w-5 shrink-0" aria-hidden="true" />
              ) : (
                <XCircle className="h-5 w-5 shrink-0" aria-hidden="true" />
              )}
              <span className="text-sm font-semibold">
                {isMatched ? 'Matches expected' : 'Differs from expected'}
              </span>
            </div>
            <span className="font-mono text-xs">
              Expected: {selectedSession.expected} · Got: {utteranceDecision}
            </span>
          </div>
        )}

        {/* Toolbar with Run Clip button and Progress Bar */}
        <div className="flex items-center justify-between gap-4 rounded-xl border border-white/[0.08] bg-white/[0.04] p-3 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={!selectedId || stream.status?.running}
              onClick={handleRunClip}
              aria-label="Run this clip"
              className={clsx(
                'inline-flex items-center gap-2 rounded-lg px-4 py-1.5 text-xs font-semibold shadow-sm transition-all focus-visible:outline-2 focus-visible:outline-accept',
                stream.status?.running
                  ? 'bg-white/[0.06] text-muted cursor-not-allowed'
                  : 'bg-accept text-bg hover:bg-accept/90',
              )}
            >
              <Play className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
              <span>Run this clip</span>
            </button>
            <span className="font-mono text-xs text-muted">
              {selectedId || 'No clip selected'}
            </span>
          </div>

          {/* Progress Bar of clip */}
          <div className="flex items-center gap-2 min-w-[120px]">
            <div className="h-1.5 w-24 rounded-full bg-white/[0.08] overflow-hidden">
              <div
                className="h-full bg-accept transition-all duration-100"
                style={{
                  width: `${Math.min(100, (elapsed / 8.0) * 100)}%`,
                }}
              />
            </div>
            <span className="font-mono text-[11px] tabular-nums text-muted">
              {elapsed.toFixed(1)}s
            </span>
          </div>
        </div>

        {/* Video Stage (live mode connecting to /stream.mjpg) */}
        <VideoStage
          frame={stream.frame}
          mock={false}
          history={stream.history}
        />

        {/* Decision & Gauges Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DecisionCard frame={stream.frame} held={stream.held} />
          <TrustGauges frame={stream.frame} config={stream.config} />
        </div>
      </section>
    </div>
  )
}
