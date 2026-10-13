import React, { useMemo } from 'react'
import type { FrameResult } from '../types'
import { Card } from './Card'
import { Chip } from './Chip'
import { EmptyState } from './EmptyState'
import { fmtScore } from '../lib/format'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
  Tooltip,
} from 'recharts'
import { Users } from 'lucide-react'

export interface SyncTimelineProps {
  history: FrameResult[]
  speakerId: number | null
  config?: Record<string, unknown> | null
  className?: string
}

const PALETTE = ['#A78BFA', '#FBBF24', '#64748B', '#38BDF8']

export const SyncTimeline: React.FC<SyncTimelineProps> = ({
  history,
  speakerId,
  config,
  className,
}) => {
  const gateCfg = (config?.gate ?? {}) as Record<string, number>
  const minSync = gateCfg.min_sync ?? 0.35

  // Identify distinct track IDs across history
  const allTrackIds = useMemo(() => {
    const ids = new Set<number>()
    for (const f of history) {
      for (const face of f.faces) {
        ids.add(face.track_id)
      }
    }
    return Array.from(ids).sort((a, b) => a - b)
  }, [history])

  // Build chart data
  const { chartData, speechIntervals } = useMemo(() => {
    if (history.length === 0) return { chartData: [], speechIntervals: [] }

    const latestT = history[history.length - 1].t
    const data: Array<Record<string, number | boolean | null>> = []
    const intervals: Array<{ start: number; end: number }> = []

    let currentIntervalStart: number | null = null

    for (let i = 0; i < history.length; i++) {
      const f = history[i]
      const relTime = Math.round((f.t - latestT) * 10) / 10
      const point: Record<string, number | boolean | null> = {
        time: relTime,
      }

      for (const tid of allTrackIds) {
        const face = f.faces.find((fc) => fc.track_id === tid)
        point[`track_${tid}`] = face ? face.sync : null
      }

      data.push(point)

      if (f.speech_active) {
        if (currentIntervalStart === null) currentIntervalStart = relTime
      } else {
        if (currentIntervalStart !== null) {
          intervals.push({ start: currentIntervalStart, end: relTime })
          currentIntervalStart = null
        }
      }
    }

    if (currentIntervalStart !== null && data.length > 0) {
      intervals.push({
        start: currentIntervalStart,
        end: data[data.length - 1].time as number,
      })
    }

    return { chartData: data, speechIntervals: intervals }
  }, [history, allTrackIds])

  // Map each track id to stable color
  const trackColors = useMemo(() => {
    const map = new Map<number, string>()
    let otherIdx = 0
    for (const tid of allTrackIds) {
      if (speakerId !== null && tid === speakerId) {
        map.set(tid, '#2DD4BF') // Teal for active speaker
      } else {
        map.set(tid, PALETTE[otherIdx % PALETTE.length])
        otherIdx++
      }
    }
    return map
  }, [allTrackIds, speakerId])

  return (
    <Card
      title="Lip-audio sync"
      action={
        <div className="flex items-center gap-1.5 flex-wrap">
          {allTrackIds.map((tid) => {
            const isSpeaker = speakerId === tid
            return (
              <Chip
                key={tid}
                tone={isSpeaker ? 'accept' : 'idle'}
                mono
                className="text-[10px] py-0 px-2"
              >
                ID {tid}
              </Chip>
            )
          })}
        </div>
      }
      className={className}
    >
      {allTrackIds.length === 0 ? (
        <EmptyState
          icon={Users}
          title="Waiting for faces"
          hint="Tracked faces will display correlation timelines here"
          className="h-[220px]"
        />
      ) : (
        <div className="h-[220px] w-full min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
            >
              {/* Horizontal gridlines only at 6% opacity */}
              <CartesianGrid
                stroke="rgba(255, 255, 255, 0.06)"
                vertical={false}
              />

              <XAxis
                dataKey="time"
                type="number"
                domain={[-10, 0]}
                ticks={[-10, -8, -6, -4, -2, 0]}
                tickFormatter={(v) => `${v}s`}
                stroke="#8B98A9"
                fontSize={11}
                tickLine={false}
              />

              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.5, 1]}
                stroke="#8B98A9"
                fontSize={11}
                tickLine={false}
              />

              {/* Speech period shading */}
              {speechIntervals.map((intv, idx) => (
                <ReferenceArea
                  key={idx}
                  x1={intv.start}
                  x2={intv.end}
                  fill="rgba(255, 255, 255, 0.04)"
                  strokeOpacity={0}
                />
              ))}

              {/* min_sync threshold line */}
              <ReferenceLine
                y={minSync}
                stroke="#2DD4BF"
                strokeDasharray="4 4"
                strokeOpacity={0.6}
                label={{
                  value: `min sync ${fmtScore(minSync)}`,
                  fill: '#8B98A9',
                  fontSize: 10,
                  position: 'insideTopLeft',
                }}
              />

              {/* Dark glass tooltip */}
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="rounded-lg border border-white/[0.08] bg-elev/95 p-2 shadow-glass backdrop-blur-xl text-xs font-mono">
                        {payload.map((entry) => {
                          const tid = String(entry.name).replace('track_', '')
                          return (
                            <div
                              key={entry.name}
                              style={{ color: entry.color }}
                              className="flex items-center gap-2"
                            >
                              <span>ID {tid}</span>
                              <span>·</span>
                              <span>sync {fmtScore(Number(entry.value))}</span>
                            </div>
                          )
                        })}
                      </div>
                    )
                  }
                  return null
                }}
              />

              {allTrackIds.map((tid) => (
                <Line
                  key={tid}
                  type="monotone"
                  dataKey={`track_${tid}`}
                  stroke={trackColors.get(tid) ?? '#64748B'}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}
