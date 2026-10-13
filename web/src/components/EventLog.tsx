import React, { useMemo } from 'react'
import type { FrameResult } from '../types'
import { Card } from './Card'
import { Chip } from './Chip'
import { EmptyState } from './EmptyState'
import { DECISION_META } from '../lib/decision'
import { fmtScore, fmtTime } from '../lib/format'
import { ListFilter } from 'lucide-react'

export interface EventLogProps {
  events: FrameResult[]
  className?: string
}

export const EventLog: React.FC<EventLogProps> = ({ events, className }) => {
  // First event timestamp (the oldest event in history)
  const firstEventT = useMemo(() => {
    if (events.length === 0) return 0
    return events[events.length - 1].t
  }, [events])

  return (
    <Card title="Decisions" className={className}>
      {events.length === 0 ? (
        <EmptyState
          icon={ListFilter}
          title="No utterances yet"
          hint="Speak a command to produce decisions"
          className="h-[220px]"
        />
      ) : (
        <div className="max-h-[320px] overflow-y-auto pr-1">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-elev text-[11px] font-semibold uppercase tracking-wider text-muted border-b border-white/[0.08]">
              <tr>
                <th className="pb-2">Time</th>
                <th className="pb-2">Decision</th>
                <th className="pb-2">Speaker</th>
                <th className="pb-2 text-right">Liveness</th>
                <th className="pb-2 text-right">Addressee</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {events.slice(0, 50).map((evt, idx) => {
                const meta = DECISION_META[evt.decision]
                const relTime = Math.max(0, evt.t - firstEventT)
                return (
                  <tr
                    key={`${evt.frame_idx}-${idx}`}
                    className="transition-opacity duration-150 ease-out hover:bg-white/[0.02]"
                  >
                    <td colSpan={5} className="py-2.5">
                      {/* Primary Row */}
                      <div className="grid grid-cols-5 items-center gap-2">
                        <span className="font-mono tabular-nums text-muted">
                          {fmtTime(relTime)}
                        </span>
                        <div>
                          <Chip tone={meta.tone} className="text-[11px] py-0">
                            {meta.short}
                          </Chip>
                        </div>
                        <span className="font-mono text-muted">
                          {evt.speaker_id !== null ? `ID ${evt.speaker_id}` : '-'}
                        </span>
                        <span className="font-mono tabular-nums text-right text-text">
                          {fmtScore(evt.liveness)}
                        </span>
                        <span className="font-mono tabular-nums text-right text-text">
                          {fmtScore(evt.addressee)}
                        </span>
                      </div>

                      {/* Second Muted Line with Reason */}
                      <div
                        title={evt.reason}
                        className="mt-1 line-clamp-2 text-[11px] text-muted/80 leading-relaxed pl-1"
                      >
                        {evt.reason}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
