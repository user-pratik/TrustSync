import React, { useState } from 'react'
import type { TrustStreamState } from '../lib/useTrustStream'
import { VideoStage } from '../components/VideoStage'
import { DecisionCard } from '../components/DecisionCard'
import { TrustGauges } from '../components/TrustGauges'
import { SyncTimeline } from '../components/SyncTimeline'
import { EventLog } from '../components/EventLog'
import { Skeleton } from '../components/Skeleton'
import { ErrorState } from '../components/ErrorState'
import { FlipHorizontal, Eye, EyeOff } from 'lucide-react'
import clsx from 'clsx'

export interface LivePageProps {
  stream: TrustStreamState
  isConnecting?: boolean
}

export const LivePage: React.FC<LivePageProps> = ({
  stream,
  isConnecting = false,
}) => {
  const [mirror, setMirror] = useState<boolean>(false)
  const [showOverlay, setShowOverlay] = useState<boolean>(true)

  if (stream.state === 'disconnected') {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <ErrorState
          message="Lost connection to robot perception stream"
          onRetry={() => window.location.reload()}
        />
      </div>
    )
  }

  if (isConnecting || stream.state === 'connecting') {
    return (
      <div className="grid grid-cols-12 gap-6 animate-pulse">
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-4">
          <Skeleton className="h-9 w-48" />
          <Skeleton className="aspect-[4/3] w-full rounded-2xl" />
          <Skeleton className="h-[220px] w-full rounded-card mt-2" />
        </div>
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          <Skeleton className="h-[180px] w-full rounded-card" />
          <Skeleton className="h-[200px] w-full rounded-card" />
          <Skeleton className="h-[280px] w-full rounded-card" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      {/* 12-column responsive layout */}
      <div className="grid grid-cols-12 gap-6">
        {/* Top-Left / Video Section (8 cols on lg, order-1 on mobile) */}
        <section className="col-span-12 lg:col-span-8 flex flex-col gap-3 order-1">
          {/* Small Toolbar above VideoStage */}
          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-semibold tracking-wider uppercase text-muted">
              Live Feed
            </span>

            <div className="flex items-center gap-2">
              {/* Mirror toggle */}
              <button
                type="button"
                onClick={() => setMirror((prev) => !prev)}
                aria-pressed={mirror}
                aria-label="Mirror video feed"
                className={clsx(
                  'inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-accept',
                  mirror
                    ? 'border-accept/40 bg-accept/10 text-accept'
                    : 'border-white/[0.08] bg-white/[0.04] text-muted hover:text-text',
                )}
              >
                <FlipHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Mirror</span>
              </button>

              {/* Segmented control: Overlay on/off */}
              <div className="inline-flex rounded-lg border border-white/[0.08] bg-white/[0.04] p-0.5 text-xs">
                <button
                  type="button"
                  onClick={() => setShowOverlay(true)}
                  aria-pressed={showOverlay}
                  className={clsx(
                    'inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-medium transition-all focus-visible:outline-2 focus-visible:outline-accept',
                    showOverlay
                      ? 'bg-elev text-text shadow-sm'
                      : 'text-muted hover:text-text',
                  )}
                >
                  <Eye className="h-3 w-3" aria-hidden="true" />
                  <span>Overlay</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowOverlay(false)}
                  aria-pressed={!showOverlay}
                  className={clsx(
                    'inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-medium transition-all focus-visible:outline-2 focus-visible:outline-accept',
                    !showOverlay
                      ? 'bg-elev text-text shadow-sm'
                      : 'text-muted hover:text-text',
                  )}
                >
                  <EyeOff className="h-3 w-3" aria-hidden="true" />
                  <span>Off</span>
                </button>
              </div>
            </div>
          </div>

          <VideoStage
            frame={showOverlay ? stream.frame : null}
            mock={stream.state === 'mock'}
            mirror={mirror}
            history={stream.history}
          />
        </section>

        {/* Top-Right / Decision Card & Trust Gauges (4 cols on lg, order-2 & order-3 on mobile) */}
        <section className="col-span-12 lg:col-span-4 flex flex-col gap-6 order-2 lg:order-2">
          <DecisionCard
            frame={stream.frame}
            held={stream.held}
            className="order-2"
          />
          <TrustGauges
            frame={stream.frame}
            config={stream.config}
            className="order-3"
          />
        </section>

        {/* Bottom-Left / Sync Timeline (8 cols on lg, order-4 on mobile) */}
        <section className="col-span-12 lg:col-span-8 order-4">
          <SyncTimeline
            history={stream.history}
            speakerId={stream.frame?.speaker_id ?? null}
            config={stream.config}
          />
        </section>

        {/* Bottom-Right / Event Log (4 cols on lg, order-5 on mobile) */}
        <section className="col-span-12 lg:col-span-4 order-5">
          <EventLog events={stream.events} />
        </section>
      </div>
    </div>
  )
}
