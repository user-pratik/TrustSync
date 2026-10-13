import React from 'react'
import type { FrameResult } from '../types'
import { DECISION_META, toneClasses } from '../lib/decision'
import { fmtScore, fmtFps } from '../lib/format'
import { EnvelopeStrip } from './EnvelopeStrip'
import clsx from 'clsx'

export interface VideoStageProps {
  frame: FrameResult | null
  mock?: boolean
  mirror?: boolean
  history?: FrameResult[]
  className?: string
}

export const VideoStage: React.FC<VideoStageProps> = ({
  frame,
  mock = true,
  mirror = false,
  history = [],
  className,
}) => {
  const decision = frame?.decision ?? 'IDLE'
  const meta = DECISION_META[decision]
  const DecisionIcon = meta.icon

  const faces = frame?.faces ?? []
  const speakerId = frame?.speaker_id

  const [streamError, setStreamError] = React.useState(false)
  const [streamKey, setStreamKey] = React.useState(Date.now())

  return (
    <div
      className={clsx(
        'relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-white/[0.08] bg-elev shadow-glass select-none',
        className,
      )}
    >
      {/* Background Video or Mock scene */}
      {mock ? (
        <div className="absolute inset-0 bg-gradient-to-b from-[#0A0E14] to-[#121926]">
          {/* Soft blurred circle heads at face bboxes */}
          <svg className="h-full w-full" aria-hidden="true">
            <defs>
              <filter id="head-blur" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="24" />
              </filter>
            </defs>
            {faces.map((face) => {
              const [x1, y1, x2, y2] = face.bbox
              const rawCx = (x1 + x2) / 2
              const cx = mirror ? 1 - rawCx : rawCx
              const cy = (y1 + y2) / 2
              const r = ((x2 - x1) + (y2 - y1)) / 4
              return (
                <circle
                  key={face.track_id}
                  cx={`${cx * 100}%`}
                  cy={`${cy * 100}%`}
                  r={`${r * 100}%`}
                  fill="rgba(255, 255, 255, 0.08)"
                  filter="url(#head-blur)"
                />
              )
            })}
          </svg>
        </div>
      ) : streamError ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-elev/90 p-4 z-10">
          <p className="text-sm font-semibold text-text">Stream unavailable</p>
          <button
            type="button"
            onClick={() => {
              setStreamError(false)
              setStreamKey(Date.now())
            }}
            className="mt-3 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3.5 py-1.5 text-xs font-medium text-text hover:bg-white/[0.08] focus-visible:outline-2 focus-visible:outline-accept transition-colors"
          >
            Retry
          </button>
        </div>
      ) : (
        <img
          src={`/stream.mjpg?t=${streamKey}`}
          alt="Camera stream"
          onError={() => setStreamError(true)}
          className={clsx(
            'h-full w-full object-cover',
            mirror && '-scale-x-100',
          )}
        />
      )}

      {/* SVG Corner-Bracket Overlays */}
      <svg
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        {faces.map((face) => {
          let [x1, y1, x2, y2] = face.bbox
          if (mirror) {
            const tempX1 = 1 - x2
            const tempX2 = 1 - x1
            x1 = tempX1
            x2 = tempX2
          }

          const isSpeaker = speakerId !== null && speakerId === face.track_id
          const isReject = decision.startsWith('REJECT_')
          // Check if this face is highest sync
          const isHighestSync =
            faces.length > 0 &&
            face.sync === Math.max(...faces.map((f) => f.sync))
          const isRejectedBestFace = isReject && isHighestSync

          let strokeColor = 'rgb(var(--idle))'
          if (isSpeaker) {
            strokeColor = 'rgb(var(--accept))'
          } else if (isRejectedBestFace) {
            strokeColor = 'rgb(var(--reject))'
          }

          const w = x2 - x1
          const h = y2 - y1
          const armX = w * 0.15
          const armY = h * 0.15

          return (
            <g key={face.track_id} style={{ transition: 'all 120ms ease-out' }}>
              {/* Top-Left */}
              <polyline
                points={`${x1 + armX},${y1} ${x1},${y1} ${x1},${y1 + armY}`}
                fill="none"
                stroke={strokeColor}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              {/* Top-Right */}
              <polyline
                points={`${x2 - armX},${y1} ${x2},${y1} ${x2},${y1 + armY}`}
                fill="none"
                stroke={strokeColor}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              {/* Bottom-Left */}
              <polyline
                points={`${x1 + armX},${y2} ${x1},${y2} ${x1},${y2 - armY}`}
                fill="none"
                stroke={strokeColor}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              {/* Bottom-Right */}
              <polyline
                points={`${x2 - armX},${y2} ${x2},${y2} ${x2},${y2 - armY}`}
                fill="none"
                stroke={strokeColor}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )
        })}
      </svg>

      {/* HTML Absolute Overlays per Face */}
      {faces.map((face) => {
        let [x1, y1, x2, y2] = face.bbox
        if (mirror) {
          const tempX1 = 1 - x2
          const tempX2 = 1 - x1
          x1 = tempX1
          x2 = tempX2
        }

        const leftPct = x1 * 100
        const topPct = y1 * 100
        const widthPct = (x2 - x1) * 100
        const heightPct = (y2 - y1) * 100

        const isSpeaker = speakerId !== null && speakerId === face.track_id
        const isReject = decision.startsWith('REJECT_')
        const isHighestSync =
          faces.length > 0 &&
          face.sync === Math.max(...faces.map((f) => f.sync))
        const isRejectedBestFace = isReject && isHighestSync

        let barColor = 'bg-idle'
        if (isSpeaker) barColor = 'bg-accept'
        else if (isRejectedBestFace) barColor = 'bg-reject'

        // Gaze Arrow calculation
        const yaw = face.yaw ?? 0
        const pitch = face.pitch ?? 0
        const dx = Math.max(-100, Math.min(100, (mirror ? -yaw : yaw) * 3))
        const dy = Math.max(-100, Math.min(100, pitch * 3))

        return (
          <div
            key={face.track_id}
            className="pointer-events-none absolute"
            style={{
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
              transition: 'all 120ms ease-out',
            }}
          >
            {/* ID and Sync Chip */}
            <div className="absolute -top-7 left-0 whitespace-nowrap rounded bg-elev/90 px-2 py-0.5 text-[11px] font-mono tabular-nums text-text border border-white/[0.08] shadow-md backdrop-blur-md">
              ID {face.track_id} · sync {fmtScore(face.sync)}
            </div>

            {/* Sync Bar under box */}
            <div className="absolute -bottom-2 left-0 h-1 w-full overflow-hidden rounded-full bg-white/[0.1]">
              <div
                className={clsx('h-full transition-all duration-120', barColor)}
                style={{ width: `${Math.min(100, Math.max(0, face.sync * 100))}%` }}
              />
            </div>

            {/* Gaze Arrow from Center */}
            <svg
              className="absolute inset-0 overflow-visible"
              style={{ left: '50%', top: '50%' }}
              aria-hidden="true"
            >
              <line
                x1="0"
                y1="0"
                x2={dx}
                y2={dy}
                stroke={isSpeaker ? '#2DD4BF' : '#8B98A9'}
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle
                cx={dx}
                cy={dy}
                r="3"
                fill={isSpeaker ? '#2DD4BF' : '#8B98A9'}
              />
            </svg>
          </div>
        )
      })}

      {/* Top-Left Pill: Decision Status */}
      <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full border border-white/[0.08] bg-elev/80 px-3.5 py-1.5 backdrop-blur-xl shadow-lg">
        <DecisionIcon className={clsx('h-4 w-4', toneClasses(meta.tone).split(' ')[0])} aria-hidden="true" />
        <span className="text-xs font-semibold tracking-wide text-text">
          {meta.label}
        </span>
      </div>

      {/* Top-Right Pill: FPS */}
      <div className="absolute right-4 top-4 rounded-full border border-white/[0.08] bg-elev/80 px-3 py-1.5 text-xs font-mono tabular-nums text-muted backdrop-blur-xl shadow-lg">
        FPS {fmtFps(frame?.fps ?? 24)}
      </div>

      {/* Bottom 48px Glass Strip with Envelope */}
      <div className="absolute bottom-0 left-0 right-0 h-12 border-t border-white/[0.08] bg-elev/75 px-4 py-1 backdrop-blur-xl">
        <EnvelopeStrip history={history} />
      </div>
    </div>
  )
}
