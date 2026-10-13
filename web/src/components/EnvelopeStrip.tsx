import React from 'react'
import type { FrameResult } from '../types'

export interface EnvelopeStripProps {
  history: FrameResult[]
  className?: string
}

export const EnvelopeStrip: React.FC<EnvelopeStripProps> = ({
  history,
  className,
}) => {
  const latestFrame = history.length > 0 ? history[history.length - 1] : null
  const speechActive = latestFrame?.speech_active ?? false

  // Take up to 200 values
  const points = history.slice(-200)
  const maxEnv = 0.15
  const width = 100
  const height = 100

  const polylineCoords = points
    .map((f, i) => {
      const x = points.length > 1 ? (i / (points.length - 1)) * width : 0
      const normVal = Math.min(1, Math.max(0, f.env / maxEnv))
      const y = height - normVal * (height - 10) // 10px headroom
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  const strokeColor = speechActive ? 'rgb(var(--listen))' : 'rgb(var(--idle))'

  return (
    <div className={`relative h-full w-full overflow-hidden ${className ?? ''}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-full w-full"
        aria-hidden="true"
      >
        {/* Faint baseline */}
        <line
          x1="0"
          y1={height - 2}
          x2={width}
          y2={height - 2}
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="1"
        />

        {polylineCoords && (
          <polyline
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={polylineCoords}
          />
        )}
      </svg>
    </div>
  )
}
