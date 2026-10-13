import React from 'react'
import type { Decision } from '../types'
import { fmtScore, clamp01 } from '../lib/format'
import clsx from 'clsx'

export interface GaugeProps {
  label: string
  value: number
  threshold: number
  decision: Decision
  className?: string
}

export const Gauge: React.FC<GaugeProps> = ({
  label,
  value,
  threshold,
  decision,
  className,
}) => {
  const normVal = clamp01(value)
  const normThresh = clamp01(threshold)

  const radius = 50
  const cx = 60
  const cy = 60
  const arcLength = Math.PI * radius // ~157.08
  const offset = arcLength * (1 - normVal)

  // Determine arc stroke colour
  let strokeColor = 'rgb(var(--idle))'
  if (value >= threshold) {
    strokeColor = 'rgb(var(--accept))'
  } else if (decision.startsWith('REJECT_')) {
    strokeColor = 'rgb(var(--reject))'
  } else if (decision === 'LISTENING') {
    strokeColor = 'rgb(var(--listen))'
  }

  // Threshold tick coordinates
  const theta = Math.PI * (1 - normThresh)
  const tx1 = cx + 42 * Math.cos(theta)
  const ty1 = cy - 42 * Math.sin(theta)
  const tx2 = cx + 58 * Math.cos(theta)
  const ty2 = cy - 58 * Math.sin(theta)

  return (
    <div className={clsx('flex flex-col items-center text-center', className)}>
      <div className="relative h-[72px] w-[120px] flex items-end justify-center">
        <svg
          viewBox="0 0 120 70"
          className="absolute inset-0 h-full w-full overflow-visible"
          aria-hidden="true"
        >
          {/* Background track */}
          <path
            d="M 10,60 A 50,50 0 0,1 110,60"
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="10"
            strokeLinecap="round"
          />

          {/* Active arc */}
          <path
            d="M 10,60 A 50,50 0 0,1 110,60"
            fill="none"
            stroke={strokeColor}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={arcLength}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 300ms ease-out, stroke 300ms ease-out' }}
          />

          {/* Threshold Tick */}
          <line
            x1={tx1}
            y1={ty1}
            x2={tx2}
            y2={ty2}
            stroke="rgba(255, 255, 255, 0.4)"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>

        {/* Value in the middle in 24px mono */}
        <span className="font-mono text-2xl font-bold tabular-nums tracking-tight text-text relative z-10 -mb-1">
          {fmtScore(value)}
        </span>
      </div>

      {/* Label under it */}
      <span className="mt-2 text-xs font-medium text-muted tracking-wide">
        {label}
      </span>
      <span className="text-[10px] font-mono text-muted/60">
        min {fmtScore(threshold)}
      </span>
    </div>
  )
}
