import React from 'react'
import type { FrameResult } from '../types'
import { Card } from './Card'
import { Gauge } from './Gauge'
import clsx from 'clsx'

export interface TrustGaugesProps {
  frame: FrameResult | null
  config?: Record<string, unknown> | null
  className?: string
}

export const TrustGauges: React.FC<TrustGaugesProps> = ({
  frame,
  config,
  className,
}) => {
  const gateCfg = (config?.gate ?? {}) as Record<string, number>
  const minSync = gateCfg.min_sync ?? 0.35
  const minMargin = gateCfg.min_margin ?? 0.1
  const minDirected = gateCfg.min_directed ?? 0.5

  const decision = frame?.decision ?? 'IDLE'

  return (
    <Card title="Trust Gauges" className={clsx('w-full', className)}>
      <div className="flex items-center justify-around gap-4 py-2 flex-wrap">
        <Gauge
          label="Liveness"
          value={frame?.liveness ?? 0}
          threshold={minSync}
          decision={decision}
        />
        <Gauge
          label="Attribution"
          value={frame?.attribution_conf ?? 0}
          threshold={minMargin}
          decision={decision}
        />
        <Gauge
          label="Addressee"
          value={frame?.addressee ?? 0}
          threshold={minDirected}
          decision={decision}
        />
      </div>
    </Card>
  )
}
