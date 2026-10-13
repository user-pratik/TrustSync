import React from 'react'
import clsx from 'clsx'

export type ConnectionState = 'connected' | 'mock' | 'connecting' | 'disconnected'

export interface ConnectionChipProps {
  state: ConnectionState
  className?: string
}

const stateConfig: Record<
  ConnectionState,
  { label: string; textClass: string; bgClass: string; dotClass: string }
> = {
  connected: {
    label: 'Connected',
    textClass: 'text-accept',
    bgClass: 'bg-accept/10 border-accept/30',
    dotClass: 'bg-accept',
  },
  mock: {
    label: 'Mock',
    textClass: 'text-info',
    bgClass: 'bg-info/10 border-info/30',
    dotClass: 'bg-info',
  },
  connecting: {
    label: 'Connecting',
    textClass: 'text-listen',
    bgClass: 'bg-listen/10 border-listen/30',
    dotClass: 'bg-listen animate-pulse',
  },
  disconnected: {
    label: 'Disconnected',
    textClass: 'text-reject',
    bgClass: 'bg-reject/10 border-reject/30',
    dotClass: 'bg-reject',
  },
}

export const ConnectionChip: React.FC<ConnectionChipProps> = ({
  state,
  className,
}) => {
  const conf = stateConfig[state]

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium backdrop-blur-md',
        conf.bgClass,
        conf.textClass,
        className,
      )}
    >
      <span
        className={clsx('h-2 w-2 rounded-full shrink-0', conf.dotClass)}
        aria-hidden="true"
      />
      <span>{conf.label}</span>
    </div>
  )
}
