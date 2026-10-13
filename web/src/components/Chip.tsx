import React from 'react'
import type { LucideIcon } from 'lucide-react'
import type { Tone } from '../lib/decision'
import clsx from 'clsx'

export type ChipTone = Tone | 'info'

export interface ChipProps {
  tone: ChipTone
  children: React.ReactNode
  icon?: LucideIcon
  mono?: boolean
  className?: string
  title?: string
}

const toneMap: Record<ChipTone, string> = {
  accept: 'text-accept bg-accept/12 border-accept/30',
  reject: 'text-reject bg-reject/12 border-reject/30',
  listen: 'text-listen bg-listen/12 border-listen/30',
  idle: 'text-idle bg-idle/12 border-idle/30',
  info: 'text-info bg-info/12 border-info/30',
}

export const Chip: React.FC<ChipProps> = ({
  tone,
  children,
  icon: Icon,
  mono,
  className,
  title,
}) => {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium leading-normal',
        toneMap[tone],
        mono && 'font-mono tabular-nums',
        className,
      )}
    >
      {Icon && <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />}
      <span>{children}</span>
    </span>
  )
}
