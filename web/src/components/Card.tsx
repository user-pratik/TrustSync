import React from 'react'
import type { LucideIcon } from 'lucide-react'
import type { Tone } from '../lib/decision'
import clsx from 'clsx'

export interface CardProps {
  title?: string
  icon?: LucideIcon
  action?: React.ReactNode
  accent?: Tone
  className?: string
  children?: React.ReactNode
}

export const Card: React.FC<CardProps> = ({
  title,
  icon: Icon,
  action,
  accent,
  className,
  children,
}) => {
  const accentGlowStyle = accent
    ? {
        backgroundImage: `radial-gradient(circle at top left, rgb(var(--${accent}) / 0.06) 0%, transparent 70%)`,
      }
    : undefined

  return (
    <section
      style={accentGlowStyle}
      className={clsx(
        'relative overflow-hidden rounded-card border border-white/[0.08] bg-white/[0.04] backdrop-blur-xl shadow-glass transition-all',
        className,
      )}
    >
      {(title || Icon || action) && (
        <div className="flex h-12 items-center justify-between border-b border-white/[0.04] px-5 py-3">
          <div className="flex items-center gap-2">
            {Icon && <Icon className="h-4 w-4 text-muted" aria-hidden="true" />}
            {title && (
              <h2 className="text-sm font-semibold tracking-wide text-text">
                {title}
              </h2>
            )}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}
