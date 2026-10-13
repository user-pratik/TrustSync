import React from 'react'
import type { FrameResult } from '../types'
import type { HeldDecision } from '../lib/useTrustStream'
import { Card } from './Card'
import { Chip } from './Chip'
import { DECISION_META } from '../lib/decision'
import { fmtScore } from '../lib/format'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import clsx from 'clsx'

export interface DecisionCardProps {
  frame: FrameResult | null
  held: HeldDecision | null
  className?: string
}

export const DecisionCard: React.FC<DecisionCardProps> = ({
  frame,
  held,
  className,
}) => {
  const activeFrame = held ? held.frame : frame
  const decision = activeFrame?.decision ?? 'IDLE'
  const meta = DECISION_META[decision]
  const Icon = meta.icon
  const tone = meta.tone
  const prefersReduced = useReducedMotion()

  const toneBgStyle: Record<string, string> = {
    accept: 'bg-accept/12 text-accept border-accept/30',
    reject: 'bg-reject/12 text-reject border-reject/30',
    listen: 'bg-listen/12 text-listen border-listen/30',
    idle: 'bg-idle/12 text-idle border-idle/30',
  }

  return (
    <Card accent={tone} className={clsx('relative', className)}>
      {/* Screen reader live region */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {meta.label}: {activeFrame?.reason || meta.description}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={decision + (held ? '-held' : '')}
          initial={prefersReduced ? false : { opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={prefersReduced ? undefined : { opacity: 0, scale: 0.97 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="flex flex-col gap-4"
        >
          <div className="flex items-start gap-4">
            {/* 56px rounded square icon */}
            <div
              className={clsx(
                'flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border',
                toneBgStyle[tone],
              )}
            >
              <Icon className="h-7 w-7" aria-hidden="true" />
            </div>

            <div className="flex flex-col min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-[28px] font-semibold tracking-tight text-text leading-none">
                  {meta.label}
                </h2>
                {held && (
                  <span className="rounded bg-info/10 border border-info/30 px-1.5 py-0.5 text-[10px] font-medium text-info uppercase tracking-wider">
                    Held
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-sm text-muted line-clamp-2">
                {activeFrame?.reason || meta.description}
              </p>
            </div>
          </div>

          {/* Speaker Chip & Liveness/Addressee line */}
          <div className="flex items-center justify-between border-t border-white/[0.04] pt-3 flex-wrap gap-2">
            <div>
              {activeFrame?.speaker_id !== null && activeFrame?.speaker_id !== undefined ? (
                <Chip tone={tone}>Speaker {activeFrame.speaker_id}</Chip>
              ) : (
                <span className="text-xs text-muted">No speaker assigned</span>
              )}
            </div>

            <div className="font-mono tabular-nums text-xs text-muted">
              liveness {fmtScore(activeFrame?.liveness ?? 0)} · addressee{' '}
              {fmtScore(activeFrame?.addressee ?? 0)}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </Card>
  )
}
