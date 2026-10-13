import React, { useState, useEffect } from 'react'
import type { TrustStreamState } from '../lib/useTrustStream'
import { getSessions, startSource, stopSource, type SessionInfo } from '../lib/api'
import { Chip } from './Chip'
import { Play, Square, Loader2, Video, Film } from 'lucide-react'
import clsx from 'clsx'

export interface SourceSelectorProps {
  stream: TrustStreamState
  className?: string
}

export const SourceSelector: React.FC<SourceSelectorProps> = ({
  stream,
  className,
}) => {
  const isMock = stream.state === 'mock'
  const isRunning = stream.status?.running ?? false
  const statusError = stream.status?.error

  const [sourceType, setSourceType] = useState<'webcam' | 'session'>('webcam')
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string>('')
  const [isPending, setIsPending] = useState<boolean>(false)

  // Fetch available sessions
  useEffect(() => {
    if (isMock) return
    let active = true
    getSessions().then((res) => {
      if (active && res.ok && res.data.length > 0) {
        setSessions(res.data)
        setSelectedSessionId(res.data[0].id)
      }
    })
    return () => {
      active = false
    }
  }, [isMock])

  // Clear pending state when running state updates
  useEffect(() => {
    setIsPending(false)
  }, [isRunning])

  const handleToggleSource = async () => {
    if (isMock || isPending) return

    setIsPending(true)
    if (isRunning) {
      await stopSource()
    } else {
      if (sourceType === 'webcam') {
        await startSource({ type: 'webcam' })
      } else if (selectedSessionId) {
        await startSource({ type: 'session', id: selectedSessionId })
      }
    }
  }

  return (
    <div className={clsx('flex items-center gap-3', className)}>
      {statusError && (
        <Chip tone="reject" className="max-w-xs truncate" title={statusError}>
          {statusError}
        </Chip>
      )}

      {/* Segmented Control */}
      <div
        className={clsx(
          'inline-flex rounded-lg border border-white/[0.08] bg-white/[0.04] p-0.5 text-xs',
          isMock && 'opacity-50 cursor-not-allowed',
        )}
        title={isMock ? 'Disabled in mock mode' : undefined}
      >
        <button
          type="button"
          disabled={isMock || isRunning}
          onClick={() => setSourceType('webcam')}
          aria-pressed={sourceType === 'webcam'}
          className={clsx(
            'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition-all focus-visible:outline-2 focus-visible:outline-accept',
            sourceType === 'webcam'
              ? 'bg-elev text-text shadow-sm'
              : 'text-muted hover:text-text',
          )}
        >
          <Video className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Webcam</span>
        </button>

        <button
          type="button"
          disabled={isMock || isRunning}
          onClick={() => setSourceType('session')}
          aria-pressed={sourceType === 'session'}
          className={clsx(
            'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition-all focus-visible:outline-2 focus-visible:outline-accept',
            sourceType === 'session'
              ? 'bg-elev text-text shadow-sm'
              : 'text-muted hover:text-text',
          )}
        >
          <Film className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Recorded clip</span>
        </button>
      </div>

      {/* Session Select (if session mode) */}
      {sourceType === 'session' && (
        <select
          disabled={isMock || isRunning}
          value={selectedSessionId}
          onChange={(e) => setSelectedSessionId(e.target.value)}
          aria-label="Select recorded session"
          title={isMock ? 'Disabled in mock mode' : undefined}
          className={clsx(
            'rounded-lg border border-white/[0.08] bg-elev px-2.5 py-1 text-xs text-text focus-visible:outline-2 focus-visible:outline-accept max-w-[200px] truncate',
            isMock && 'opacity-50 cursor-not-allowed',
          )}
        >
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} · expects {s.expected}
            </option>
          ))}
        </select>
      )}

      {/* Start / Stop Button */}
      <button
        type="button"
        disabled={isMock || isPending}
        onClick={handleToggleSource}
        aria-label={isRunning ? 'Stop source' : 'Start source'}
        title={isMock ? 'Disabled in mock mode' : isRunning ? 'Stop source' : 'Start source'}
        className={clsx(
          'inline-flex items-center justify-center rounded-lg p-1.5 transition-all focus-visible:outline-2 focus-visible:outline-accept',
          isMock
            ? 'opacity-50 cursor-not-allowed bg-white/[0.04] text-muted'
            : isRunning
              ? 'bg-reject/12 text-reject hover:bg-reject/20 border border-reject/30'
              : 'bg-accept/12 text-accept hover:bg-accept/20 border border-accept/30',
        )}
      >
        {isPending ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        ) : isRunning ? (
          <Square className="h-4 w-4 fill-current" aria-hidden="true" />
        ) : (
          <Play className="h-4 w-4 fill-current" aria-hidden="true" />
        )}
      </button>
    </div>
  )
}
