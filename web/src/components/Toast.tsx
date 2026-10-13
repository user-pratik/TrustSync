import React, { createContext, useContext, useState, useCallback } from 'react'
import type { Tone } from '../lib/decision'
import { toneClasses } from '../lib/decision'
import clsx from 'clsx'

export interface ToastItem {
  id: string
  message: string
  tone: Tone
}

export interface ToastContextType {
  toast: (message: string, tone?: Tone) => void
}

const ToastContext = createContext<ToastContextType | null>(null)

export const useToast = (): ToastContextType => {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return ctx
}

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback(
    (message: string, tone: Tone = 'accept') => {
      const id = Math.random().toString(36).substring(2, 9)
      const newItem: ToastItem = { id, message, tone }
      setToasts((prev) => [...prev, newItem])

      setTimeout(() => {
        removeToast(id)
      }, 4000)
    },
    [removeToast],
  )

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {/* Bottom-right toast container */}
      <aside
        aria-live="polite"
        aria-atomic="false"
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className={clsx(
              'pointer-events-auto flex items-center justify-between rounded-xl border border-white/[0.08] bg-elev/95 p-3.5 shadow-glass backdrop-blur-xl transition-all',
              toneClasses(t.tone).split(' ')[0],
            )}
          >
            <span className="text-xs font-medium text-text">{t.message}</span>
            <button
              type="button"
              onClick={() => removeToast(t.id)}
              aria-label="Dismiss toast"
              className="ml-3 text-muted hover:text-text focus-visible:outline-2 focus-visible:outline-accept rounded"
            >
              ×
            </button>
          </div>
        ))}
      </aside>
    </ToastContext.Provider>
  )
}
