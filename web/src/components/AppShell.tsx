import React from 'react'
import { Activity, PlayCircle, BarChart3, Settings, Shield } from 'lucide-react'
import { ConnectionChip, type ConnectionState } from './ConnectionChip'
import clsx from 'clsx'

export interface NavItem {
  id: string
  route: string
  label: string
  icon: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean | 'true' | 'false' }>
}

export const NAV_ITEMS: NavItem[] = [
  { id: 'live', route: '#/live', label: 'Live', icon: Activity },
  { id: 'replay', route: '#/replay', label: 'Replay', icon: PlayCircle },
  { id: 'results', route: '#/results', label: 'Results', icon: BarChart3 },
  { id: 'settings', route: '#/settings', label: 'Settings', icon: Settings },
]

export interface AppShellProps {
  currentRoute: string
  onNavigate: (route: string) => void
  title?: string
  connectionState?: ConnectionState
  topBarActions?: React.ReactNode
  children: React.ReactNode
}

export const AppShell: React.FC<AppShellProps> = ({
  currentRoute,
  onNavigate,
  title,
  connectionState = 'mock',
  topBarActions,
  children,
}) => {
  const activeRoute = currentRoute.startsWith('#/') ? currentRoute : `/${currentRoute}`

  return (
    <div className="min-h-screen bg-bg text-text antialiased flex flex-col">
      {/* Desktop Left Rail (72px fixed) */}
      <nav
        aria-label="Sidebar Navigation"
        className="hidden md:flex fixed left-0 top-0 h-screen w-[72px] flex-col items-center justify-between border-r border-white/[0.08] bg-elev/90 py-6 backdrop-blur-xl z-30"
      >
        <div className="flex flex-col items-center gap-8 w-full">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-accept/10 border border-accept/20 text-accept"
            title="TrustSync"
          >
            <Shield className="h-5 w-5" aria-hidden="true" />
          </div>

          <div className="flex flex-col items-center gap-3 w-full">
            {NAV_ITEMS.map((item) => {
              const isActive =
                activeRoute === item.route ||
                (activeRoute === '#/' && item.route === '#/live')
              const Icon = item.icon
              return (
                <div key={item.id} className="relative w-full flex justify-center group">
                  {isActive && (
                    <span
                      className="absolute left-0 top-1/2 -translate-y-1/2 h-8 w-1 rounded-r bg-accept"
                      aria-hidden="true"
                    />
                  )}
                  <button
                    type="button"
                    onClick={() => onNavigate(item.route)}
                    aria-label={item.label}
                    title={item.label}
                    className={clsx(
                      'relative flex h-12 w-12 items-center justify-center rounded-xl transition-all focus-visible:outline-2 focus-visible:outline-accept',
                      isActive
                        ? 'bg-white/[0.08] text-accept'
                        : 'text-muted hover:bg-white/[0.04] hover:text-text',
                    )}
                  >
                    <Icon className="h-5 w-5" aria-hidden="true" />
                  </button>
                  {/* Tooltip on hover/focus */}
                  <span className="pointer-events-none absolute left-full ml-3 top-1/2 -translate-y-1/2 whitespace-nowrap rounded-md bg-elev px-2 py-1 text-xs font-medium text-text border border-white/[0.08] opacity-0 shadow-lg group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity z-50">
                    {item.label}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Layout Area */}
      <div className="flex flex-col min-h-screen md:pl-[72px] pb-16 md:pb-0">
        {/* Top Bar */}
        <header className="sticky top-0 z-20 flex h-16 w-full items-center justify-between border-b border-white/[0.08] bg-elev/80 px-6 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <div className="md:hidden flex h-8 w-8 items-center justify-center rounded-lg bg-accept/10 text-accept">
              <Shield className="h-4 w-4" aria-hidden="true" />
            </div>
            <div className="flex items-baseline gap-3">
              <span className="text-base font-bold tracking-tight text-text">
                TrustSync
              </span>
              {title && (
                <>
                  <span className="text-muted/40 font-mono">/</span>
                  <span className="text-sm font-medium text-muted">{title}</span>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4">
            {topBarActions}
            <ConnectionChip state={connectionState} />
          </div>
        </header>

        {/* Content Body */}
        <main className="flex-1 p-4 md:p-6 max-w-7xl w-full mx-auto">{children}</main>
      </div>

      {/* Mobile Bottom Tab Bar (<768px) */}
      <nav
        aria-label="Mobile Navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 h-16 border-t border-white/[0.08] bg-elev/95 backdrop-blur-xl flex items-center justify-around z-30 px-2"
      >
        {NAV_ITEMS.map((item) => {
          const isActive =
            activeRoute === item.route ||
            (activeRoute === '#/' && item.route === '#/live')
          const Icon = item.icon
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.route)}
              aria-label={item.label}
              className={clsx(
                'relative flex min-h-[44px] min-w-[44px] flex-col items-center justify-center gap-1 rounded-lg px-3 py-1 transition-colors focus-visible:outline-2 focus-visible:outline-accept',
                isActive ? 'text-accept' : 'text-muted hover:text-text',
              )}
            >
              {isActive && (
                <span
                  className="absolute top-0 h-0.5 w-6 rounded-b bg-accept"
                  aria-hidden="true"
                />
              )}
              <Icon className="h-5 w-5" aria-hidden="true" />
              <span className="text-[10px] font-medium leading-none">{item.label}</span>
            </button>
          )
        })}
      </nav>
    </div>
  )
}
