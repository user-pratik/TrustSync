import React, { useState, useEffect } from 'react'
import { AppShell } from './components/AppShell'
import { ToastProvider } from './components/Toast'
import { SourceSelector } from './components/SourceSelector'
import { LivePage } from './pages/LivePage'
import { ReplayPage } from './pages/ReplayPage'
import { ResultsPage } from './pages/ResultsPage'
import { SettingsPage } from './pages/SettingsPage'
import { useTrustStream } from './lib/useTrustStream'
import { useHashRoute } from './lib/useHashRoute'

export const AppContent: React.FC = () => {
  const [route, navigate] = useHashRoute()
  const stream = useTrustStream()
  const [isInitialConnecting, setIsInitialConnecting] = useState<boolean>(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsInitialConnecting(false)
    }, 400)
    return () => clearTimeout(timer)
  }, [])

  let pageTitle = 'Live'
  let content = <LivePage stream={stream} isConnecting={isInitialConnecting} />
  let topBarActions: React.ReactNode = null

  if (route.startsWith('#/live') || route === '#' || route === '#/') {
    pageTitle = 'Live'
    topBarActions = <SourceSelector stream={stream} />
    content = <LivePage stream={stream} isConnecting={isInitialConnecting} />
  } else if (route.startsWith('#/replay')) {
    pageTitle = 'Replay'
    content = <ReplayPage />
  } else if (route.startsWith('#/results')) {
    pageTitle = 'Results'
    content = <ResultsPage />
  } else if (route.startsWith('#/settings')) {
    pageTitle = 'Settings'
    content = <SettingsPage />
  }

  return (
    <AppShell
      currentRoute={route}
      onNavigate={navigate}
      title={pageTitle}
      connectionState={stream.state}
      topBarActions={topBarActions}
    >
      {content}
    </AppShell>
  )
}

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  )
}

export default App
