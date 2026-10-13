import { useState, useEffect, useCallback } from 'react'

export function useHashRoute(): [string, (newRoute: string) => void] {
  const getHash = () => {
    const hash = window.location.hash
    if (!hash || hash === '#' || hash === '#/') {
      return '#/live'
    }
    return hash
  }

  const [route, setRoute] = useState<string>(getHash)

  useEffect(() => {
    const handleHashChange = () => {
      setRoute(getHash())
    }

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  const navigate = useCallback((newRoute: string) => {
    const formatted = newRoute.startsWith('#') ? newRoute : `#${newRoute}`
    window.location.hash = formatted
    setRoute(formatted)
  }, [])

  return [route, navigate]
}
