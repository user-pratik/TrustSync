import { useState, useEffect, useRef, useReducer } from 'react'
import type { FrameResult } from '../types'
import { mockFrame, MOCK_CONFIG } from './mock'
import {
  streamReducer,
  INITIAL_STREAM_STATE,
  type StreamState,
} from './streamReducer'

export interface HeldDecision {
  frame: FrameResult
  until: number
}

export type StreamConnectionState =
  | 'connected'
  | 'mock'
  | 'connecting'
  | 'disconnected'

export interface TrustStreamState extends StreamState {
  state: StreamConnectionState
}

export function isMockRequested(): boolean {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search)
    if (params.get('mock') === '1') return true
  }
  return import.meta.env.VITE_MOCK === '1'
}

export function useTrustStream(mockRequested?: boolean): TrustStreamState {
  const isMock = mockRequested ?? isMockRequested()

  const [connectionState, setConnectionState] =
    useState<StreamConnectionState>(isMock ? 'mock' : 'connecting')
  const [streamState, dispatch] = useReducer(
    streamReducer,
    INITIAL_STREAM_STATE,
  )

  const kRef = useRef<number>(0)
  const malformedCountRef = useRef<number>(0)
  const retryCountRef = useRef<number>(0)

  // 1. Mock Mode Branch
  useEffect(() => {
    if (!isMock) return

    setConnectionState('mock')
    dispatch({
      type: 'hello',
      config: MOCK_CONFIG,
      decisions: [
        'IDLE',
        'LISTENING',
        'ACCEPT',
        'REJECT_SPOOF',
        'REJECT_AMBIGUOUS',
        'REJECT_NOT_ADDRESSED',
        'REJECT_LOW_QUALITY',
        'REJECT_NO_FACE',
      ],
    })
    dispatch({
      type: 'status',
      data: { running: true, source: 'mock', error: null },
    })

    const intervalId = setInterval(() => {
      const curK = kRef.current
      const newFrame = mockFrame(curK)
      kRef.current = curK + 1

      dispatch({ type: 'frame', data: newFrame })

      if (newFrame.utterance_final) {
        dispatch({ type: 'utterance', data: newFrame, ts: Date.now() })
      }

      dispatch({ type: 'tick-hold', now: Date.now() })
    }, 50)

    return () => clearInterval(intervalId)
  }, [isMock])

  // 2. Real WebSocket Branch with exponential backoff
  useEffect(() => {
    if (isMock) return

    let isUnmounted = false
    let socket: WebSocket | null = null
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null

    const BACKOFF_MS = [500, 1000, 2000, 4000, 5000]

    const connect = () => {
      if (isUnmounted) return

      setConnectionState('connecting')
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws`

      try {
        socket = new WebSocket(wsUrl)
      } catch {
        scheduleReconnect()
        return
      }

      socket.onopen = () => {
        if (isUnmounted) {
          socket?.close()
          return
        }
        setConnectionState('connected')
        retryCountRef.current = 0
      }

      socket.onmessage = (event) => {
        if (isUnmounted) return
        try {
          const msg = JSON.parse(event.data)
          dispatch(msg)
          dispatch({ type: 'tick-hold', now: Date.now() })
        } catch {
          malformedCountRef.current += 1
        }
      }

      socket.onclose = () => {
        if (isUnmounted) return
        setConnectionState('disconnected')
        scheduleReconnect()
      }

      socket.onerror = () => {
        if (isUnmounted) return
        setConnectionState('disconnected')
        // onclose will fire after onerror and trigger reconnect
      }
    }

    const scheduleReconnect = () => {
      if (isUnmounted) return
      const backoffIndex = Math.min(
        retryCountRef.current,
        BACKOFF_MS.length - 1,
      )
      const delay = BACKOFF_MS[backoffIndex]
      retryCountRef.current += 1

      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      reconnectTimeout = setTimeout(() => {
        connect()
      }, delay)
    }

    connect()

    // Interval to tick-hold when in real WS mode
    const holdInterval = setInterval(() => {
      dispatch({ type: 'tick-hold', now: Date.now() })
    }, 200)

    return () => {
      isUnmounted = true
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      clearInterval(holdInterval)
      if (socket) {
        socket.onopen = null
        socket.onmessage = null
        socket.onclose = null
        socket.onerror = null
        socket.close()
      }
    }
  }, [isMock])

  return {
    state: connectionState,
    ...streamState,
  }
}
