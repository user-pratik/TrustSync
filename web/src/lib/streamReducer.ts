import type { FrameResult, ServerMsg } from '../types'

export interface StreamState {
  frame: FrameResult | null
  history: FrameResult[]
  events: FrameResult[]
  config: Record<string, unknown> | null
  status: { running: boolean; source: string | null; error: string | null } | null
  held: { frame: FrameResult; until: number } | null
}

export type StreamAction =
  | (ServerMsg & { ts?: number })
  | { type: 'tick-hold'; now?: number }

export const INITIAL_STREAM_STATE: StreamState = {
  frame: null,
  history: [],
  events: [],
  config: null,
  status: null,
  held: null,
}

export function streamReducer(
  state: StreamState,
  msg: StreamAction,
): StreamState {
  switch (msg.type) {
    case 'hello':
      return {
        ...state,
        config: msg.config,
      }

    case 'frame': {
      const nextHistory = [...state.history, msg.data]
      return {
        ...state,
        frame: msg.data,
        history: nextHistory.length > 200 ? nextHistory.slice(-200) : nextHistory,
      }
    }

    case 'utterance': {
      const now = (msg as { ts?: number }).ts ?? Date.now()
      const nextEvents = [msg.data, ...state.events]
      return {
        ...state,
        events: nextEvents.length > 50 ? nextEvents.slice(0, 50) : nextEvents,
        held: {
          frame: msg.data,
          until: now + 4000,
        },
      }
    }

    case 'status':
      return {
        ...state,
        status: msg.data,
      }

    case 'tick-hold': {
      const now = msg.now ?? Date.now()
      if (state.held && now >= state.held.until) {
        return {
          ...state,
          held: null,
        }
      }
      return state
    }

    default:
      return state
  }
}
