import { describe, it, expect } from 'vitest'
import {
  streamReducer,
  INITIAL_STREAM_STATE,
  type StreamState,
} from './streamReducer'
import type { FrameResult } from '../types'

function makeTestFrame(idx: number, decision: 'IDLE' | 'ACCEPT' = 'IDLE'): FrameResult {
  return {
    t: idx * 0.05,
    frame_idx: idx,
    fps: 24,
    faces: [],
    speech_active: false,
    env: 0.002,
    noise_floor: 0.002,
    speaker_id: null,
    attribution_conf: 0.0,
    liveness: 0.0,
    addressee: 0.8,
    decision,
    reason: '',
    utterance_final: decision === 'ACCEPT',
  }
}

describe('streamReducer', () => {
  it('handles hello message by setting config', () => {
    const next = streamReducer(INITIAL_STREAM_STATE, {
      type: 'hello',
      config: { gate: { min_sync: 0.35 } },
      decisions: ['IDLE', 'ACCEPT'],
    })
    expect(next.config).toEqual({ gate: { min_sync: 0.35 } })
  })

  it('handles frame message and trims history at 200 after 250 frames', () => {
    let state: StreamState = INITIAL_STREAM_STATE

    for (let i = 0; i < 250; i++) {
      state = streamReducer(state, {
        type: 'frame',
        data: makeTestFrame(i),
      })
    }

    expect(state.frame?.frame_idx).toBe(249)
    expect(state.history.length).toBe(200)
    expect(state.history[0].frame_idx).toBe(50)
    expect(state.history[199].frame_idx).toBe(249)
  })

  it('handles utterance message, prepends to events and trims at 50', () => {
    let state: StreamState = INITIAL_STREAM_STATE
    const tBase = 1000

    for (let i = 0; i < 60; i++) {
      state = streamReducer(state, {
        type: 'utterance',
        data: makeTestFrame(i, 'ACCEPT'),
        ts: tBase + i * 10,
      })
    }

    expect(state.events.length).toBe(50)
    expect(state.events[0].frame_idx).toBe(59) // Newest first
    expect(state.events[49].frame_idx).toBe(10)
    expect(state.held?.frame.frame_idx).toBe(59)
    expect(state.held?.until).toBe(tBase + 59 * 10 + 4000)
  })

  it('handles tick-hold clearing held when expired', () => {
    const tNow = 5000
    const stateWithHeld = streamReducer(INITIAL_STREAM_STATE, {
      type: 'utterance',
      data: makeTestFrame(1, 'ACCEPT'),
      ts: tNow,
    })
    expect(stateWithHeld.held).not.toBeNull()
    expect(stateWithHeld.held?.until).toBe(tNow + 4000)

    // Not yet expired
    const s1 = streamReducer(stateWithHeld, { type: 'tick-hold', now: tNow + 3999 })
    expect(s1.held).not.toBeNull()

    // Expired
    const s2 = streamReducer(stateWithHeld, { type: 'tick-hold', now: tNow + 4000 })
    expect(s2.held).toBeNull()
  })

  it('handles status message and preserves it across subsequent frame messages', () => {
    let state = streamReducer(INITIAL_STREAM_STATE, {
      type: 'status',
      data: { running: true, source: 'session:s1', error: null },
    })
    expect(state.status).toEqual({ running: true, source: 'session:s1', error: null })

    // Later frame message
    state = streamReducer(state, {
      type: 'frame',
      data: makeTestFrame(10),
    })
    expect(state.status).toEqual({ running: true, source: 'session:s1', error: null })
    expect(state.frame?.frame_idx).toBe(10)
  })
})
