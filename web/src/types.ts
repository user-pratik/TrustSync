export type Decision =
  | 'IDLE'
  | 'LISTENING'
  | 'ACCEPT'
  | 'REJECT_SPOOF'
  | 'REJECT_AMBIGUOUS'
  | 'REJECT_NOT_ADDRESSED'
  | 'REJECT_LOW_QUALITY'
  | 'REJECT_NO_FACE'

export interface FaceScores {
  track_id: number
  bbox: [number, number, number, number]
  sync: number
  lip_std: number
  best_lag_s: number
  yaw: number | null
  pitch: number | null
  roll: number | null
  gaze_off: number
  directed: number
  quality: number
}

export interface FrameResult {
  t: number
  frame_idx: number
  fps: number
  faces: FaceScores[]
  speech_active: boolean
  env: number
  noise_floor: number
  speaker_id: number | null
  attribution_conf: number
  liveness: number
  addressee: number
  decision: Decision
  reason: string
  utterance_final: boolean
}

export interface HelloMsg {
  type: 'hello'
  config: Record<string, Record<string, unknown>>
  decisions: Decision[]
}

export interface FrameMsg {
  type: 'frame'
  data: FrameResult
}

export interface UtteranceMsg {
  type: 'utterance'
  data: FrameResult
}

export interface StatusMsg {
  type: 'status'
  data: {
    running: boolean
    source: string | null
    error: string | null
  }
}

export type ServerMsg = HelloMsg | FrameMsg | UtteranceMsg | StatusMsg
