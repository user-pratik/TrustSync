import type { Decision, FaceScores, FrameResult } from '../types'

export const MOCK_CONFIG = {
  gate: {
    min_sync: 0.35,
    min_margin: 0.1,
    min_directed: 0.5,
    min_quality: 0.5,
  },
  addressee: {
    max_yaw_deg: 25,
    max_pitch_deg: 20,
    yaw_offset_deg: 0.0,
    pitch_offset_deg: 0.0,
    gaze_weight: 0.3,
    gaze_max_off: 0.6,
    use_gaze: true,
  },
}

export function mockFrame(k: number): FrameResult {
  // Normalize k to the 980-frame cycle (7 acts * 7s * 20fps = 980 frames)
  const actFrames = 140 // 7.0 s * 20 fps
  const totalFrames = 980
  const normK = ((k % totalFrames) + totalFrames) % totalFrames

  const actIdx = Math.floor(normK / actFrames) // 0..6
  const frameInAct = normK % actFrames
  const t = k / 20.0

  // Timeline within an act:
  // 0.0 to 1.2 s: IDLE (frameInAct 0..23)
  // 1.2 to 3.7 s: LISTENING (frameInAct 24..73)
  // exactly at 3.7 s: utterance_final (frameInAct 74 = Math.round(3.7 * 20))
  // > 3.7 s: IDLE (frameInAct 75..139)
  const finalFrameIdx = 74 // 3.7 * 20

  const isIdleBefore = frameInAct < 24
  const isListening = frameInAct >= 24 && frameInAct < finalFrameIdx
  const isFinal = frameInAct === finalFrameIdx
  const isIdleAfter = frameInAct > finalFrameIdx

  const speechActive = isListening || isFinal

  // Deterministic envelope
  const env = speechActive
    ? 0.05 + 0.035 * (Math.sin(k * 0.7) + 1.0) / 2.0 + 0.035 * (Math.sin(k * 1.9) + 1.0) / 2.0
    : 0.002

  // Progress of listening phase (0..1)
  const listenProgress = isListening
    ? (frameInAct - 24) / (finalFrameIdx - 24)
    : isFinal
      ? 1.0
      : 0.0

  let decision: Decision = 'IDLE'
  let reason = ''
  let speakerId: number | null = null
  let liveness = 0.0
  let attributionConf = 0.0
  let addressee = 0.8
  let faces: FaceScores[] = []

  // Drift for visual movement
  const driftX = 0.02 * Math.sin(k * 0.1)
  const driftY = 0.01 * Math.cos(k * 0.1)
  const w = 0.22
  const h = 0.35

  const makeFace = (
    id: number,
    baseCx: number,
    baseCy: number,
    sync: number,
    lipStd: number,
    directedScore: number,
    yawDeg: number | null,
    qualityScore: number,
  ): FaceScores => {
    const cx = Math.max(0.15, Math.min(0.85, baseCx + driftX))
    const cy = Math.max(0.2, Math.min(0.8, baseCy + driftY))
    return {
      track_id: id,
      bbox: [
        Math.max(0, cx - w / 2),
        Math.max(0, cy - h / 2),
        Math.min(1, cx + w / 2),
        Math.min(1, cy + h / 2),
      ],
      sync,
      lip_std: lipStd,
      best_lag_s: 0.033,
      yaw: yawDeg,
      pitch: 0,
      roll: 0,
      gaze_off: 0.1,
      directed: directedScore,
      quality: qualityScore,
    }
  }

  // Configure per act
  switch (actIdx) {
    case 0: {
      // Act 0: live accept (one face id 0 centred, final sync 0.78, directed 0.95, ACCEPT, speaker 0)
      const curSync = speechActive ? 0.1 + (0.78 - 0.1) * listenProgress : 0.0
      const face0 = makeFace(0, 0.5, 0.45, curSync, 0.015, 0.95, 0, 0.95)
      faces = [face0]
      liveness = curSync
      attributionConf = curSync
      addressee = 0.95
      speakerId = 0
      if (isFinal) {
        decision = 'ACCEPT'
        reason = 'best sync 0.78, directed 0.95: live speech directed at robot'
      }
      break
    }
    case 1: {
      // Act 1: replay spoof (one face, sync 0.04, lip_std 0.001, REJECT_SPOOF)
      const curSync = speechActive ? 0.02 + 0.02 * listenProgress : 0.0
      const face0 = makeFace(0, 0.5, 0.45, curSync, 0.001, 0.9, 0, 0.9)
      faces = [face0]
      liveness = curSync
      attributionConf = curSync
      addressee = 0.9
      speakerId = 0
      if (isFinal) {
        decision = 'REJECT_SPOOF'
        reason = 'best sync 0.04 and lip motion 0.001: audio not explained by mouth'
      }
      break
    }
    case 2: {
      // Act 2: two faces id 0 left and id 1 right, id 1 sync 0.74 and id 0 sync 0.18, ACCEPT speaker 1
      const sync1 = speechActive ? 0.1 + (0.74 - 0.1) * listenProgress : 0.0
      const sync0 = speechActive ? 0.05 + (0.18 - 0.05) * listenProgress : 0.0
      const face0 = makeFace(0, 0.3, 0.45, sync0, 0.008, 0.85, -10, 0.9)
      const face1 = makeFace(1, 0.7, 0.45, sync1, 0.016, 0.92, 5, 0.92)
      faces = [face0, face1]
      liveness = sync1
      attributionConf = sync1 - sync0
      addressee = 0.92
      speakerId = 1
      if (isFinal) {
        decision = 'ACCEPT'
        reason = 'best sync 0.74 (margin 0.56) attributed to speaker 1'
      }
      break
    }
    case 3: {
      // Act 3: not addressed (yaw 45, directed 0.1, sync 0.7, REJECT_NOT_ADDRESSED)
      const curSync = speechActive ? 0.1 + (0.7 - 0.1) * listenProgress : 0.0
      const face0 = makeFace(0, 0.5, 0.45, curSync, 0.014, 0.1, 45, 0.9)
      faces = [face0]
      liveness = curSync
      attributionConf = curSync
      addressee = 0.1
      speakerId = 0
      if (isFinal) {
        decision = 'REJECT_NOT_ADDRESSED'
        reason = 'directed score 0.10 below threshold 0.50 (yaw 45°)'
      }
      break
    }
    case 4: {
      // Act 4: no face (empty faces, REJECT_NO_FACE)
      faces = []
      liveness = 0.0
      attributionConf = 0.0
      addressee = 0.0
      speakerId = null
      if (isFinal) {
        decision = 'REJECT_NO_FACE'
        reason = 'no face detected during utterance'
      }
      break
    }
    case 5: {
      // Act 5: ambiguous (two faces both sync about 0.6, REJECT_AMBIGUOUS)
      const sync0 = speechActive ? 0.1 + (0.61 - 0.1) * listenProgress : 0.0
      const sync1 = speechActive ? 0.1 + (0.59 - 0.1) * listenProgress : 0.0
      const face0 = makeFace(0, 0.3, 0.45, sync0, 0.012, 0.88, -5, 0.9)
      const face1 = makeFace(1, 0.7, 0.45, sync1, 0.012, 0.86, 5, 0.9)
      faces = [face0, face1]
      liveness = Math.max(sync0, sync1)
      attributionConf = Math.abs(sync0 - sync1) // ~0.02 < 0.10 margin
      addressee = 0.88
      speakerId = 0
      if (isFinal) {
        decision = 'REJECT_AMBIGUOUS'
        reason = 'sync margin 0.02 below threshold 0.10 between faces'
      }
      break
    }
    case 6: {
      // Act 6: low quality (quality 0.2, REJECT_LOW_QUALITY)
      const curSync = speechActive ? 0.1 + (0.6 - 0.1) * listenProgress : 0.0
      const face0 = makeFace(0, 0.5, 0.45, curSync, 0.01, 0.85, 0, 0.2)
      faces = [face0]
      liveness = curSync
      attributionConf = curSync
      addressee = 0.85
      speakerId = 0
      if (isFinal) {
        decision = 'REJECT_LOW_QUALITY'
        reason = 'face quality 0.20 below threshold 0.50'
      }
      break
    }
  }

  if (isListening) {
    decision = 'LISTENING'
    reason = 'listening'
  } else if (isIdleBefore || isIdleAfter) {
    decision = 'IDLE'
    reason = ''
    speakerId = null
  }

  return {
    t,
    frame_idx: k,
    fps: 24,
    faces,
    speech_active: speechActive,
    env,
    noise_floor: 0.002,
    speaker_id: speakerId,
    attribution_conf: Math.max(0, Math.min(1, attributionConf)),
    liveness: Math.max(0, Math.min(1, liveness)),
    addressee: Math.max(0, Math.min(1, addressee)),
    decision,
    reason,
    utterance_final: isFinal,
  }
}
