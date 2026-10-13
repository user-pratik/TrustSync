import type { LucideIcon } from 'lucide-react'
import {
  Moon,
  AudioLines,
  ShieldCheck,
  ShieldAlert,
  Users,
  EyeOff,
  CameraOff,
  UserX,
} from 'lucide-react'
import type { Decision } from '../types'

export type Tone = 'accept' | 'reject' | 'listen' | 'idle'

export interface DecisionMeta {
  label: string
  short: string
  tone: Tone
  icon: LucideIcon
  description: string
}

export const DECISION_META: Record<Decision, DecisionMeta> = {
  IDLE: {
    label: 'Idle',
    short: 'Idle',
    tone: 'idle',
    icon: Moon,
    description: 'Waiting for speech',
  },
  LISTENING: {
    label: 'Listening',
    short: 'Listening',
    tone: 'listen',
    icon: AudioLines,
    description: 'Speech in progress',
  },
  ACCEPT: {
    label: 'Accepted',
    short: 'Accepted',
    tone: 'accept',
    icon: ShieldCheck,
    description: 'Speaker verified and addressing robot',
  },
  REJECT_SPOOF: {
    label: 'Rejected: not live',
    short: 'Not live',
    tone: 'reject',
    icon: ShieldAlert,
    description: 'Lip motion does not match audio',
  },
  REJECT_AMBIGUOUS: {
    label: 'Rejected: unclear speaker',
    short: 'Ambiguous',
    tone: 'reject',
    icon: Users,
    description: 'Multiple faces matching audio',
  },
  REJECT_NOT_ADDRESSED: {
    label: 'Rejected: not for the robot',
    short: 'Not addressed',
    tone: 'reject',
    icon: EyeOff,
    description: 'Speaker is looking away',
  },
  REJECT_LOW_QUALITY: {
    label: 'Rejected: poor view',
    short: 'Low quality',
    tone: 'reject',
    icon: CameraOff,
    description: 'Face too small, dark, or clipped',
  },
  REJECT_NO_FACE: {
    label: 'Rejected: no face',
    short: 'No face',
    tone: 'reject',
    icon: UserX,
    description: 'No face detected in frame',
  },
}

export function toneClasses(tone: Tone): string {
  switch (tone) {
    case 'accept':
      return 'text-accept bg-accept/12 border-accept/30 ring-accept'
    case 'reject':
      return 'text-reject bg-reject/12 border-reject/30 ring-reject'
    case 'listen':
      return 'text-listen bg-listen/12 border-listen/30 ring-listen'
    case 'idle':
      return 'text-idle bg-idle/12 border-idle/30 ring-idle'
  }
}
