export function fmtScore(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) {
    return '--'
  }
  return x.toFixed(2)
}

export function fmtAngle(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) {
    return '--'
  }
  return `${Math.round(x)}°`
}

export function fmtFps(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) {
    return '--'
  }
  return Math.round(x).toString()
}

export function fmtTime(seconds: number): string {
  if (Number.isNaN(seconds) || seconds < 0) {
    return '00:00'
  }
  const totalSecs = Math.floor(seconds)
  const mins = Math.floor(totalSecs / 60)
  const secs = totalSecs % 60
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export function clamp01(x: number): number {
  if (Number.isNaN(x)) return 0
  return Math.max(0, Math.min(1, x))
}
