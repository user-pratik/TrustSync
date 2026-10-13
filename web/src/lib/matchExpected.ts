export function matchExpected(
  expected: string,
  decision: string | null | undefined,
): boolean {
  if (expected === 'NONE') {
    return decision === null || decision === undefined || decision === 'NONE' || decision === 'IDLE'
  }

  if (!decision) {
    return false
  }

  if (expected === 'ACCEPT') {
    return decision === 'ACCEPT'
  }

  if (expected === 'REJECT_ANY') {
    return decision.startsWith('REJECT_')
  }

  return expected === decision
}
