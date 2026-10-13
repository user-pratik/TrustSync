export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; message: string }

export interface SessionInfo {
  id: string
  scenario: string
  expected: string
  subjects: string[]
}

async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<ApiResult<T>> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 8000)

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })

    clearTimeout(timeoutId)

    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}: ${res.statusText}`
      try {
        const json = await res.json()
        if (json?.detail) {
          errorMsg = typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail)
        }
      } catch {
        // Fallback to errorMsg
      }
      return { ok: false, status: res.status, message: errorMsg }
    }

    const data = (await res.json()) as T
    return { ok: true, data }
  } catch (err: unknown) {
    clearTimeout(timeoutId)
    const message = err instanceof Error ? err.message : 'Unknown network error'
    return { ok: false, status: 0, message }
  }
}

export async function getSessions(): Promise<ApiResult<SessionInfo[]>> {
  return request<SessionInfo[]>('/api/sessions')
}

export async function startSource(body: {
  type: 'webcam' | 'session'
  id?: string
}): Promise<ApiResult<{ ok: boolean }>> {
  return request<{ ok: boolean }>('/api/source', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function stopSource(): Promise<ApiResult<{ ok: boolean }>> {
  return request<{ ok: boolean }>('/api/stop', {
    method: 'POST',
  })
}

export async function getConfig(): Promise<
  ApiResult<Record<string, Record<string, unknown>>>
> {
  return request<Record<string, Record<string, unknown>>>('/api/config')
}

export async function setConfig(
  section: string,
  key: string,
  value: unknown,
): Promise<ApiResult<Record<string, Record<string, unknown>>>> {
  return request<Record<string, Record<string, unknown>>>('/api/config', {
    method: 'POST',
    body: JSON.stringify({ section, key, value }),
  })
}

export async function getResults(): Promise<
  ApiResult<Record<string, unknown>>
> {
  return request<Record<string, unknown>>('/api/results')
}
