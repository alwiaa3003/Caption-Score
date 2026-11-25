// Small fetch wrapper for the Express API. Every failure becomes an ApiError with a message
// that is safe to show to the user.
export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:4000').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, { status = 0, details = [] } = {}) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

const FRIENDLY = {
  MODEL_SERVICE_UNAVAILABLE:
    'The scoring model is offline right now. Please try again in a minute.',
  MODEL_SERVICE_TIMEOUT: 'The scoring model took too long to answer. Please try again.',
};

export async function request(
  path,
  { method = 'GET', body, timeoutMs = 15000, fetchImpl = fetch } = {},
) {
  let res;
  try {
    res = await fetchImpl(`${API_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    if (err?.name === 'TimeoutError') {
      throw new ApiError('The server took too long to respond. Please try again.');
    }
    throw new ApiError("Can't reach the CaptionScore server. Is the backend running?");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 429) {
      throw new ApiError('Too many requests. Please wait a minute and try again.', { status: 429 });
    }
    const message = FRIENDLY[data?.code] ?? data?.error ?? `Request failed (${res.status}).`;
    throw new ApiError(message, { status: res.status, details: data?.details ?? [] });
  }
  return data;
}

export const api = {
  predict: (payload, opts) => request('/api/predict', { method: 'POST', body: payload, ...opts }),
  history: (limit = 20, opts) => request(`/api/history?limit=${limit}`, opts),
  modelInfo: (opts) => request('/api/model-info', opts),
};
