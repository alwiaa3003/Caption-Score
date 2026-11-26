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
  MODEL_SERVICE_UNAVAILABLE: "Couldn't reach the model. Try again in a minute.",
  MODEL_SERVICE_TIMEOUT: 'The model took too long. Try again in a minute.',
};

// Long timeout: on free hosting the API and the model can both be asleep, and waking both can
// take well over a minute.
export async function request(
  path,
  { method = 'GET', body, timeoutMs = 120000, fetchImpl = fetch } = {},
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
      throw new ApiError('The server took too long. Try again in a minute.');
    }
    throw new ApiError(
      "Couldn't reach the server. Check that the backend is running, then try again.",
    );
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 429) {
      throw new ApiError('Too many requests. Wait a minute and try again.', { status: 429 });
    }
    const message = FRIENDLY[data?.code] ?? data?.error ?? 'Something went wrong. Try again.';
    throw new ApiError(message, { status: res.status, details: data?.details ?? [] });
  }
  return data;
}

export const api = {
  predict: (payload, opts) => request('/api/predict', { method: 'POST', body: payload, ...opts }),
  history: (limit = 20, opts) => request(`/api/history?limit=${limit}`, opts),
  modelInfo: (opts) => request('/api/model-info', opts),
};
