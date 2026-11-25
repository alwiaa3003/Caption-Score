// HTTP client for the FastAPI model service, with a timeout and clear error types.

export class ModelServiceError extends Error {
  constructor(message, { status = 503, code = 'MODEL_SERVICE_UNAVAILABLE', cause } = {}) {
    super(message, { cause });
    this.status = status;
    this.code = code;
  }
}

export function createModelClient({ baseUrl, timeoutMs = 5000, fetchImpl = fetch }) {
  async function call(path, init = {}) {
    let res;
    try {
      res = await fetchImpl(`${baseUrl}${path}`, {
        ...init,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
        throw new ModelServiceError('The model service took too long to respond.', {
          status: 504,
          code: 'MODEL_SERVICE_TIMEOUT',
          cause: err,
        });
      }
      throw new ModelServiceError('The model service is unavailable. Please try again shortly.', {
        cause: err,
      });
    }
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      if (res.status === 422) {
        throw new ModelServiceError('The model service rejected the input.', {
          status: 400,
          code: 'MODEL_SERVICE_REJECTED_INPUT',
        });
      }
      throw new ModelServiceError(`The model service returned an error (${res.status}).`, {
        status: 502,
        code: 'MODEL_SERVICE_ERROR',
      });
    }
    return body;
  }

  return {
    predict: (payload) =>
      call('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }),
    health: () => call('/health'),
    modelInfo: () => call('/model-info'),
  };
}
