// HTTP client for the FastAPI model service, with a timeout and clear error types.
//
// On free hosting the model service sleeps when idle. While it wakes up, the host answers with
// 502/503/504 or refuses the connection, so those are retried (with a short pause) until
// `timeoutMs` has passed in total.

export class ModelServiceError extends Error {
  constructor(message, { status = 503, code = 'MODEL_SERVICE_UNAVAILABLE', cause } = {}) {
    super(message, { cause });
    this.status = status;
    this.code = code;
  }
}

const WAKING_STATUSES = new Set([502, 503, 504]);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function createModelClient({
  baseUrl,
  timeoutMs = 5000,
  retryDelayMs = 2000,
  fetchImpl = fetch,
}) {
  const timeoutError = (cause) =>
    new ModelServiceError('The model service took too long to respond.', {
      status: 504,
      code: 'MODEL_SERVICE_TIMEOUT',
      cause,
    });

  async function call(path, init = {}) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw timeoutError();
      const canRetry = () => deadline - Date.now() > retryDelayMs;

      let res;
      try {
        res = await fetchImpl(`${baseUrl}${path}`, {
          ...init,
          signal: AbortSignal.timeout(remaining),
        });
      } catch (err) {
        if (err?.name === 'TimeoutError' || err?.name === 'AbortError') throw timeoutError(err);
        if (canRetry()) {
          await sleep(retryDelayMs);
          continue;
        }
        throw new ModelServiceError('The model service is unavailable. Please try again shortly.', {
          cause: err,
        });
      }

      if (WAKING_STATUSES.has(res.status) && canRetry()) {
        await res.body?.cancel().catch(() => {});
        await sleep(retryDelayMs);
        continue;
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
