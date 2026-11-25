// All configuration comes from environment variables. In development they are read from the
// single .env file at the repo root (copy .env.example). Secrets (MONGODB_URI) are never logged.
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import dotenv from 'dotenv';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
dotenv.config({ path: resolve(repoRoot, '.env'), quiet: true });

function int(name, fallback) {
  const v = process.env[name];
  const n = v === undefined || v === '' ? fallback : Number.parseInt(v, 10);
  if (!Number.isFinite(n)) throw new Error(`${name} must be an integer`);
  return n;
}

export const config = {
  port: int('PORT', 4000),
  mongodbUri: process.env.MONGODB_URI ?? '',
  modelServiceUrl: (process.env.MODEL_SERVICE_URL ?? 'http://localhost:8000').replace(/\/$/, ''),
  modelServiceTimeoutMs: int('MODEL_SERVICE_TIMEOUT_MS', 5000),
  corsOrigin: (process.env.CORS_ORIGIN ?? 'http://localhost:5173')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),
  rateLimitPerMinute: int('RATE_LIMIT_PER_MINUTE', 30),
};
