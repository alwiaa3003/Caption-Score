// Express app factory. Dependencies (model client, rate limit) are injected so tests can swap
// them; server.js wires the real ones.
import cors from 'cors';
import express from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import mongoose from 'mongoose';

import { Prediction } from './models/Prediction.js';
import { ModelServiceError } from './services/modelClient.js';
import { historyQuerySchema, predictSchema, validate } from './validation.js';

export function createApp({ modelClient, corsOrigin, rateLimitPerMinute = 30, logger = console }) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', 1); // correct client IPs for rate limiting behind one proxy
  app.use(helmet());
  app.use(cors({ origin: corsOrigin, methods: ['GET', 'POST'] }));
  app.use(express.json({ limit: '20kb' }));

  const predictLimiter = rateLimit({
    windowMs: 60_000,
    limit: rateLimitPerMinute,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: { error: 'Too many requests. Please wait a minute and try again.' },
  });

  // Liveness for the host's health check: the process is up. It doesn't depend on the model
  // service, which may be asleep or still deploying on a free host.
  app.get('/api/live', (req, res) => res.json({ status: 'ok' }));

  app.get('/api/health', async (req, res) => {
    const db = mongoose.connection.readyState === 1 ? 'connected' : 'disconnected';
    let model = 'unreachable';
    let modelVersion = null;
    try {
      modelVersion = (await modelClient.health()).model_version;
      model = 'ok';
    } catch {
      /* reported below */
    }
    const ok = db === 'connected' && model === 'ok';
    res.status(ok ? 200 : 503).json({ status: ok ? 'ok' : 'degraded', db, model, modelVersion });
  });

  app.get('/api/model-info', async (req, res) => {
    res.json(await modelClient.modelInfo());
  });

  app.post('/api/predict', predictLimiter, validate(predictSchema), async (req, res) => {
    const input = req.validated;
    const result = await modelClient.predict(input);

    let saved = true;
    let id = null;
    try {
      const doc = await Prediction.create({
        caption: input.caption,
        hashtags: input.hashtags ?? [],
        postType: input.post_type ?? null,
        durationSec: input.duration_sec ?? null,
        probability: result.probability,
        label: result.label,
        modelVersion: result.model_version,
        topPositiveTokens: result.top_positive_tokens,
        topNegativeTokens: result.top_negative_tokens,
        suggestedHashtags: result.suggested_hashtags,
      });
      id = doc._id.toString();
    } catch (err) {
      // The score is still useful even if history can't be saved; say so instead of failing.
      saved = false;
      logger.error('could not save prediction:', err.message);
    }
    res.json({ ...result, id, saved });
  });

  app.get('/api/history', validate(historyQuerySchema, 'query'), async (req, res) => {
    if (mongoose.connection.readyState !== 1) {
      return res.status(503).json({ error: 'History is unavailable: database not connected.' });
    }
    const docs = await Prediction.find({}, { __v: 0 })
      .sort({ createdAt: -1 })
      .limit(req.validated.limit)
      .lean();
    res.json({
      items: docs.map(({ _id, ...rest }) => ({ id: _id.toString(), ...rest })),
    });
  });

  app.use('/api', (req, res) => res.status(404).json({ error: 'Not found' }));

  // Central error handler (Express 5 forwards rejected async handlers here).
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err instanceof ModelServiceError) {
      return res.status(err.status).json({ error: err.message, code: err.code });
    }
    if (err?.type === 'entity.parse.failed') {
      return res.status(400).json({ error: 'Request body must be valid JSON' });
    }
    if (err?.type === 'entity.too.large') {
      return res.status(413).json({ error: 'Request body too large' });
    }
    logger.error(err);
    return res.status(500).json({ error: 'Internal server error' });
  });

  return app;
}
