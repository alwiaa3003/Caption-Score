// API tests: in-memory MongoDB (mongodb-memory-server) + a fake model client.
// No real database or model service is needed.
import { MongoMemoryServer } from 'mongodb-memory-server';
import mongoose from 'mongoose';
import request from 'supertest';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';
import { Prediction } from '../src/models/Prediction.js';
import { createModelClient, ModelServiceError } from '../src/services/modelClient.js';
import { MAX_CAPTION_CHARS } from '../src/validation.js';

const RESULT = {
  probability: 0.23,
  label: 'likely',
  top_positive_tokens: [{ token: 'funny', weight: 0.4 }],
  top_negative_tokens: [{ token: 'my', weight: -0.2 }],
  suggested_hashtags: [{ hashtag: 'comedy', lift: 1.9, relevance: 0.3, probability_change: 0.02 }],
  model_version: 'test-model',
  base_rate: 0.1,
  likely_threshold: 0.18,
  tokens: [],
  notes: [],
};

function fakeModel({ fail } = {}) {
  const calls = [];
  return {
    calls,
    predict: async (payload) => {
      calls.push(payload);
      if (fail) throw fail;
      return RESULT;
    },
    health: async () => {
      if (fail) throw fail;
      return { status: 'ok', model_version: 'test-model' };
    },
    modelInfo: async () => ({ model_version: 'test-model' }),
  };
}

const quietLogger = { error: () => {}, info: () => {}, warn: () => {} };
const makeApp = (opts = {}) =>
  createApp({
    modelClient: fakeModel(),
    corsOrigin: ['http://localhost:5173'],
    rateLimitPerMinute: 1000,
    logger: quietLogger,
    ...opts,
  });

let mongod;
beforeAll(async () => {
  mongod = await MongoMemoryServer.create();
  await mongoose.connect(mongod.getUri());
}, 120_000);
afterAll(async () => {
  await mongoose.disconnect();
  await mongod?.stop();
});
beforeEach(async () => {
  await Prediction.deleteMany({});
});

describe('POST /api/predict validation', () => {
  const cases = [
    ['missing caption', {}, 'caption'],
    ['blank caption', { caption: '   ' }, 'caption'],
    ['caption too long', { caption: 'x'.repeat(MAX_CAPTION_CHARS + 1) }, 'caption'],
    ['caption not a string', { caption: 42 }, 'caption'],
    ['bad hashtag', { caption: 'ok', hashtags: ['no spaces!'] }, 'hashtags.0'],
    ['too many hashtags', { caption: 'ok', hashtags: Array(31).fill('a') }, 'hashtags'],
    ['bad post type', { caption: 'ok', post_type: 'story' }, 'post_type'],
    ['negative duration', { caption: 'ok', duration_sec: -1 }, 'duration_sec'],
    ['unknown field', { caption: 'ok', email: 'a@b.c' }, '(body)'],
  ];
  it.each(cases)('rejects %s with 400', async (_, body, field) => {
    const model = fakeModel();
    const res = await request(makeApp({ modelClient: model }))
      .post('/api/predict')
      .send(body);
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Invalid input');
    expect(res.body.details.map((d) => d.field)).toContain(field);
    expect(model.calls).toHaveLength(0); // never reaches the model service
  });

  it('rejects malformed JSON with 400', async () => {
    const res = await request(makeApp())
      .post('/api/predict')
      .set('Content-Type', 'application/json')
      .send('{"caption": ');
    expect(res.status).toBe(400);
  });

  it('accepts a caption at the length limit', async () => {
    const res = await request(makeApp())
      .post('/api/predict')
      .send({ caption: 'a'.repeat(MAX_CAPTION_CHARS) });
    expect(res.status).toBe(200);
  });
});

describe('POST /api/predict', () => {
  it('calls the model, saves to MongoDB and returns the result', async () => {
    const model = fakeModel();
    const body = {
      caption: 'funny dog 😂',
      hashtags: ['#dog'],
      post_type: 'video',
      duration_sec: 12,
    };
    const res = await request(makeApp({ modelClient: model }))
      .post('/api/predict')
      .send(body);
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ probability: 0.23, label: 'likely', saved: true });
    expect(model.calls[0]).toEqual(body);
    const docs = await Prediction.find().lean();
    expect(docs).toHaveLength(1);
    expect(docs[0]).toMatchObject({
      caption: 'funny dog 😂',
      hashtags: ['#dog'],
      postType: 'video',
      durationSec: 12,
      probability: 0.23,
      label: 'likely',
      modelVersion: 'test-model',
    });
    expect(docs[0].createdAt).toBeInstanceOf(Date);
    expect(res.body.id).toBe(docs[0]._id.toString());
  });

  it('returns 503 with a clear message when the model service is down', async () => {
    const down = new ModelServiceError(
      'The model service is unavailable. Please try again shortly.',
    );
    const res = await request(makeApp({ modelClient: fakeModel({ fail: down }) }))
      .post('/api/predict')
      .send({ caption: 'hello' });
    expect(res.status).toBe(503);
    expect(res.body).toEqual({
      error: 'The model service is unavailable. Please try again shortly.',
      code: 'MODEL_SERVICE_UNAVAILABLE',
    });
    expect(await Prediction.countDocuments()).toBe(0);
  });

  it('rate limits per client', async () => {
    const app = makeApp({ rateLimitPerMinute: 2 });
    for (let i = 0; i < 2; i += 1) {
      expect((await request(app).post('/api/predict').send({ caption: 'hi' })).status).toBe(200);
    }
    const res = await request(app).post('/api/predict').send({ caption: 'hi' });
    expect(res.status).toBe(429);
    expect(res.body.error).toMatch(/Too many requests/);
  });

  it('sends CORS headers only for the allowed origin', async () => {
    const app = makeApp();
    const ok = await request(app).get('/api/model-info').set('Origin', 'http://localhost:5173');
    expect(ok.headers['access-control-allow-origin']).toBe('http://localhost:5173');
    const bad = await request(app).get('/api/model-info').set('Origin', 'http://evil.example');
    expect(bad.headers['access-control-allow-origin']).toBeUndefined();
  });
});

describe('GET /api/history', () => {
  it('lists recent predictions, newest first, with a limit', async () => {
    const app = makeApp();
    for (const caption of ['first', 'second', 'third']) {
      await request(app).post('/api/predict').send({ caption });
    }
    const res = await request(app).get('/api/history?limit=2');
    expect(res.status).toBe(200);
    expect(res.body.items.map((i) => i.caption)).toEqual(['third', 'second']);
    expect(res.body.items[0]).toHaveProperty('id');
    expect(res.body.items[0]).not.toHaveProperty('_id');
  });

  it('validates the limit', async () => {
    expect((await request(makeApp()).get('/api/history?limit=500')).status).toBe(400);
    expect((await request(makeApp()).get('/api/history?limit=abc')).status).toBe(400);
  });
});

describe('GET /api/health', () => {
  it('reports db and model status', async () => {
    const res = await request(makeApp()).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({
      status: 'ok',
      db: 'connected',
      model: 'ok',
      modelVersion: 'test-model',
    });
  });

  it('is degraded when the model service is unreachable', async () => {
    const res = await request(makeApp({ modelClient: fakeModel({ fail: new Error('x') }) })).get(
      '/api/health',
    );
    expect(res.status).toBe(503);
    expect(res.body.model).toBe('unreachable');
  });
});

describe('GET /api/live', () => {
  it('is ok even when the model service is unreachable', async () => {
    const res = await request(makeApp({ modelClient: fakeModel({ fail: new Error('x') }) })).get(
      '/api/live',
    );
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: 'ok' });
  });
});

describe('model client', () => {
  it('maps connection errors to 503 and timeouts to 504', async () => {
    const refused = createModelClient({
      baseUrl: 'http://x',
      fetchImpl: async () => {
        throw new TypeError('fetch failed');
      },
    });
    await expect(refused.predict({ caption: 'a' })).rejects.toMatchObject({ status: 503 });

    const slow = createModelClient({
      baseUrl: 'http://x',
      timeoutMs: 10,
      fetchImpl: (url, { signal }) =>
        new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason))),
    });
    await expect(slow.predict({ caption: 'a' })).rejects.toMatchObject({
      status: 504,
      code: 'MODEL_SERVICE_TIMEOUT',
    });
  });

  it('maps a 500 from the model service to 502', async () => {
    const broken = createModelClient({
      baseUrl: 'http://x',
      fetchImpl: async () => new Response('oops', { status: 500 }),
    });
    await expect(broken.health()).rejects.toMatchObject({ status: 502 });
  });
});
