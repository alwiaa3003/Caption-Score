# Backend (Node + Express + MongoDB)

| Endpoint | |
|---|---|
| `POST /api/predict` | validates `{caption, hashtags?, post_type?, duration_sec?}` (zod), rate-limited, calls the model service, saves to MongoDB, returns the model result plus `id` and `saved` |
| `GET /api/history?limit=20` | recent predictions, newest first (max 100) |
| `GET /api/health` | `{status, db, model, modelVersion}`; 503 if MongoDB or the model service is down |
| `GET /api/model-info` | model card from the model service (for the About page) |

Errors: invalid input -> 400 with `details`; model service down -> 503
`MODEL_SERVICE_UNAVAILABLE`; model timeout -> 504; rate limit -> 429. If MongoDB can't save,
the score is still returned with `saved: false`.

Stored per prediction (collection `predictions`): caption, hashtags, postType, durationSec,
probability, label, modelVersion, top tokens, suggested hashtags, createdAt. No IP address or
any other personal data is stored.

## Run
Configuration comes from the repo-root `.env` (see `../.env.example`): `MONGODB_URI`,
`MODEL_SERVICE_URL`, `PORT`, `CORS_ORIGIN`, `RATE_LIMIT_PER_MINUTE`.
```bash
cd backend
npm install
npm run dev        # http://localhost:4000 (needs MONGODB_URI and the model service)
npm test           # in-memory MongoDB + fake model client; needs neither
npm run lint
```
