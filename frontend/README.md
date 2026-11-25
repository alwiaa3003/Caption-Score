# Frontend (React + Vite)

Pages: **Score** (caption, hashtag chips, post type, "Score it"; result with gauge, verdict,
highlighted words, top up/down words, suggested hashtags), **History** (recent scored captions),
**About** (what "top 10%" means, TikTok caveat, live validation/test PR-AUC from the model card).
Every request has loading and error states.

The API URL comes from `VITE_API_URL` in the repo-root `.env` (default `http://localhost:4000`).

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm test         # component + page tests with a mocked API
npm run lint
npm run build    # production build in dist/
```
