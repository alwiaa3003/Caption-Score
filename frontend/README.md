# Frontend (React + Vite)

Pages: **Score** (caption, optional hashtags, post type and duration; the result shows the
chance of reaching the top 10%, a comparison with a typical post, the caption with the words
that moved the score marked, and hashtag suggestions), **History** (recent scored captions) and
**About** (how the project works, with the real validation and test numbers from the model).
Every request has loading and error states. Colours are checked for WCAG AA in
`src/test/contrast.test.js`; the font is Figtree, bundled with @fontsource.

The API URL comes from `VITE_API_URL` in the repo-root `.env` (default `http://localhost:4000`).

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm test         # component + page tests with a mocked API
npm run lint
npm run build    # production build in dist/
```
