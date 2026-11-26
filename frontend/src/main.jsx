import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

// Self-hosted font (bundled by Vite, no CDN): Figtree, one family for everything.
import '@fontsource-variable/figtree/wght.css';

import App from './App.jsx';
import { API_URL } from './api.js';
import './styles.css';

// Wake the API and the model as soon as the page opens (free hosting sleeps when idle), so
// they are usually awake by the time a caption is typed. The result is not needed.
fetch(`${API_URL}/api/health`).catch(() => {});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
