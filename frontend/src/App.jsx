import { NavLink, Route, Routes } from 'react-router-dom';

import AboutPage from './pages/AboutPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';
import ScorePage from './pages/ScorePage.jsx';

export default function App() {
  return (
    <>
      <header className="site-header">
        <div className="container header-inner">
          <NavLink to="/" className="brand">
            CaptionScore
          </NavLink>
          <nav aria-label="Main">
            <NavLink to="/" end>
              Score
            </NavLink>
            <NavLink to="/history">History</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
        </div>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<ScorePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route
            path="*"
            element={
              <section className="page-narrow">
                <h1>Page not found</h1>
                <NavLink to="/">Go to Score</NavLink>
              </section>
            }
          />
        </Routes>
      </main>
      <footer className="container site-footer small muted">
        Trained on US TikTok captions. Treat the score as a rough guide.{' '}
        <NavLink to="/about">About</NavLink>
      </footer>
    </>
  );
}
