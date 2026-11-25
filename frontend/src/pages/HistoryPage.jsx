import { useEffect } from 'react';

import { api } from '../api.js';
import { VERDICT } from '../constants.js';
import { useAsync } from '../useAsync.js';

const fmtDate = (iso) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

export default function HistoryPage() {
  const { status, data, error, run } = useAsync();
  const load = () => run(() => api.history(50));
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="card">
      <div className="row space-between">
        <h1>History</h1>
        <button type="button" onClick={load} disabled={status === 'loading'}>
          {status === 'loading' ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      <p className="muted">The 50 most recent captions scored on this site.</p>

      {status === 'loading' && !data && <div className="skeleton" aria-busy="true" />}
      {status === 'error' && (
        <div className="error" role="alert">
          <strong>Couldn't load history.</strong> {error.message}
        </div>
      )}
      {status === 'success' && data.items.length === 0 && (
        <p className="muted">Nothing here yet. Score a caption and it will show up here.</p>
      )}
      {data?.items?.length > 0 && (
        <ul className="history">
          {data.items.map((item) => (
            <li key={item.id}>
              <div className="history-score">
                <strong>{Math.round(item.probability * 1000) / 10}%</strong>
                <span className={`badge badge-${item.label}`}>
                  {item.label === 'likely' ? 'likely' : 'unlikely'}
                </span>
              </div>
              <div className="history-body">
                <p className="history-caption">{item.caption}</p>
                <p className="muted small">
                  {item.hashtags?.length > 0 && (
                    <>{item.hashtags.map((h) => `#${h.replace(/^#/, '')}`).join(' ')} · </>
                  )}
                  {item.postType && <>{item.postType} · </>}
                  {fmtDate(item.createdAt)} · model {item.modelVersion}
                </p>
                <p className="sr-only">{VERDICT[item.label]}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
