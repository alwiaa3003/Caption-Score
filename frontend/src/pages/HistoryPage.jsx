import { useEffect } from 'react';

import { api } from '../api.js';
import { percentText } from '../constants.js';
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
    <section className="page-narrow">
      <div className="page-head">
        <h1>History</h1>
        <button
          type="button"
          className="text-button"
          onClick={load}
          disabled={status === 'loading'}
        >
          {status === 'loading' ? 'Loading' : 'Refresh'}
        </button>
      </div>
      <p className="sub">The last 50 captions scored here.</p>

      {status === 'loading' && !data && <p className="muted">Loading</p>}
      {status === 'error' && (
        <div className="error-line" role="alert">
          <p>Couldn't load history. {error.message}</p>
        </div>
      )}
      {status === 'success' && data.items.length === 0 && (
        <p className="muted">Nothing scored yet.</p>
      )}
      {data?.items?.length > 0 && (
        <ul className="history">
          {data.items.map((item) => (
            <li key={item.id}>
              <span className="history-score">{percentText(item.probability)}</span>
              <div className="history-body">
                <p className="history-caption">{item.caption}</p>
                <p className="small muted">
                  {item.hashtags?.length > 0 && (
                    <>{item.hashtags.map((h) => `#${h.replace(/^#/, '')}`).join(' ')} · </>
                  )}
                  {item.postType && <>{item.postType} · </>}
                  {fmtDate(item.createdAt)}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
