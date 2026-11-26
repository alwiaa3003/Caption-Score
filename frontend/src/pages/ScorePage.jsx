import { useEffect, useRef, useState } from 'react';

import { api } from '../api.js';
import HashtagInput from '../components/HashtagInput.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { EXAMPLE, MAX_CAPTION, MAX_HASHTAGS } from '../constants.js';
import { useAsync } from '../useAsync.js';

// After this long, tell the user the server may be waking up (free hosting sleeps when idle).
const SLOW_MS = 8000;

const WIDE_QUERY = '(min-width: 1100px)'; // two-column layout from here up (see styles.css)

function isWide() {
  return typeof window !== 'undefined' && window.matchMedia?.(WIDE_QUERY).matches === true;
}

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
}

/** Why the Score button can't be used yet, or '' if it can. */
function blockedReason({ caption, duration }) {
  if (!caption.trim()) return 'Add a caption first';
  if (caption.length > MAX_CAPTION) return `Shorten the caption to ${MAX_CAPTION} characters`;
  if (duration !== '' && !(Number(duration) > 0 && Number(duration) <= 3600))
    return 'Use a duration from 1 to 3600 seconds';
  return '';
}

export default function ScorePage() {
  const [caption, setCaption] = useState('');
  const [hashtags, setHashtags] = useState([]);
  const [postType, setPostType] = useState('video');
  const [duration, setDuration] = useState('');
  const { status, data, error, run } = useAsync();
  const resultRef = useRef(null);
  const [slow, setSlow] = useState(false);

  const loading = status === 'loading';
  const reason = blockedReason({ caption, duration });
  const canSubmit = !reason && !loading;

  useEffect(() => {
    if (!loading) return undefined;
    const id = setTimeout(() => setSlow(true), SLOW_MS);
    return () => {
      clearTimeout(id);
      setSlow(false);
    };
  }, [loading]);

  // On narrow screens the result sits below the form: bring it into view once it's ready.
  useEffect(() => {
    if ((status === 'success' || status === 'error') && !isWide()) {
      resultRef.current?.scrollIntoView?.({
        behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        block: 'start',
      });
    }
  }, [status]);

  function score(payload) {
    run(() => api.predict(payload));
  }

  function submit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    const payload = { caption, hashtags, post_type: postType };
    if (duration !== '') payload.duration_sec = Number(duration);
    score(payload);
  }

  function tryExample() {
    setCaption(EXAMPLE.caption);
    setHashtags(EXAMPLE.hashtags);
    setPostType('video');
    setDuration('');
    score({ caption: EXAMPLE.caption, hashtags: EXAMPLE.hashtags, post_type: 'video' });
  }

  function addSuggested(tag) {
    if (!hashtags.includes(tag) && hashtags.length < MAX_HASHTAGS) setHashtags([...hashtags, tag]);
  }

  const tooLong = caption.length > MAX_CAPTION;
  const badDuration = reason.startsWith('Use a duration');

  return (
    <div className="score-layout">
      <section className="editor">
        <h1>Score a caption</h1>
        <p className="sub">Compare a caption with similar posts.</p>
        <form onSubmit={submit} noValidate>
          <label htmlFor="caption">Caption</label>
          <textarea
            id="caption"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            rows={6}
            placeholder="Type or paste your caption"
            aria-invalid={tooLong}
            aria-describedby="caption-count"
          />
          <div className="below-caption">
            <span id="caption-count" className={tooLong ? 'field-error' : 'field-help'}>
              {caption.length} / {MAX_CAPTION}
            </span>
            <button type="button" className="text-button" onClick={tryExample} disabled={loading}>
              Try an example
            </button>
          </div>

          <div className="submit-row">
            <button
              type="submit"
              className="primary"
              disabled={!canSubmit}
              aria-describedby={reason ? 'score-hint' : undefined}
            >
              {loading ? 'Scoring' : 'Score'}
            </button>
            {reason && !loading && (
              <p id="score-hint" className="field-help">
                {reason}
              </p>
            )}
          </div>

          <div className="optional">
            <label htmlFor="hashtags">
              Hashtags <span className="optional-tag">optional</span>
            </label>
            <HashtagInput value={hashtags} onChange={setHashtags} />

            <div className="compact-row">
              <div>
                <label htmlFor="post-type">Post type</label>
                <select
                  id="post-type"
                  value={postType}
                  onChange={(e) => setPostType(e.target.value)}
                >
                  <option value="video">Video</option>
                  <option value="reel">Reel</option>
                  <option value="photo">Photo</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label htmlFor="duration">Duration (seconds)</label>
                <input
                  id="duration"
                  type="number"
                  inputMode="numeric"
                  min="1"
                  max="3600"
                  step="1"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  aria-invalid={badDuration}
                />
              </div>
            </div>
          </div>
          <p className="field-help footnote">Scored captions are saved in History.</p>
        </form>
      </section>

      <div className="result-col" ref={resultRef}>
        {status === 'idle' && <p className="empty-line">Your result shows here.</p>}
        {loading && (
          <p className="empty-line" aria-busy="true">
            Scoring your caption
            {slow && (
              <span className="slow-note small muted">
                The server is waking up. This can take a minute or two.
              </span>
            )}
          </p>
        )}
        {status === 'error' && (
          <div className="error-line" role="alert">
            <p>{error.message}</p>
            {error.details?.length > 0 && (
              <ul>
                {error.details.map((d) => (
                  <li key={d.field + d.message}>
                    {d.field}: {d.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        {status === 'success' && data && (
          <ResultPanel result={data} onAddHashtag={addSuggested} addedHashtags={hashtags} />
        )}
      </div>
    </div>
  );
}
