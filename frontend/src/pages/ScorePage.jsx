import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { api } from '../api.js';
import HashtagInput from '../components/HashtagInput.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { EXAMPLE, MAX_CAPTION, MAX_HASHTAGS } from '../constants.js';
import { useAsync } from '../useAsync.js';

const WIDE_QUERY = '(min-width: 1100px)'; // two-column layout from here up (see styles.css)

function isWide() {
  return typeof window !== 'undefined' && window.matchMedia?.(WIDE_QUERY).matches === true;
}

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
}

/** Why the button is disabled (shown as its label), or '' if it can be pressed. */
function blockedReason({ caption, duration, loading }) {
  if (loading) return '';
  if (!caption.trim()) return 'Add a caption to score it';
  if (caption.length > MAX_CAPTION) return 'Shorten the caption to score it';
  if (duration !== '' && !(Number(duration) > 0 && Number(duration) <= 3600))
    return 'Fix the duration to score it';
  return '';
}

function EmptyResult({ onExample }) {
  return (
    <section className="card result-empty">
      <h2>Your score will appear here</h2>
      <p>
        You'll see the chance of reaching the top 10% of comparable posts, which words push it up or
        down, and related hashtags.
      </p>
      <button type="button" className="link-button" onClick={onExample}>
        Try an example caption
      </button>
    </section>
  );
}

export default function ScorePage() {
  const [caption, setCaption] = useState('');
  const [hashtags, setHashtags] = useState([]);
  const [postType, setPostType] = useState('video');
  const [duration, setDuration] = useState('');
  const { status, data, error, run } = useAsync();
  const resultRef = useRef(null);

  const loading = status === 'loading';
  const reason = blockedReason({ caption, duration, loading });
  const canSubmit = !reason && !loading;

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
  const badDuration = reason === 'Fix the duration to score it';

  return (
    <div className="score-layout">
      <section className="card editor">
        <h1>Score a caption</h1>
        <p className="intro">
          See how likely a short-video caption is to reach the top 10% of comparable posts, and
          which words help or hurt.
        </p>
        <form onSubmit={submit} noValidate>
          <label htmlFor="caption">Caption</label>
          <textarea
            id="caption"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            rows={6}
            placeholder="Paste or type your caption, e.g. day 3 of learning to skate"
            aria-invalid={tooLong}
            aria-describedby="caption-help"
          />
          <div className="below-caption">
            <span id="caption-help" className={tooLong ? 'field-error' : 'field-help'}>
              {caption.length}/{MAX_CAPTION} characters{tooLong && ' (too long)'}
            </span>
            <button type="button" className="link-button" onClick={tryExample} disabled={loading}>
              Try an example
            </button>
          </div>

          <button
            type="submit"
            className={`primary big${reason ? ' is-blocked' : ''}${loading ? ' is-loading' : ''}`}
            disabled={!canSubmit}
          >
            {loading ? 'Scoring…' : reason || 'Score caption'}
          </button>

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
                <label htmlFor="duration">Duration (sec)</label>
                <input
                  id="duration"
                  type="number"
                  inputMode="numeric"
                  min="1"
                  max="3600"
                  step="1"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  placeholder="e.g. 30"
                  aria-invalid={badDuration}
                />
              </div>
            </div>
          </div>
          <p className="field-help footnote">
            Scored captions are saved to History. <Link to="/about">How it works</Link>
          </p>
        </form>
      </section>

      <div className="result-col" ref={resultRef}>
        {status === 'idle' && <EmptyResult onExample={tryExample} />}
        {loading && (
          <section className="card" aria-busy="true">
            <p className="muted">Scoring your caption…</p>
            <div className="skeleton" />
            <div className="skeleton short" />
          </section>
        )}
        {status === 'error' && (
          <section className="card error" role="alert">
            <strong>Couldn't score this caption.</strong> {error.message}
            {error.details?.length > 0 && (
              <ul>
                {error.details.map((d) => (
                  <li key={d.field + d.message}>
                    {d.field}: {d.message}
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
        {status === 'success' && data && (
          <ResultPanel result={data} onAddHashtag={addSuggested} addedHashtags={hashtags} />
        )}
      </div>
    </div>
  );
}
