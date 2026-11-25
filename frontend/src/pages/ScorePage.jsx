import { useState } from 'react';

import { api } from '../api.js';
import HashtagInput from '../components/HashtagInput.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { MAX_CAPTION, MAX_HASHTAGS } from '../constants.js';
import { useAsync } from '../useAsync.js';

export default function ScorePage() {
  const [caption, setCaption] = useState('');
  const [hashtags, setHashtags] = useState([]);
  const [postType, setPostType] = useState('video');
  const { status, data, error, run } = useAsync();

  const trimmed = caption.trim();
  const tooLong = caption.length > MAX_CAPTION;
  const canSubmit = trimmed.length > 0 && !tooLong && status !== 'loading';

  function submit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    run(() => api.predict({ caption, hashtags, post_type: postType }));
  }

  function addSuggested(tag) {
    if (!hashtags.includes(tag) && hashtags.length < MAX_HASHTAGS) setHashtags([...hashtags, tag]);
  }

  return (
    <>
      <section className="card">
        <h1>Score a caption</h1>
        <p className="muted">
          Paste a short-video caption. The model estimates how likely it is to land in the top 10%
          of comparable posts, and shows which words push that up or down.
        </p>
        <form onSubmit={submit} noValidate>
          <label htmlFor="caption">Caption</label>
          <textarea
            id="caption"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            rows={5}
            placeholder="e.g. 3 ingredient pasta you can make in 10 minutes 🍝 #recipe"
            aria-invalid={tooLong}
            aria-describedby="caption-help"
          />
          <p id="caption-help" className={tooLong ? 'field-error' : 'field-help'}>
            {caption.length}/{MAX_CAPTION} characters
            {tooLong && ' (too long)'}
          </p>

          <label htmlFor="hashtags">Hashtags</label>
          <HashtagInput value={hashtags} onChange={setHashtags} />

          <div className="row">
            <div>
              <label htmlFor="post-type">Post type</label>
              <select id="post-type" value={postType} onChange={(e) => setPostType(e.target.value)}>
                <option value="video">Video</option>
                <option value="reel">Reel</option>
                <option value="photo">Photo</option>
                <option value="other">Other</option>
              </select>
            </div>
            <button type="submit" className="primary" disabled={!canSubmit}>
              {status === 'loading' ? 'Scoring…' : 'Score it'}
            </button>
          </div>
          <p className="muted small">
            The model reads text only; post type is recorded but doesn't change the score. Captions
            you score are saved to the History page, so don't include personal information.
          </p>
        </form>
      </section>

      {status === 'loading' && (
        <section className="card" aria-busy="true">
          <div className="skeleton" />
          <p className="muted">Scoring your caption…</p>
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
    </>
  );
}
