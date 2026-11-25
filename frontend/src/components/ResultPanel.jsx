import { Fragment } from 'react';

import { MAX_TINT, MIN_TINT, percentText, relativeText } from '../constants.js';

// Only words that moved the score at least this much, relative to the strongest word, are
// marked. Smaller effects stay plain so the caption doesn't turn into a wall of highlights.
const MIN_SHARE = 0.2;

const TIP = { up: 'Raises the score', down: 'Lowers the score' };

/** The caption with each word tinted by how much it moved the score. */
function HighlightedCaption({ tokens }) {
  const max = Math.max(1e-9, ...tokens.map((t) => Math.abs(t.weight)));
  return (
    <p className="highlighted" aria-label="Your caption, with words marked by their effect">
      {tokens.map((t, i) => {
        const strength = Math.abs(t.weight) / max;
        const dir = strength < MIN_SHARE ? 'flat' : t.weight > 0 ? 'up' : 'down';
        // A real space between tokens gives the browser line-break points (and makes the
        // caption copy-paste correctly); without it the whole caption is one unbreakable word.
        return (
          <Fragment key={i}>
            {i > 0 && ' '}
            <span
              className={`tok tok-${dir}`}
              style={
                dir === 'flat'
                  ? undefined
                  : { '--strength': (MIN_TINT + (MAX_TINT - MIN_TINT) * strength).toFixed(2) }
              }
              title={TIP[dir]}
            >
              {t.token}
            </span>
          </Fragment>
        );
      })}
    </p>
  );
}

function WordList({ label, items, max }) {
  const words = items.filter((t) => Math.abs(t.weight) >= MIN_SHARE * max).map((t) => t.token);
  if (!words.length) return null;
  return (
    <p className="word-list">
      <span className="muted">{label}</span> {words.join(', ')}
    </p>
  );
}

export default function ResultPanel({ result, onAddHashtag, addedHashtags = [] }) {
  const {
    probability,
    base_rate: baseRate,
    tokens = [],
    top_positive_tokens: up = [],
    top_negative_tokens: down = [],
    suggested_hashtags: suggestions = [],
    model_version: modelVersion,
    saved,
  } = result;

  const max = Math.max(1e-9, ...tokens.map((t) => Math.abs(t.weight)));

  return (
    <section className="result" aria-live="polite">
      <p className="score-line">
        <span className="score-number">{percentText(probability)}</span> chance of the top 10%
      </p>
      <p className="relative-line">{relativeText(probability, baseRate)}</p>

      <HighlightedCaption tokens={tokens} />
      <p className="key small muted">
        <span className="key-up">Solid underline</span> raises the score.{' '}
        <span className="key-down">Dashed underline</span> lowers it.
      </p>

      <WordList label="Raised it most:" items={up} max={max} />
      <WordList label="Lowered it most:" items={down} max={max} />

      <h2 className="section-title">Hashtags you could add</h2>
      {suggestions.length ? (
        <>
          <ul className="suggestions">
            {suggestions.map((s) => {
              const added = addedHashtags.includes(s.hashtag);
              return (
                <li key={s.hashtag}>
                  <button
                    type="button"
                    className="chip chip-button"
                    onClick={() => onAddHashtag?.(s.hashtag)}
                    disabled={added}
                    aria-label={`Add #${s.hashtag}`}
                  >
                    {added ? 'Added' : '+'} #{s.hashtag}
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="small muted">
            These tags showed up on top posts more often. That's a pattern, not a guarantee.
          </p>
        </>
      ) : (
        <p className="muted">No suggestions for this caption.</p>
      )}

      <p className="result-foot small muted">
        Model {modelVersion}
        {saved === false && '. Not saved to History.'}
      </p>
    </section>
  );
}
