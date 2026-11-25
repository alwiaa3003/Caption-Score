import { Fragment } from 'react';

import { MAX_TINT, MIN_TINT, VERDICT } from '../constants.js';
import ScoreGauge from './ScoreGauge.jsx';

const pct = (p) => `${Math.round(p * 1000) / 10}%`;

// Contributions this small round to ±0.00 and are shown as neutral in the highlight.
const NEGLIGIBLE = 0.005;

/** Caption tokens with a background whose strength follows each token's contribution. */
function HighlightedCaption({ tokens }) {
  const max = Math.max(1e-9, ...tokens.map((t) => Math.abs(t.weight)));
  return (
    <p className="highlighted" aria-label="Your caption, with words coloured by their effect">
      {tokens.map((t, i) => {
        const strength = Math.abs(t.weight) / max;
        const dir = t.weight >= NEGLIGIBLE ? 'up' : t.weight <= -NEGLIGIBLE ? 'down' : 'flat';
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
              title={`${t.weight > 0 ? '+' : ''}${t.weight.toFixed(3)}`}
            >
              {t.token}
            </span>
          </Fragment>
        );
      })}
    </p>
  );
}

function TokenList({ title, items: all, dir }) {
  const items = all.filter((t) => Math.abs(t.weight) >= NEGLIGIBLE);
  return (
    <div className="token-list">
      <h4>
        <span className={`legend-swatch tok-${dir}`} aria-hidden="true" /> {title}
      </h4>
      {items.length ? (
        <ul>
          {items.map((t) => (
            <li key={t.token}>
              <span className="mono">{t.token}</span>
              <span className="muted">
                {t.weight > 0 ? '+' : ''}
                {t.weight.toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">None</p>
      )}
    </div>
  );
}

export default function ResultPanel({ result, onAddHashtag, addedHashtags = [] }) {
  const {
    probability,
    label,
    base_rate: baseRate,
    likely_threshold: threshold,
    tokens = [],
    top_positive_tokens: up = [],
    top_negative_tokens: down = [],
    suggested_hashtags: suggestions = [],
    model_version: modelVersion,
    notes = [],
    saved,
  } = result;
  const ratio = probability / baseRate;

  return (
    <section className="card result" aria-live="polite">
      <div className="result-head">
        <ScoreGauge probability={probability} baseRate={baseRate} threshold={threshold} />
        <div>
          <p className={`verdict verdict-${label}`}>{VERDICT[label]}</p>
          <p className="muted">
            Estimated chance: <strong>{pct(probability)}</strong> vs {pct(baseRate)} for a typical
            post ({ratio.toFixed(1)}× typical). This is a ranking signal from the caption text only,
            not a prediction of views.
          </p>
        </div>
      </div>

      <h3>How the model read your caption</h3>
      <div className="legend" aria-hidden="true">
        <span>
          <span className="legend-swatch tok-up" /> pushes the score up
        </span>
        <span>
          <span className="legend-swatch tok-down" /> pushes it down
        </span>
      </div>
      <HighlightedCaption tokens={tokens} />

      <div className="token-lists">
        <TokenList title="Pushing up" items={up} dir="up" />
        <TokenList title="Pushing down" items={down} dir="down" />
      </div>

      <h3>Suggested hashtags</h3>
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
                    {added ? '✓' : '+'} #{s.hashtag}
                  </button>
                  <span className="muted small">
                    {s.lift.toFixed(1)}× top-10% rate · +{(s.probability_change * 100).toFixed(1)}{' '}
                    pts
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="muted small">
            Related hashtags whose posts reached the top 10% more often in the training data. That's
            an association, not a guarantee: add them only if they fit your post.
          </p>
        </>
      ) : (
        <p className="muted">No confident suggestions for this caption.</p>
      )}

      <footer className="result-foot muted small">
        Model {modelVersion}
        {saved === false && ' · not saved to history (database unavailable)'}
        {notes.map((n) => (
          <span key={n}> · {n}</span>
        ))}
      </footer>
    </section>
  );
}
