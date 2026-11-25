import { useState } from 'react';

import { MAX_HASHTAGS, normaliseTag } from '../constants.js';

/**
 * Hashtag chips. Enter, comma or space adds a tag; Backspace on an empty field removes the last.
 * Pasting "#a #b, c" adds several at once.
 */
export default function HashtagInput({ value, onChange, id = 'hashtags' }) {
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');

  function add(text) {
    const parts = text.split(/[\s,]+/).filter(Boolean);
    let next = [...value];
    let rejected = '';
    for (const p of parts) {
      const tag = normaliseTag(p);
      if (!tag) rejected = p;
      else if (!next.includes(tag) && next.length < MAX_HASHTAGS) next = [...next, tag];
    }
    if (next.length >= MAX_HASHTAGS && parts.length) setError(`Up to ${MAX_HASHTAGS} hashtags.`);
    else
      setError(
        rejected ? `"${rejected}" isn't a valid hashtag (letters, numbers and _ only).` : '',
      );
    onChange(next);
    setDraft('');
  }

  function onKeyDown(e) {
    if (['Enter', ',', ' '].includes(e.key)) {
      if (draft.trim()) {
        e.preventDefault();
        add(draft);
      } else if (e.key === 'Enter') {
        e.preventDefault();
      }
    } else if (e.key === 'Backspace' && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  }

  return (
    <div>
      <div className="chips-field" onClick={() => document.getElementById(id)?.focus()}>
        {value.map((tag) => (
          <span key={tag} className="chip">
            #{tag}
            <button
              type="button"
              className="chip-remove"
              aria-label={`Remove #${tag}`}
              onClick={() => onChange(value.filter((t) => t !== tag))}
            >
              ×
            </button>
          </span>
        ))}
        <input
          id={id}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          onBlur={() => draft.trim() && add(draft)}
          onPaste={(e) => {
            const text = e.clipboardData.getData('text');
            if (/[\s,]/.test(text)) {
              e.preventDefault();
              add(text);
            }
          }}
          placeholder={value.length ? '' : 'Add hashtags, e.g. cooking'}
          aria-describedby={`${id}-help`}
          disabled={value.length >= MAX_HASHTAGS}
        />
      </div>
      <p
        id={`${id}-help`}
        className={error ? 'field-error' : 'field-help'}
        role={error ? 'alert' : undefined}
      >
        {error || 'Press Enter or comma to add. Hashtags already in your caption count too.'}
      </p>
    </div>
  );
}
