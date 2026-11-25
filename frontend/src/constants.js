// Shared constants and helpers (kept out of component files so React fast refresh works).
export const MAX_CAPTION = 2200;
export const MAX_HASHTAGS = 30;

export const VERDICT = {
  likely: 'Likely top 10% of comparable posts',
  unlikely: 'Unlikely to reach the top 10% of comparable posts',
};

const VALID_TAG = /^\w{1,100}$/u;

/** Normalise user input like "#Food" or " food " to "food"; returns '' if invalid. */
export function normaliseTag(raw) {
  const t = raw.trim().replace(/^#+/, '').toLowerCase();
  return VALID_TAG.test(t) ? t : '';
}

// Word-highlight tint range (share of the up/down colour mixed into the background).
// MAX_TINT is capped so text on the strongest tint keeps >= 4.5:1 contrast in both themes
// (checked in src/test/contrast.test.js).
export const MIN_TINT = 0.12;
export const MAX_TINT = 0.45;

// Hand-written example (never a caption from the training dataset).
export const EXAMPLE = {
  caption: '3 ingredient pasta you can make in 10 minutes 🍝 wait for the end',
  hashtags: ['recipe', 'pasta'],
};
