// Shared constants and helpers (kept out of component files so React fast refresh works).
export const MAX_CAPTION = 2200;
export const MAX_HASHTAGS = 30;

const VALID_TAG = /^\w{1,100}$/u;

/** Normalise user input like "#Food" or " food " to "food"; returns '' if invalid. */
export function normaliseTag(raw) {
  const t = raw.trim().replace(/^#+/, '').toLowerCase();
  return VALID_TAG.test(t) ? t : '';
}

// Word-highlight tint range (share of the raise/lower colour mixed into the background).
// MAX_TINT is capped so text on the strongest tint keeps >= 4.5:1 contrast in both themes
// (checked in src/test/contrast.test.js).
export const MIN_TINT = 0.1;
export const MAX_TINT = 0.35;

// Hand-written example (never a caption from the training dataset).
export const EXAMPLE = {
  caption: '3 ingredient pasta you can make in 10 minutes 🍝 wait for the end',
  hashtags: ['recipe', 'pasta'],
};

/** "23%" (or "Under 1%"). */
export function percentText(p) {
  return p < 0.005 ? 'Under 1%' : `${Math.round(p * 100)}%`;
}

/** One plain line comparing the score with a typical post (the base rate, about 10%). */
export function relativeText(p, baseRate) {
  const ratio = p / baseRate;
  if (ratio >= 1.15) {
    const x = ratio.toFixed(1).replace(/\.0$/, '');
    return `About ${x} times a typical post`;
  }
  if (ratio >= 0.85) return 'About the same as a typical post';
  return 'Below a typical post';
}
