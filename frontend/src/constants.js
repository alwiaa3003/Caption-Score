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
