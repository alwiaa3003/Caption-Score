// WCAG contrast checks for both themes, computed from the real tokens in styles.css.
// Text needs >= 4.5:1 (AA, normal size); non-text marks (underlines, gauge, borders that carry
// meaning) need >= 3:1.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { MAX_TINT } from '../constants.js';

// Vitest runs from the frontend/ folder (import.meta.url isn't a file: URL under jsdom).
const css = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8');

function tokens(block) {
  const out = {};
  for (const m of block.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\b/g)) out[m[1]] = m[2];
  return out;
}

const light = tokens(
  css.slice(css.indexOf(':root {'), css.indexOf('@media (prefers-color-scheme: dark)')),
);
const darkStart = css.indexOf('@media (prefers-color-scheme: dark)');
const dark = {
  ...light,
  ...tokens(css.slice(darkStart, css.indexOf('color-scheme: dark', darkStart))),
};

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const lum = (hex) => {
  const [r, g, b] = rgb(hex).map(lin);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
function contrast(a, b) {
  const [x, y] = [lum(a), lum(b)];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
/** sRGB mix like CSS color-mix(in srgb, a share, b). */
function mix(a, b, share) {
  const [pa, pb] = [rgb(a), rgb(b)];
  return (
    '#' +
    pa
      .map((v, i) =>
        Math.round((v * share + pb[i] * (1 - share)) * 255)
          .toString(16)
          .padStart(2, '0'),
      )
      .join('')
  );
}

const TEXT = [
  ['ink', 'bg'],
  ['ink', 'surface'],
  ['muted', 'bg'], // helper text, counter, empty line, placeholder on the page
  ['muted', 'surface'], // placeholder inside fields
  ['link', 'bg'], // links and text buttons
  ['link', 'surface'],
  ['accent-ink', 'accent'], // primary button
  ['accent-ink', 'accent-hover'],
  ['chip-ink', 'chip-bg'], // hashtag chips
  ['ink', 'chip-bg'],
  ['disabled-ink', 'disabled-bg'], // primary button while blocked
  ['error', 'bg'], // error lines
];
const NON_TEXT = [
  ['control', 'bg'], // input outlines (WCAG 1.4.11)
  ['control', 'surface'],
  ['focus', 'bg'], // keyboard focus ring
  ['focus', 'surface'],
  ['raise', 'bg'], // highlight underlines
  ['lower', 'bg'],
  ['raise', 'surface'],
  ['lower', 'surface'],
  ['accent', 'bg'], // primary button shape against the page
];

describe.each([
  ['light', light],
  ['dark', dark],
])('%s theme', (_, t) => {
  it('defines every token it needs', () => {
    for (const [a, b] of [...TEXT, ...NON_TEXT]) {
      expect(t[a], a).toMatch(/^#/);
      expect(t[b], b).toMatch(/^#/);
    }
  });

  it.each(TEXT)('text %s on %s >= 4.5:1', (fg, bg) => {
    expect(contrast(t[fg], t[bg])).toBeGreaterThanOrEqual(4.5);
  });

  it.each(NON_TEXT)('mark %s on %s >= 3:1', (fg, bg) => {
    expect(contrast(t[fg], t[bg])).toBeGreaterThanOrEqual(3);
  });

  it.each(['raise', 'lower'])('text on the strongest %s highlight >= 4.5:1', (dir) => {
    expect(contrast(t.ink, mix(t[dir], t.bg, MAX_TINT))).toBeGreaterThanOrEqual(4.5);
  });
});
