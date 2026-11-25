// Light is the default. The visitor's choice is remembered in localStorage when available;
// if storage is blocked (private mode, strict settings) the page simply stays light.
const KEY = 'captionscore-theme';

export function getTheme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* storage unavailable: the choice lasts for this page view only */
  }
}
