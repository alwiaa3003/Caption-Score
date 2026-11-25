import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from '../App.jsx';
import { ApiError, request } from '../api.js';
import HashtagInput from '../components/HashtagInput.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { EXAMPLE, normaliseTag, percentText, relativeText } from '../constants.js';

const RESULT = {
  probability: 0.2338,
  label: 'likely',
  base_rate: 0.1001,
  likely_threshold: 0.1842,
  top_positive_tokens: [
    { token: '#comedy', weight: 0.51 },
    { token: '😂', weight: 0.3 },
  ],
  top_negative_tokens: [{ token: 'my', weight: -0.2 }],
  tokens: [
    { token: 'wait', weight: 0.01 },
    { token: 'for', weight: 0.0 },
    { token: 'my', weight: -0.2 },
    { token: '😂', weight: 0.3 },
    { token: '#comedy', weight: 0.51 },
  ],
  suggested_hashtags: [{ hashtag: 'skit', lift: 1.8, relevance: 0.09, probability_change: 0.044 }],
  model_version: 'tfidf-lr-v1',
  notes: [],
  saved: true,
};
const SCORE_LINE = /23%\s*chance of the top 10%/;
// The number is its own <span>, so match the whole score paragraph's text.
const scoreLine = (_, el) =>
  el?.classList?.contains('score-line') === true && SCORE_LINE.test(el.textContent);

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const renderApp = (path = '/') =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );

afterEach(() => vi.unstubAllGlobals());

describe('normaliseTag', () => {
  it.each([
    ['#Food', 'food'],
    ['  cooking ', 'cooking'],
    ['##fyp', 'fyp'],
    ['bad tag', ''],
    ['emoji🔥', ''],
    ['', ''],
  ])('%s -> %s', (raw, out) => expect(normaliseTag(raw)).toBe(out));
});

describe('result wording', () => {
  it.each([
    [0.2338, 0.1, '23%', 'About 2.3 times a typical post'],
    [0.2, 0.1, '20%', 'About 2 times a typical post'],
    [0.105, 0.1, '11%', 'About the same as a typical post'],
    [0.05, 0.1, '5%', 'Below a typical post'],
    [0.003, 0.1, 'Under 1%', 'Below a typical post'],
  ])('p=%s base=%s -> %s / %s', (p, base, pct, rel) => {
    expect(percentText(p)).toBe(pct);
    expect(relativeText(p, base)).toBe(rel);
  });
});

describe('HashtagInput', () => {
  function Harness() {
    const [tags, setTags] = useState([]);
    return (
      <>
        <HashtagInput value={tags} onChange={setTags} />
        <output data-testid="tags">{tags.join(',')}</output>
      </>
    );
  }

  it('adds tags on Enter/comma, dedupes, and removes with the × button', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByRole('textbox');
    await user.type(input, '#Food{Enter}cooking,food{Enter}');
    expect(screen.getByTestId('tags')).toHaveTextContent('food,cooking');
    await user.click(screen.getByRole('button', { name: 'Remove #food' }));
    expect(screen.getByTestId('tags')).toHaveTextContent('cooking');
  });

  it('says what is allowed when a tag is invalid', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.type(screen.getByRole('textbox'), 'bad!{Enter}');
    expect(screen.getByRole('alert')).toHaveTextContent('can only use letters, numbers and _');
    expect(screen.getByTestId('tags')).toHaveTextContent('');
  });
});

describe('ResultPanel', () => {
  it('shows the percentage and one plain comparison line, never a view count', () => {
    render(<ResultPanel result={RESULT} />);
    expect(screen.getByText(scoreLine)).toBeInTheDocument();
    expect(screen.getByText('About 2.3 times a typical post')).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/views|viral|likely top/i);
  });

  it('marks words by direction with plain tooltips and offers suggestions', async () => {
    const onAdd = vi.fn();
    const user = userEvent.setup();
    render(<ResultPanel result={RESULT} onAddHashtag={onAdd} />);
    const caption = screen.getByLabelText(/Your caption/);
    expect(within(caption).getByText('#comedy')).toHaveClass('tok-up');
    expect(within(caption).getByText('#comedy')).toHaveAttribute('title', 'Raises the score');
    expect(within(caption).getByText('my')).toHaveClass('tok-down');
    expect(within(caption).getByText('my')).toHaveAttribute('title', 'Lowers the score');
    expect(within(caption).getByText('for')).toHaveClass('tok-flat');
    expect(screen.getByText(/Raised it most:/).parentElement).toHaveTextContent('#comedy, 😂');
    await user.click(screen.getByRole('button', { name: 'Add #skit' }));
    expect(onAdd).toHaveBeenCalledWith('skit');
  });

  it('says so when there are no suggestions', () => {
    render(<ResultPanel result={{ ...RESULT, suggested_hashtags: [] }} />);
    expect(screen.getByText('No suggestions for this caption.')).toBeInTheDocument();
  });
});

describe('Score page', () => {
  it('shows why the Score button is disabled, then enables it', async () => {
    renderApp();
    const button = screen.getByRole('button', { name: 'Score' });
    expect(button).toBeDisabled();
    expect(screen.getByText('Add a caption first')).toBeInTheDocument();
    expect(screen.getByText('0 / 2200')).toBeInTheDocument();
    await userEvent.setup().type(screen.getByLabelText('Caption'), 'hello');
    expect(button).toBeEnabled();
    expect(screen.queryByText('Add a caption first')).not.toBeInTheDocument();
  });

  it('blocks an invalid duration with a reason', async () => {
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'hello');
    await user.type(screen.getByLabelText('Duration (seconds)'), '0');
    expect(screen.getByRole('button', { name: 'Score' })).toBeDisabled();
    expect(screen.getByText('Use a duration from 1 to 3600 seconds')).toBeInTheDocument();
  });

  it('sends caption, hashtags, post type and duration, then shows the result', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(RESULT));
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'wait for my 😂');
    await user.type(screen.getByRole('textbox', { name: /Hashtags/ }), 'comedy{Enter}');
    await user.selectOptions(screen.getByLabelText('Post type'), 'reel');
    await user.type(screen.getByLabelText('Duration (seconds)'), '15');
    await user.click(screen.getByRole('button', { name: 'Score' }));

    expect(await screen.findByText(scoreLine)).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/predict$/);
    expect(JSON.parse(init.body)).toEqual({
      caption: 'wait for my 😂',
      hashtags: ['comedy'],
      post_type: 'reel',
      duration_sec: 15,
    });
  });

  it('"Try an example" fills the form and scores it in one click', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(RESULT));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    expect(screen.getByText('Your result shows here.')).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole('button', { name: 'Try an example' }));
    expect(await screen.findByText(scoreLine)).toBeInTheDocument();
    expect(screen.getByLabelText('Caption').value).toBe(EXAMPLE.caption);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({
      caption: EXAMPLE.caption,
      hashtags: EXAMPLE.hashtags,
    });
  });

  it('scrolls the result into view on narrow screens only', async () => {
    vi.stubGlobal('fetch', async () => jsonResponse(RESULT));
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    const user = userEvent.setup();

    vi.stubGlobal('matchMedia', (q) => ({ matches: false, media: q }));
    const { unmount } = renderApp();
    await user.click(screen.getByRole('button', { name: 'Try an example' }));
    await screen.findByText(scoreLine);
    expect(scroll).toHaveBeenCalledTimes(1);
    unmount();

    scroll.mockClear();
    vi.stubGlobal('matchMedia', (q) => ({ matches: q.includes('min-width'), media: q }));
    renderApp();
    await user.click(screen.getByRole('button', { name: 'Try an example' }));
    await screen.findByText(scoreLine);
    expect(scroll).not.toHaveBeenCalled();
    delete Element.prototype.scrollIntoView;
  });

  it('says what to do when the model is down', async () => {
    vi.stubGlobal('fetch', async () =>
      jsonResponse({ error: 'x', code: 'MODEL_SERVICE_UNAVAILABLE' }, 503),
    );
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'hello');
    await user.click(screen.getByRole('button', { name: 'Score' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      "Couldn't reach the model. Try again in a minute.",
    );
  });

  it('says what to do when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'hello');
    await user.click(screen.getByRole('button', { name: 'Score' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      "Couldn't reach the server. Check that the backend is running",
    );
  });
});

describe('History page', () => {
  it('lists saved predictions', async () => {
    vi.stubGlobal('fetch', async () =>
      jsonResponse({
        items: [
          {
            id: '1',
            caption: 'my caption',
            hashtags: ['dog'],
            probability: 0.2,
            label: 'likely',
            modelVersion: 'tfidf-lr-v1',
            postType: 'video',
            createdAt: '2026-11-26T10:00:00Z',
          },
        ],
      }),
    );
    renderApp('/history');
    expect(await screen.findByText('my caption')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
  });

  it('shows an empty state and an error state', async () => {
    vi.stubGlobal('fetch', async () => jsonResponse({ items: [] }));
    const { unmount } = renderApp('/history');
    expect(await screen.findByText('Nothing scored yet.')).toBeInTheDocument();
    unmount();
    vi.stubGlobal('fetch', async () => jsonResponse({ error: 'History is unavailable' }, 503));
    renderApp('/history');
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't load history.");
  });
});

describe('About page', () => {
  it('explains the project in the first person with the real numbers', async () => {
    vi.stubGlobal('fetch', async () =>
      jsonResponse({
        model_version: 'tfidf-lr-v1',
        base_rate: 0.1001,
        label_cutoff: { val_precision_when_likely: 0.2221 },
        validation: {
          pr_auc: 0.1803,
          positive_rate: 0.1001,
          roc_auc: 0.6698,
          precision_at_top10: 0.222,
        },
        test: {
          pr_auc: 0.1863,
          positive_rate: 0.1001,
          roc_auc: 0.6753,
          precision_at_top10: 0.2299,
        },
      }),
    );
    renderApp('/about');
    expect(await screen.findAllByText(/0\.1803/)).not.toHaveLength(0);
    const text = document.body.textContent;
    expect(text).toMatch(/I built CaptionScore as a student project/);
    expect(text).toMatch(/trained on TikTok/i);
    await waitFor(() => expect(document.body.textContent).toMatch(/22\.2%/));
    expect(document.body.textContent).toMatch(/0\.1863/);
  });
});

describe('copy rules', () => {
  it('has no em dashes on any page', async () => {
    vi.stubGlobal('fetch', async () => jsonResponse({ items: [] }));
    for (const path of ['/', '/history', '/about']) {
      const { unmount } = renderApp(path);
      await waitFor(() => expect(document.body.textContent.length).toBeGreaterThan(0));
      expect(document.body.textContent).not.toMatch(/[—–]/);
      unmount();
    }
    render(<ResultPanel result={RESULT} />);
    expect(document.body.textContent).not.toMatch(/[—–]/);
  });
});

describe('api.request', () => {
  it('maps 429 to a friendly message', async () => {
    await expect(
      request('/x', { fetchImpl: async () => jsonResponse({ error: 'Too many' }, 429) }),
    ).rejects.toMatchObject({ status: 429, message: expect.stringMatching(/Wait a minute/) });
  });

  it('keeps validation details from the backend', async () => {
    const err = await request('/x', {
      fetchImpl: async () =>
        jsonResponse(
          { error: 'Invalid input', details: [{ field: 'caption', message: 'bad' }] },
          400,
        ),
    }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.details).toEqual([{ field: 'caption', message: 'bad' }]);
  });
});
