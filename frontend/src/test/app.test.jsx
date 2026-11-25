import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from '../App.jsx';
import { ApiError, request } from '../api.js';
import HashtagInput from '../components/HashtagInput.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { normaliseTag, VERDICT } from '../constants.js';

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

  it('shows an error for invalid tags', async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.type(screen.getByRole('textbox'), 'bad!{Enter}');
    expect(screen.getByRole('alert')).toHaveTextContent("isn't a valid hashtag");
    expect(screen.getByTestId('tags')).toHaveTextContent('');
  });
});

describe('ResultPanel', () => {
  it('uses "top 10% of comparable posts" wording and never promises views', () => {
    render(<ResultPanel result={RESULT} />);
    expect(screen.getByText(VERDICT.likely)).toBeInTheDocument();
    expect(screen.getByText(/23\.4%/, { selector: 'strong' })).toBeInTheDocument();
    expect(document.body.textContent).toMatch(/not a prediction of views/);
    expect(document.body.textContent).not.toMatch(
      /will get \d|you will go viral|guarantee(d)? views/i,
    );
  });

  it('highlights tokens by direction and offers suggestions', async () => {
    const onAdd = vi.fn();
    const user = userEvent.setup();
    render(<ResultPanel result={RESULT} onAddHashtag={onAdd} />);
    const caption = screen.getByLabelText(/Your caption/);
    expect(within(caption).getByText('#comedy')).toHaveClass('tok-up');
    expect(within(caption).getByText('my')).toHaveClass('tok-down');
    expect(within(caption).getByText('for')).toHaveClass('tok-flat');
    await user.click(screen.getByRole('button', { name: 'Add #skit' }));
    expect(onAdd).toHaveBeenCalledWith('skit');
  });

  it('says so when there are no suggestions', () => {
    render(<ResultPanel result={{ ...RESULT, suggested_hashtags: [] }} />);
    expect(screen.getByText(/No confident suggestions/)).toBeInTheDocument();
  });
});

describe('Score page', () => {
  it('disables the button until there is a caption', async () => {
    renderApp();
    const button = screen.getByRole('button', { name: 'Score it' });
    expect(button).toBeDisabled();
    await userEvent.setup().type(screen.getByLabelText('Caption'), 'hello');
    expect(button).toBeEnabled();
  });

  it('sends the caption, hashtags and post type, then shows the result', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(RESULT));
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'wait for my 😂');
    await user.type(screen.getByRole('textbox', { name: 'Hashtags' }), 'comedy{Enter}');
    await user.selectOptions(screen.getByLabelText('Post type'), 'reel');
    await user.click(screen.getByRole('button', { name: 'Score it' }));

    expect(await screen.findByText(VERDICT.likely)).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/predict$/);
    expect(JSON.parse(init.body)).toEqual({
      caption: 'wait for my 😂',
      hashtags: ['comedy'],
      post_type: 'reel',
    });
  });

  it('shows a clear error when the model service is down', async () => {
    vi.stubGlobal('fetch', async () =>
      jsonResponse({ error: 'x', code: 'MODEL_SERVICE_UNAVAILABLE' }, 503),
    );
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'hello');
    await user.click(screen.getByRole('button', { name: 'Score it' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The scoring model is offline');
  });

  it('shows a clear error when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText('Caption'), 'hello');
    await user.click(screen.getByRole('button', { name: 'Score it' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      "Can't reach the CaptionScore server",
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
    expect(await screen.findByText(/Nothing here yet/)).toBeInTheDocument();
    unmount();
    vi.stubGlobal('fetch', async () => jsonResponse({ error: 'History is unavailable' }, 503));
    renderApp('/history');
    expect(await screen.findByRole('alert')).toHaveTextContent('History is unavailable');
  });
});

describe('About page', () => {
  it('shows the real validation PR-AUC from the model card next to chance', async () => {
    vi.stubGlobal('fetch', async () =>
      jsonResponse({
        model_version: 'tfidf-lr-v1',
        model: 'TF-IDF + LR',
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
    expect(await screen.findAllByText('0.1803')).not.toHaveLength(0);
    expect(document.body.textContent).toMatch(/Trained on TikTok/);
    expect(document.body.textContent).toMatch(/weak/);
    await waitFor(() => expect(document.body.textContent).toMatch(/22\.2%/));
  });
});

describe('api.request', () => {
  it('maps 429 to a friendly message', async () => {
    await expect(
      request('/x', { fetchImpl: async () => jsonResponse({ error: 'Too many' }, 429) }),
    ).rejects.toMatchObject({ status: 429, message: expect.stringMatching(/wait a minute/) });
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
