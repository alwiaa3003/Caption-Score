import { useEffect } from 'react';

import { api } from '../api.js';
import { useAsync } from '../useAsync.js';

const f4 = (x) => (typeof x === 'number' ? x.toFixed(4) : '—');
const p1 = (x) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : '—');

/** Model performance, read live from the model card so the page never drifts from the model. */
function Performance() {
  const { status, data, error, run } = useAsync();
  useEffect(() => {
    run(() => api.modelInfo());
  }, [run]);

  if (status === 'loading' || status === 'idle')
    return <p className="muted">Loading model details…</p>;
  if (status === 'error')
    return (
      <p className="error" role="alert">
        Couldn't load the model's numbers: {error.message}
      </p>
    );
  const val = data.validation ?? {};
  const test = data.test ?? {};
  const cut = data.label_cutoff ?? {};
  return (
    <>
      <p>
        On held-out validation posts, the model's PR-AUC is <strong>{f4(val.pr_auc)}</strong>,
        against <strong>{f4(val.positive_rate)}</strong> for random guessing (the share of posts
        that are top 10%). On the later test months it scored <strong>{f4(test.pr_auc)}</strong>.
        That is a real but <strong>weak</strong> signal: among captions the model ranks in its top
        10%, about {p1(cut.val_precision_when_likely)} actually reached the top 10%, versus{' '}
        {p1(data.base_rate)} for a typical post.
      </p>
      <table className="metrics">
        <thead>
          <tr>
            <th scope="col">Data</th>
            <th scope="col">PR-AUC</th>
            <th scope="col">Chance</th>
            <th scope="col">ROC-AUC</th>
            <th scope="col">Precision in top 10%</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Validation (2025-09 – 2026-04)</th>
            <td>{f4(val.pr_auc)}</td>
            <td>{f4(val.positive_rate)}</td>
            <td>{f4(val.roc_auc)}</td>
            <td>{f4(val.precision_at_top10)}</td>
          </tr>
          <tr>
            <th scope="row">Test (2026-05 – 2026-07)</th>
            <td>{f4(test.pr_auc)}</td>
            <td>{f4(test.positive_rate)}</td>
            <td>{f4(test.roc_auc)}</td>
            <td>{f4(test.precision_at_top10)}</td>
          </tr>
        </tbody>
      </table>
      <p className="muted small">
        Model {data.model_version}: {data.model}.
      </p>
    </>
  );
}

export default function AboutPage() {
  return (
    <section className="card prose">
      <h1>About CaptionScore</h1>

      <h2>What it does</h2>
      <p>
        CaptionScore reads a short-video caption and its hashtags and estimates how likely the post
        is to land in the <strong>top 10% of comparable posts</strong>. It highlights the words that
        push that estimate up or down, and suggests related hashtags. It does not predict view
        counts.
      </p>

      <h2>How it works</h2>
      <ul>
        <li>
          Paste a caption (and optionally hashtags). The model reads the <strong>text only</strong>:
          words, short word pairs, character patterns, emojis and hashtags.
        </li>
        <li>
          It returns an estimated chance of reaching the top 10% of comparable posts. The gauge
          marks a typical post (about 10%) and the "likely" cut-off: a score in the model's top 10%
          of scores on held-out posts.
        </li>
        <li>
          Highlighted words show what drove the score. The model is linear, so each word's
          contribution is exact, not an approximation: blue (solid underline) pushes the score up,
          orange (dashed underline) pushes it down.
        </li>
        <li>
          Suggested hashtags are related tags whose posts reached the top 10% more often in the
          training data, and that raise this caption's score when added.
        </li>
        <li>
          Post type and duration are recorded with your entry but don't change the score: the model
          was trained on caption text alone.
        </li>
      </ul>

      <h2>How "top 10%" is defined</h2>
      <p>
        The model was trained on public metadata of US, English-language TikTok videos posted
        between 2022 and 2025. A video counts as top 10% if its views were in the top 10% of videos
        posted in the same month <em>and</em> captured at a similar age (so a video isn't compared
        with ones that had months longer to collect views). Paid ads were excluded.
      </p>

      <h2>How well it works</h2>
      <Performance />

      <h2>Limits: please read</h2>
      <ul>
        <li>
          <strong>Trained on TikTok.</strong> Using it for Instagram Reels or other platforms is an
          experiment; it hasn't been validated there.
        </li>
        <li>
          <strong>Captions are a weak signal.</strong> Most of what makes a video take off isn't in
          the caption: the video itself, timing, and above all the creator's audience size, which
          the training data doesn't include.
        </li>
        <li>
          <strong>Correlation, not causation.</strong> Words and hashtags that score well were
          associated with top posts in the past. Adding them won't guarantee anything.
        </li>
        <li>
          <strong>Privacy.</strong> Captions you score are stored (with the score and time) to show
          on the History page. No account, IP address or other personal data is stored. Captions
          from the training dataset are never shown on this site.
        </li>
      </ul>
    </section>
  );
}
