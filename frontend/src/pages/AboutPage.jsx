import { useEffect } from 'react';

import { api } from '../api.js';
import { useAsync } from '../useAsync.js';

const f4 = (x) => (typeof x === 'number' ? x.toFixed(4) : 'not available');
const p1 = (x) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : 'not available');

/** The real numbers, read from the model card so this page can't drift from the model. */
function Results() {
  const { status, data, error, run } = useAsync();
  useEffect(() => {
    run(() => api.modelInfo());
  }, [run]);

  if (status === 'loading' || status === 'idle')
    return <p className="muted">Loading the numbers</p>;
  if (status === 'error')
    return (
      <div className="error-line" role="alert">
        <p>Couldn't load the numbers. {error.message}</p>
      </div>
    );
  const val = data.validation ?? {};
  const test = data.test ?? {};
  const cut = data.label_cutoff ?? {};
  return (
    <>
      <p>
        I measured it on posts the model never saw during training. On posts from September 2025 to
        April 2026 its PR-AUC was {f4(val.pr_auc)}. Random guessing gets {f4(val.positive_rate)}. On
        May to July 2026, which I kept aside and checked only once at the end, it got{' '}
        {f4(test.pr_auc)}.
      </p>
      <p>
        In plain terms: of the captions the model ranked in its top 10%,{' '}
        {p1(cut.val_precision_when_likely)} really were top 10%. For a random post it's{' '}
        {p1(data.base_rate)}. So the caption carries a real signal, but a small one.
      </p>
      <table className="metrics">
        <thead>
          <tr>
            <th scope="col">Posts</th>
            <th scope="col">PR-AUC</th>
            <th scope="col">Random</th>
            <th scope="col">ROC-AUC</th>
            <th scope="col">Hit rate in top 10%</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Sep 2025 to Apr 2026</th>
            <td>{f4(val.pr_auc)}</td>
            <td>{f4(val.positive_rate)}</td>
            <td>{f4(val.roc_auc)}</td>
            <td>{p1(val.precision_at_top10)}</td>
          </tr>
          <tr>
            <th scope="row">May to Jul 2026</th>
            <td>{f4(test.pr_auc)}</td>
            <td>{f4(test.positive_rate)}</td>
            <td>{f4(test.roc_auc)}</td>
            <td>{p1(test.precision_at_top10)}</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}

export default function AboutPage() {
  return (
    <section className="page-narrow prose">
      <h1>About</h1>
      <p>
        I built CaptionScore as a student project. It reads the caption of a short video and
        estimates the chance that the post ends up in the top 10% of similar posts by views.
      </p>

      <h2>What it learned from</h2>
      <p>
        I trained it on public data about 656,136 US, English-language TikTok videos posted between
        January 2022 and August 2025, from a dataset on Hugging Face. It only sees the data, never
        the videos.
      </p>
      <p>
        A video counts as top 10% if its views were in the top 10% of videos posted in the same
        month and checked at a similar age. Comparing within a month keeps older videos, which had
        more time to collect views, from looking better. I left out paid ads.
      </p>

      <h2>How it works</h2>
      <p>
        The model is a logistic regression on the words, word pairs and character patterns in the
        caption, including emojis and hashtags. Because it's a linear model, I can show exactly how
        much each word moved the score. Those are the highlighted words.
      </p>
      <p>
        Hashtag suggestions are tags that showed up on top posts more often and that also raise your
        caption's score. Post type and duration are saved with your caption but don't change the
        score.
      </p>

      <h2>How well it works</h2>
      <Results />

      <h2>Limits</h2>
      <ul>
        <li>It was trained on TikTok. Using it for Instagram is a guess I haven't tested.</li>
        <li>
          Most of what makes a video do well isn't in the caption: the video itself, timing, and
          above all how many followers the creator has. The data has no follower counts.
        </li>
        <li>
          Words and hashtags that scored well were linked to top posts in the past. Adding them
          doesn't guarantee anything.
        </li>
        <li>
          Captions you score here are saved so they show up in History. I don't store accounts or IP
          addresses, and I never show captions from the training data.
        </li>
      </ul>
    </section>
  );
}
