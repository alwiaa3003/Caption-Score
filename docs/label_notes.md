# Label notes

Code: `ml/src/labels.py` (tested in `ml/tests/test_labels.py`), config `ml/configs/label.yaml`,
build script `python -m src.build_dataset`. Evidence: `ml/reports/label_analysis.json`
(`python -m src.analyze_labels`) and `ml/reports/label_summary.json`.

## Definition
> A video is **viral** if its view count is in the **top 10%** of videos posted in the **same
> calendar month** and in the **same age-at-capture tertile** of that month.

- **Age at capture** = `stats_updated_at − posted_at`. Inside each month, videos are split into
  3 equal-size buckets by this age (0 = captured soonest after posting).
- A group of n videos gets ceil(0.10 · n) positives. Ties at the cut-off are all positive.
- `top_pct` (0.05 / 0.10 / 0.20) and the number of buckets (`age_buckets: 1` = month only)
  are config values.
- All data is **US, English (`language == 'en'`), `type == 'video'`**.

## Rows removed before labelling
| Rule | Why |
|---|---|
| views missing or 0 | no signal; possibly removed videos |
| age at capture < 30 days | views still accumulating |
| empty caption | nothing to learn from |
| `is_ad == 1` (11,258 rows, 1.34%) | paid Spark Ads buy reach. Ads were viral 21.5% of the time vs 9.8% for non-ads, and they cluster in a few months (8.7% of 2026-08, 7–9% of 2025-10/11, against a 0.4% median) |

## Why month × age bucket rather than month only
We first labelled by **month only** and then checked the positive rate inside each month's age
tertiles. If month-only labels were fair, every tertile would be near 10%. Instead:

| Age tertile (within month) | Mean viral rate with month-only labels |
|---|---|
| 0 (captured soonest) | 13.2% |
| 1 | 11.0% |
| 2 (captured latest) | 5.8% |

In **50 of 56 months** at least one tertile fell outside 7–13%, and the largest gap from 10%
was 9.1 points. A month-only label would partly mean "the collector captured this video's
stats early", which has nothing to do with the caption.

**Why age varies inside a month:** each sampled month covers only 2–5 posting days (posting
time has a within-month SD of 7.7 days), but stats were captured in waves between 2025-07 and
2026-10 (capture time has a within-month SD of 108 days). The waves peak in 2025-08/09,
2026-01, 2026-03, 2026-06 and 2026-09. So **age at capture inside a month is ~all capture
timing**. Views can't go down, yet videos captured *sooner* have *more* views (in 2022-01,
tertile 0 median 559 vs tertile 2 median 342). The likely cause is that different capture waves
refreshed different kinds of videos (for example, popular ones first). Labelling within a
bucket means each video is compared only with videos from the same month and the same capture
wave.

**Why equal-size tertiles rather than fixed day ranges:** the age range differs a lot by month
(2022-01: 1,278–1,736 days; 2026-08: 30–63 days), so fixed ranges would leave many buckets
empty or tiny. Tertiles give every group ~4,900 rows (~490 positives), so the 10% cut-off is
stable. With month-only labels the age vs log-views correlation is weak overall (Spearman
−0.09 for videos ≥ 365 days old, +0.03 for videos < 90 days old), but the bucket effect above
is large, which is why we bucket.

## Why 10%
- "Top 10%" is a common, easy-to-explain reading of "went viral" relative to one's peers.
- It gives ~65k positives in train, enough for TF-IDF and fine-tuning, and ~490 per group so
  the cut-off isn't noisy.
- 5% is stricter (closer to the everyday meaning of "viral") but leaves only ~245 positives
  per group. 20% is closer to "did better than average". Both are one config change away and
  are worth comparing in Phase 3.

## Splits (by posting month, inclusive)
| Split | Months | Rows | Positives | Positive rate |
|---|---|---|---|---|
| train | 2022-01 … 2025-08 (44) | 656,136 | 65,676 | 10.01% |
| val | 2025-09 … 2026-04 (8) | 114,813 | 11,496 | 10.01% |
| test | 2026-05 … 2026-07 (3) | 44,104 | 4,415 | 10.01% |
| stress | 2026-08 (1) | 13,689 | 1,371 | 10.02% |

Every month and every age bucket sits at 10.00–10.02% by construction (per-month table in
`label_summary.json`). The test set is used once, at the end. 2026-08 is reported separately.

### The view bar behind the label varies a lot
The minimum views needed to be "viral" ranges from about 1,300 to 23,000 depending on month and
bucket (for example 2022-01: 15,176 / 6,242 / 2,006 across tertiles; 2026-05: 2,154 / 3,573 /
3,738). The pattern flips around 2024-10, when the capture waves change. This is expected: the
label is relative on purpose. But it means "viral" is **not** a fixed number of views, and the
app should say "top 10% of comparable posts", not "will get N views".

## 2026-08 (stress set)
- It is the youngest month: 30–63 days old at capture.
- At matched age (30–63 days), non-ad 2026-08 videos have higher views than 2026-05 to 07:
  median 630 vs 540–582, p90 7,628 vs 2,634–5,068. Ads (8.7% of the month) explained part of
  the original 3× gap and are now excluded.
- The **shape** of the distribution is normal: p90/median = 12.3 against a median of 9.6
  across months (range up to 17.3 in 2022-03).
- **Effect on labels:** the class balance is unchanged (10.02%), because labels are relative.
  The view bar is higher (8,630 / 5,425 / 9,952 by tertile vs 2,154–5,523 in the test months).
  Our best guess is selection bias: very recent videos were probably found more often when they
  were already being recommended, so the pool is more competitive. This is unverified. That is
  why 2026-08 is evaluated separately as a stress set rather than mixed into test.

## Weaknesses and confounds
1. **Creator size is not controlled.** There is no author or follower column. A big account's
   average post beats a small account's best post, so much of what the label measures is
   account size, which a caption cannot reveal. Expect modest scores; captions can only explain
   a small part of virality.
2. **Capture-wave effects.** Bucketing removes the main age effect, but within a bucket,
   capture times still differ by weeks.
3. **Few posting days per month (2–5),** caused by whole-row-group sampling. Posting hour and
   day of week are therefore **not** used as features. Topics tied to specific days (news,
   holidays) may look more or less viral than they really are.
4. **Relative label.** "Top 10% of the month" says nothing about absolute reach, and the view bar
   moves between months.
5. **Language filter.** `un` (undetermined) rows, 52.8% of raw rows, were dropped. Spot checks
   show many are English, but we haven't measured how many.
6. **Survivorship.** Deleted or private videos are absent, and views are a single snapshot.
7. **US-only.** Results may not transfer to other markets, or to Instagram (see README).
