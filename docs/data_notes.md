# Data notes

All numbers below were computed by the scripts named next to them. Raw files:
`ml/reports/sampling_stats.json` (raw rows read, before filtering) and
`ml/reports/sample_profile.json` (the cleaned sample). Neither file is hand-edited.

## Source
- Hugging Face `datasocial/tiktok-5.6B-videos`, revision `a466d033`. Public, not gated.
- License **CC BY-NC 4.0**: non-commercial use only. The sample in `ml/data/` is gitignored
  and is never redistributed.
- The rows are video **metadata** (captions, counts, timestamps), not videos. The dataset card
  says it was collected through TikTok's private mobile API. See the ethics section of the README.
- Size: 5,598,141,717 rows, about 460 GB of compressed Parquet. **We never download it in
  full.**

## Schema (verified with `datasets` streaming: `python -m src.inspect_schema`)
| Column | Type | Notes |
|---|---|---|
| `video_id` | uint64 | |
| `posted_at` | timestamp[ms, UTC] | |
| `country` | string | region TikTok assigns; `''` occurs |
| `language` | string | `un` = undetermined |
| `type` | string | `video` or `photo` |
| `duration_ms` | uint32 | 0 on photos |
| `image_count` | uint8 | photo posts |
| `width`, `height` | uint16 | 0 in many rows |
| `caption` | string | |
| `hashtags`, `hashtag_ids` | list | lower-case, parallel arrays |
| `mentions` | list[uint64] | IDs of **mentioned** creators, not the author |
| `on_screen_text` | list[string] | text stickers |
| `sound_id` | uint64 | |
| `is_ad`, `is_shop_video`, `sound_muted` | uint8 | |
| `branded_content` | string | `paid_partnership`, `shop_affiliate` or empty |
| `product_id`, `seller_id` | uint64 | |
| `is_ai_generated`, `not_recommended_for_fyp` | uint8, nullable | |
| `views`, `likes`, `comments`, `shares`, `saves`, `downloads` | uint64 | as of `stats_updated_at` |
| `engagement_rate` | float32 | |
| `stats_updated_at` | timestamp[ms, UTC] | when the counts were captured |

**There is no creator ID or follower count column.** Creator size, the main driver of views,
cannot be controlled for. This is the biggest limitation of the project (see `label_notes.md`).

## Physical layout (verified from Parquet footers)
- There is one file per calendar month: `data/2014-07.parquet` … `data/2026-10.parquet`
  (148 files). They grow from a few MB (2014–2017) to 10–22 GB (2025–2026). 2026-10 is a
  partial month.
- Files were written by ClickHouse with ZSTD compression and contain row groups of
  1,046,544 rows (the last row group of each file is smaller).
- **Each file is sorted by `(country, posted_at)`.** In 2024-03, row groups 8–47 are all US,
  each covering about one day. The other row groups contain the other countries and each spans
  the whole month.
- So "take the first N rows" would give 2014 (whole dataset) or the first country and day
  (one file). We avoid both.

## Sampling method (`python -m src.sample_data`, config `ml/configs/sample.yaml`)
1. **Window:** 2022-01 to 2026-08 (56 months). Earlier data is thin and partly archival, with
   zeroed fields and rounded durations. Later months are too young to have settled views.
2. **Row groups:** pyarrow reads whole row groups. For each month we split the file's row
   groups into 2 equal blocks and draw 1 at random from each (seed 42). Every row group has the
   same chance of being picked, so this is an unbiased **cluster** sample of each month.
3. **Column projection:** we read only 13 columns. We skip `hashtags`, because a pilot on 2
   row groups showed that 99.1% and 99.4% of rows with hashtags already contain every tag as
   `#tag` in the caption. We extract tags from the caption by regex instead, which saves about
   18 MB per row group.
4. **Download:** 4.95 GB over HTTP range requests, approved before running. It is streamed
   through memory and not stored.
5. **Cleaning** (`ml/src/cleaning.py`, tested in `ml/tests/test_cleaning.py`):
   `language == 'en'`, `type == 'video'`, non-empty caption, `views > 0`, and age at capture
   (`stats_updated_at − posted_at`) ≥ 30 days.
6. **Subsample:** a seeded random 7,500 clean rows per row group, so 15,000 per month.

### Raw rows read (before any filter): 113,818,969
- **Language mix:** `un` 52.83%, `en` 39.06%, `es` 3.21%, `ru` 0.92%, `ms` 0.35%, `id` 0.33%,
  `ar` 0.26%, `it` 0.23%, `fr` 0.22%, `hi` 0.20%, everything else below 0.2% each.
- **Type:** video 93.2% (106,110,646), photo 6.8% (7,708,323).
- **Country:** US 88.0%, GB 1.4%, MY 1.0%, MX 0.7%, NG 0.6%, GH 0.6%, PK 0.6%.
- **Empty captions:** 28.1%. **Zero or missing views:** 0.34%.
- **Null rates:** `is_ai_generated` 98.5%, `not_recommended_for_fyp` 51.5%. Every other column
  we read has 0% nulls.

**Why English only:** one language keeps TF-IDF and DistilBERT simple, and keeps the
"top 10% within a month" label from comparing different markets. **Caveat:** `un` is the
largest group, and the 5 example rows we inspected were `un` but plainly English. So the
`en`-only filter drops many real English captions. Running our own language detection on `un`
rows is a possible v2 improvement.

## The cleaned sample: `ml/data/sample_en_2022-01_2026-08.parquet`
- **833,091 rows**, 88 MB, 0 duplicate `video_id`s.
- **Rows per month:** 15,000 in every month except **2022-06 (8,091)**. Its second sampled row
  group (35) is the final, partial row group of that file and produced only 591 clean rows. We
  kept it as is so the file reproduces exactly from the code. Phase 2 labels are computed
  within each month, so 8,091 rows is still plenty.
- **`posted_at` range:** 2022-01-01 00:00 to 2026-08-31 20:48 UTC.
- **`stats_updated_at` range:** 2025-07-31 to 2026-10-03 UTC. The counts were captured over
  about 14 months.
- **Null rates:** `is_ai_generated` 97.2%, `not_recommended_for_fyp` 48.8%. All other columns
  have 0% nulls. We won't use these two columns as features.
- **Country:** US 87.8%, GB 2.3%, MY 1.1%, NG 0.9%, GH 0.9%, PK 0.6%.
- **Flags:** `is_ad` 1.4%, `not_recommended_for_fyp == 1` 1.9%, branded content 0.14%.
- **Duration (s):** median 16, p90 68, p99 261, max 3600. 0% zero durations.
- **Caption length (chars):** median 80, p90 223, p99 714, max 5357.
- **Hashtags per caption (regex `#\w+`):** median 4, p90 10. 77.2% of captions have at least
  one hashtag.
- **Views:** median 559, p90 5,662, p99 277,975, max 298,951,120. Very heavy-tailed.
- **Age at capture (days):** min 30, median 695, p90 1,392, max 1,736.

### Per-month view of the sample (from `sample_profile.json`, selected months)
| Month | Rows | Views median | Views p90 | Age min | Age median | Age max |
|---|---|---|---|---|---|---|
| 2022-01 | 15,000 | 387 | 6,992 | 1,278 | 1,530 | 1,736 |
| 2023-06 | 15,000 | 581 | 6,776 | 776 | 1,014 | 1,208 |
| 2024-06 | 15,000 | 587 | 4,973 | 401 | 651 | 846 |
| 2025-06 | 15,000 | 392 | 3,888 | 31 | 343 | 488 |
| 2026-06 | 15,000 | 615 | 4,797 | 30 | 96 | 121 |
| 2026-08 | 15,000 | 809 | 16,098 | 30 | 35 | 63 |

## Step 2: US-only repair (`python -m src.resample_us`). **This is the sample v1 uses.**
Step 1's sample had a cluster effect. 43 of 56 months were 100% US, but some months were about
50% non-US because one of their row groups came from a mixed-country block. Both of 2023-11's
row groups were mixed blocks, so it had 0% US rows, even though the 2023-11 file has 38 pure-US
row groups. v1 is US-only, so:

- A row group is "pure US" when its footer statistics give `country` min = max = `US`.
- Every step-1 row group that was already pure US is kept as is, with no download.
- Every other one is replaced by a random pure-US row group from the same half of the file
  (seed 4242), cleaned with `country == 'US'` added, and subsampled to 7,500 rows.
- **15 months affected, 17 row groups replaced, 0.75 GB** read (approved).
- Result: `ml/data/sample_us_en_2022-01_2026-08.parquet` has **840,000 rows, exactly 15,000 in
  every one of the 56 months, 100% US**. 2023-11 is filled (no gap), and the 2022-06 shortfall
  is fixed because its small final row group was a mixed block and was replaced.
- Running step 1 and then step 2 with the same config reproduces this file exactly.

## Known weaknesses of this sample
1. **Few posting days per month.** Each US row group covers about one day, so each month of
   the US sample covers only **2–5 distinct posting days (median 3.5)**. Posting hour and day
   of week are therefore **not used as features**: they would encode which days were sampled.
   (Step 1's country-mix problem is solved by step 2.)
2. **Age at capture varies within a month** (interquartile range median 137 days in step 1).
   This is almost entirely *capture timing*: stats were captured in waves between 2025-07 and
   2026-10. It biases a month-only label, so labels are computed within month × age tertile
   (see `label_notes.md`).
3. **The newest month looks different.** In step 1, 2026-08 had p90 views of 16,098, about 3×
   a typical month. Part of that came from ads (8.7% of the month, now excluded) and from a
   non-US row group (now replaced). Even US non-ad 2026-08 videos are still higher than
   2026-05 to 07 at the same age. It is evaluated separately as a stress set
   (see `label_notes.md`).
4. **No creator information** (see above).
