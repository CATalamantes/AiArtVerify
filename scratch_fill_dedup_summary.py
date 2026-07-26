import nbformat as nbf

path = '/Users/prabesharyal/ai4all/notebooks/04_dedup_group_split.ipynb'
nb = nbf.read(path, as_version=4)

summary = """## Summary

**Dataset shrank by 57.6%** on dedup: 9,876,866 → 4,183,259 rows (dropped 5,693,607 same-day country-duplicate rows). The 1,039,987 unique videos average 4.02 distinct trending dates each post-dedup (down from 9.50 raw rows/video pre-dedup — confirming most of the original row count was country repetition, not date-level signal). GroupShuffleSplit then produced Train: 3,343,308 rows / 831,989 videos, Val: 839,951 rows / 207,998 videos, with confirmed zero video_id overlap between the two.

**Results, corrected split:**

| | RMSE | R² |
|---|---:|---:|
| Linear Regression — row-random (leaked) | 18,932,915 | 0.0972 |
| Linear Regression — dedup + grouped | 7,152,347 | 0.0458 |
| Random Forest — row-random (leaked) | 9,429,241 | 0.7761 |
| Random Forest — dedup + grouped | 7,011,337 | 0.0831 |

**Random Forest's leaked 0.776 R² was almost entirely the video-identity leakage — it collapses to 0.083 once no video can appear on both sides of the split.** That confirms the diagnosis: a depth-15 forest could memorize `(title_length, tag_count, category, hour, dow) → view_count` lookups for videos it had already seen in train, and 95.3% of the old validation rows had exactly that leak available. With leakage closed, Random Forest still modestly outperforms Linear Regression (R² 0.083 vs. 0.046, RMSE 7.01M vs. 7.15M) — a real but far more modest edge from capturing non-linear/interaction effects, not from memorization.

**RMSE fell for both models even though R² also fell — not a contradiction.** The old row-random validation set's target had std ≈ 19.93M (dominated by mega-viral videos whose ~9 country-duplicate rows each inflated their weight in both the split and its variance). The corrected validation set's target has std ≈ 7.32M — removing the country duplication removed most of that inflation, shrinking the achievable error (lower RMSE) but also shrinking the variance R² is measured against, which fell faster than the errors did (both models now explain a smaller share of a smaller, harder pie). This is the more honest number: predicting raw view count from five pre-publish, video-level fields, without knowing which day of a video's trending run is being sampled, is a genuinely hard problem — R² in the 0.05–0.08 range is a believable ceiling for this feature set, not the 0.78 the leaked split implied.

**Next step worth considering**: even post-fix, a video contributes one row per trending date with *identical* pre-publish features but a *growing* target (the same video's view count 3 weeks into trending vs. day 1 look wildly different for the same X) — this is irreducible label noise given the current feature set. Adding a "days since publish" / "day of trending run" feature, or restricting to one canonical snapshot per video (e.g. first trending day), would likely be the next real lever on R², more so than further model tuning on the current feature set."""

nb['cells'][-1]['source'] = summary
nbf.write(nb, path)
print('summary written')
