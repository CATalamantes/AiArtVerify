import nbformat as nbf

path = '/Users/prabesharyal/ai4all/notebooks/05_days_to_trend.ipynb'
nb = nbf.read(path, as_version=4)

summary = """## Summary

**The theory holds.** Adding `days_to_trend` (days between publish and the trending-date snapshot) to the deduped, group-split feature set improved both models meaningfully:

| | RMSE | R² |
|---|---:|---:|
| Linear Regression — v2 (no days_to_trend) | 7,152,347 | 0.0458 |
| Linear Regression — + days_to_trend | 6,902,329 | 0.1114 |
| Random Forest — v2 (no days_to_trend) | 7,011,337 | 0.0831 |
| Random Forest — + days_to_trend | 6,243,814 | 0.2729 |

Linear R² more than doubled in relative terms (+143%, 0.046 → 0.111); Random Forest's more than tripled (+228%, 0.083 → 0.273). RMSE fell too (-3.5% linear, -11.0% RF) — a genuine improvement, not the variance-shrinkage artifact seen when the leaked split was fixed, since the validation set (and its target distribution) is identical here to `04_dedup_group_split.ipynb`; only the feature set changed.

**`days_to_trend` is the single most important feature for Random Forest** — importance 0.213, ranked #1 of 20, ahead of `title_length` (0.210), `publish_hour` (0.159), and `tag_count` (0.152). For Linear Regression, its raw coefficient (+262,284 views per additional day trending) ranks only #15 by magnitude among 20 features — but that's a scale artifact, not a real weakness: `days_to_trend` spans dozens of days on this dataset, so a swing of ~20 days corresponds to ~5.2M predicted views, comparable to the biggest category-dummy coefficients (3–6M). Raw coefficient magnitude isn't directly comparable across features on different scales (same caveat noted in `02_baseline_linear_regression.ipynb`) — the R² jump is the more trustworthy signal of its contribution for the linear model.

**Conclusion**: the hypothesis from `04_dedup_group_split.ipynb` is confirmed — a real, sizeable share of the "unexplained" variance in the deduped/grouped baseline was exactly the missing "which day of the trending run is this" signal, not just noise. `days_to_trend` is not usable as a true pre-publish feature (you can't know it before a video starts trending), but it's valuable for understanding the data-generating process, and for any modeling task that isn't strictly "predict virality before publish" (e.g. "given a video is N days into trending, estimate its current view count")."""

nb['cells'][-1]['source'] = summary
nbf.write(nb, path)
print('summary written')
