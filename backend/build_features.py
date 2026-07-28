"""Step 1 of the v2 pipeline: cleaned_data.csv -> features_v2.csv + reference artifacts.

Unlike featureExtraction.py, every feature produced here is knowable BEFORE the
video is published, so the same rows can be served from a web form.

Outputs
    features_v2.csv        model-ready features + targets + the is_hit label
    artifacts/reference.json   country groups, category list, input defaults and
                               the target distributions the dashboard plots
"""

import json

import numpy as np
import pandas as pd

import config as C
from features import apply_country_grouping, extract_features, learn_country_groups

print("Loading cleaned_data.csv ...")
raw = pd.read_csv(C.CLEANED_DATA)
print(f"  {len(raw):,} rows x {raw.shape[1]} columns")

# ---------------------------------------------------------------------------
# 1. FEATURES
# ---------------------------------------------------------------------------
feat = extract_features(raw)
country_groups = learn_country_groups(feat)
feat = apply_country_grouping(feat, country_groups)
print(f"  built {len(C.ALL_FEATURES)} model features")

# ---------------------------------------------------------------------------
# 2. TARGETS
# ---------------------------------------------------------------------------
views = pd.to_numeric(raw["video_view_count"], errors="coerce").fillna(0)
likes = pd.to_numeric(raw["video_like_count"], errors="coerce").fillna(0)
comments = pd.to_numeric(raw["video_comment_count"], errors="coerce").fillna(0)

feat["_target_views"] = views
feat["_target_likes"] = likes
feat["_target_comments"] = comments
feat["_target_log_views"] = np.log1p(views)
feat["_target_log_likes"] = np.log1p(likes)
feat["_target_log_comments"] = np.log1p(comments)
feat[C.TARGET_ENGAGEMENT] = ((likes + comments) / views.replace(0, np.nan)).fillna(0)

# ---------------------------------------------------------------------------
# 3. THE HIT LABEL
# ---------------------------------------------------------------------------
# Blend reach and engagement percentiles. Reach alone would just relabel "big
# channel"; engagement alone would crown tiny videos with loyal audiences.
reach_pct = feat["_target_log_views"].rank(pct=True)
eng_pct = feat[C.TARGET_ENGAGEMENT].rank(pct=True)
feat["hit_score"] = C.REACH_WEIGHT * reach_pct + C.ENGAGEMENT_WEIGHT * eng_pct
feat[C.TARGET_HIT] = (feat["hit_score"].rank(pct=True) >= C.HIT_THRESHOLD).astype(int)

print(
    f"  hit label: {feat[C.TARGET_HIT].sum():,} hits / {len(feat):,} "
    f"({feat[C.TARGET_HIT].mean():.1%} base rate)"
)

# ---------------------------------------------------------------------------
# 4. REFERENCE ARTIFACT FOR THE APP
# ---------------------------------------------------------------------------
# The app must never load the 15.6 MB cleaned_data.csv, so precompute the small
# summaries it actually needs.
C.ARTIFACTS.mkdir(exist_ok=True)

numeric_defaults = {
    col: float(feat[col].median())
    for col in C.ALL_FEATURES
    if col not in C.CATEGORICAL_FEATURES
}

category_stats = (
    feat.groupby("video_category")
    .agg(
        n=(C.TARGET_HIT, "size"),
        hit_rate=(C.TARGET_HIT, "mean"),
        median_engagement=(C.TARGET_ENGAGEMENT, "median"),
        median_views=("_target_views", "median"),
    )
    .sort_values("n", ascending=False)
)

# Empirical hit-rate curves: the model-free half of the advice engine. advice.py
# only surfaces a recommendation when the model's counterfactual and this raw
# data agree on the direction, which is what separates advice from noise.
ADVISABLE = [
    "publish_hour",
    "publish_day_of_week",
    "publish_is_weekend",
    "title_length_chars",
    "title_word_count",
    "title_has_number",
    "title_has_bracket",
    "title_has_capitalized_word",
    "title_emoji_count",
    "title_exclamation_count",
    "title_question_count",
    "title_uppercase_ratio",
    "description_length_chars",
    "description_word_count",
    "description_url_count",
    "description_line_count",
    "tag_count",
    "duration_seconds",
    "is_short",
]


def empirical_curve(col, n_bins=8):
    """Hit rate as a function of one feature: exact values if low-cardinality,
    quantile bins otherwise (equal-count bins avoid empty buckets on skewed
    columns like duration_seconds)."""
    s = feat[col]
    if s.nunique() <= 12:
        g = feat.groupby(col)[C.TARGET_HIT].agg(["mean", "size"])
        return {
            "kind": "discrete",
            "values": [float(v) for v in g.index],
            "hit_rate": [round(float(v), 4) for v in g["mean"]],
            "n": [int(v) for v in g["size"]],
        }

    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, len(edges) - 2)
    g = feat.assign(_bin=idx).groupby("_bin")[C.TARGET_HIT].agg(["mean", "size"])
    return {
        "kind": "binned",
        "edges": [round(float(e), 4) for e in edges],
        "hit_rate": [round(float(g["mean"].get(i, np.nan)), 4) for i in range(len(edges) - 1)],
        "n": [int(g["size"].get(i, 0)) for i in range(len(edges) - 1)],
    }


empirical_curves = {c: empirical_curve(c) for c in ADVISABLE}
empirical_curves = {k: v for k, v in empirical_curves.items() if v is not None}

reference = {
    "n_videos": int(len(feat)),
    "empirical_curves": empirical_curves,
    "advisable_features": list(empirical_curves),
    "country_groups": country_groups,
    "categories": sorted(feat["video_category"].unique().tolist()),
    "trending_countries": sorted(feat["trending_country_grouped"].unique().tolist()),
    "channel_countries": sorted(feat["channel_country_grouped"].unique().tolist()),
    "numeric_defaults": numeric_defaults,
    "base_rate": float(feat[C.TARGET_HIT].mean()),
    # distributions the "where do you land" chart draws, rounded to keep the
    # artifact small
    "dist_engagement_rate": [round(v, 5) for v in feat[C.TARGET_ENGAGEMENT].tolist()],
    "dist_log_views": [round(v, 4) for v in feat["_target_log_views"].tolist()],
    "dist_log_likes": [round(v, 4) for v in feat["_target_log_likes"].tolist()],
    "dist_log_comments": [round(v, 4) for v in feat["_target_log_comments"].tolist()],
    "category_stats": {
        str(idx): {
            "n": int(r["n"]),
            "hit_rate": float(r["hit_rate"]),
            "median_engagement": float(r["median_engagement"]),
            "median_views": float(r["median_views"]),
        }
        for idx, r in category_stats.iterrows()
    },
    # channel-size presets for the form, derived from real quantiles
    "channel_quantiles": {
        q: {
            "subscribers": float(
                pd.to_numeric(raw["channel_subscriber_count"], errors="coerce").quantile(q)
            ),
            "views": float(pd.to_numeric(raw["channel_view_count"], errors="coerce").quantile(q)),
            "videos": float(pd.to_numeric(raw["channel_video_count"], errors="coerce").quantile(q)),
        }
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]
    },
}

with open(C.REFERENCE_JSON, "w", encoding="utf-8") as fh:
    json.dump(reference, fh)

feat.to_csv(C.FEATURES_V2, index=False)
print(f"\nSaved {C.FEATURES_V2.name}  ({feat.shape[0]:,} x {feat.shape[1]})")
print(f"Saved {C.REFERENCE_JSON.name} ({C.REFERENCE_JSON.stat().st_size / 1024:.0f} KB)")
print("\nCategory breakdown:")
print(category_stats.head(20).to_string())
