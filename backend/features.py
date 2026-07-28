"""Feature extraction shared by training and inference.

This module is deliberately the ONLY place features get built. build_features.py
calls it on the full dataset; inference.py calls it on a one-row frame assembled
from the user's form. Because both go through the same function, the training
and serving feature definitions cannot drift apart -- the classic way a model
ends up confidently wrong in production.

The extraction logic is ported verbatim from featureExtraction.py; the regexes
and edge cases there are already correct.
"""

import re

import numpy as np
import pandas as pd

from config import RARE_CATEGORY_LABEL, TOP_N_COUNTRIES

EMOJI_RE = r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
URL_RE = r"https?://"


def _to_int_bool(series):
    """Coerce True/False, 'True'/'False', 1/0 and NaN into a 0/1 int column."""
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
        .astype(int)
    )


def parse_duration_to_seconds(val):
    """Accept either a timedelta string ('0 days 00:04:48') or plain seconds.

    cleaned_data.csv stores timedelta strings; the app hands us a number. Both
    converge here so there is still only one code path downstream.
    """
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    try:
        return pd.Timedelta(val).total_seconds()
    except (ValueError, TypeError):
        return np.nan


def count_tags(val):
    """Tags are pipe- or comma-separated. Blank parts don't count."""
    if pd.isna(val) or val == "":
        return 0
    return len([p for p in re.split(r"[|,]", str(val)) if p.strip()])


def split_tags(val):
    """Same split as count_tags, but returns the normalised tag strings."""
    if pd.isna(val) or val == "":
        return []
    return [p.strip().lower() for p in re.split(r"[|,]", str(val)) if p.strip()]


def extract_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature frame from raw video/channel columns.

    Expects the cleaned_data.csv schema. `video_trending__date` is interpreted
    as "the moment of measurement" -- at training that is the trending snapshot,
    at inference it is publish time plus the user's chosen horizon.
    """
    df = raw.copy()

    for col in ["video_published_at", "video_trending__date", "channel_published_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed")

    f = pd.DataFrame(index=df.index)

    # --- 1. timing ---------------------------------------------------------
    f["publish_hour"] = df["video_published_at"].dt.hour
    f["publish_day_of_week"] = df["video_published_at"].dt.dayofweek  # 0 = Monday
    f["publish_month"] = df["video_published_at"].dt.month
    f["publish_is_weekend"] = f["publish_day_of_week"].isin([5, 6]).astype(int)

    # Video age at measurement time. Formerly `hours_to_trending`: identical
    # arithmetic, but reframed as a prediction horizon the user controls.
    f["hours_since_publish"] = (
        (df["video_trending__date"] - df["video_published_at"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    f["channel_age_days_at_publish"] = (
        (df["video_published_at"] - df["channel_published_at"]).dt.total_seconds() / 86400
    ).clip(lower=0)

    # --- 2. duration -------------------------------------------------------
    f["duration_seconds"] = df["video_duration"].apply(parse_duration_to_seconds)
    f["is_short"] = (f["duration_seconds"] <= 60).astype(int)

    # --- 3. title ----------------------------------------------------------
    title = df["video_title"].fillna("").astype(str)
    f["title_length_chars"] = title.str.len()
    f["title_word_count"] = title.str.split().apply(len)
    f["title_has_capitalized_word"] = title.apply(
        lambda s: int(any(w.isupper() and len(w) > 1 for w in s.split()))
    )
    f["title_uppercase_ratio"] = title.apply(
        lambda s: sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
    )
    f["title_exclamation_count"] = title.str.count("!")
    f["title_question_count"] = title.str.count(r"\?")
    f["title_has_number"] = title.str.contains(r"\d", regex=True).astype(int)
    f["title_has_bracket"] = title.str.contains(r"[\[\(]", regex=True).astype(int)
    f["title_emoji_count"] = title.apply(lambda s: len(re.findall(EMOJI_RE, s)))

    # --- 4. description ----------------------------------------------------
    desc = df["video_description"].fillna("").astype(str)
    f["description_length_chars"] = desc.str.len()
    f["description_word_count"] = desc.str.split().apply(len)
    f["description_has_url"] = desc.str.contains(URL_RE, regex=True, case=False).astype(int)
    f["description_url_count"] = desc.str.count(URL_RE)
    f["description_line_count"] = desc.str.count("\n") + 1
    f["has_description"] = (df["video_description"].notna() & (desc.str.len() > 0)).astype(int)

    # --- 5. tags -----------------------------------------------------------
    f["tag_count"] = df["video_tags"].apply(count_tags)
    f["has_tags"] = (f["tag_count"] > 0).astype(int)

    # --- 6. categorical / metadata ----------------------------------------
    f["video_category"] = df["video_category_id"].astype(str)
    f["video_definition_hd"] = (df["video_definition"] == "hd").astype(int)
    f["video_licensed_content"] = _to_int_bool(df["video_licensed_content"])
    f["trending_country"] = df["video_trending_country"].fillna("Unknown").astype(str)
    f["channel_country"] = df["channel_country"].fillna("Unknown").astype(str)

    # --- 7. channel authority ---------------------------------------------
    subs = pd.to_numeric(df["channel_subscriber_count"], errors="coerce").fillna(0)
    ch_views = pd.to_numeric(df["channel_view_count"], errors="coerce").fillna(0)
    ch_videos = pd.to_numeric(df["channel_video_count"], errors="coerce").fillna(0)

    f["channel_have_hidden_subscribers"] = _to_int_bool(df["channel_have_hidden_subscribers"])
    avg_views = (ch_views / ch_videos.replace(0, np.nan)).fillna(0)
    f["channel_sub_to_view_ratio"] = (subs / ch_views.replace(0, np.nan)).fillna(0)
    f["channel_description_length"] = df["channel_description"].fillna("").astype(str).str.len()
    f["channel_has_custom_url"] = df["channel_custom_url"].notna().astype(int)

    # Raw channel counts are skewed ~20-40; log them so linear regression isn't
    # dominated by a handful of enormous channels.
    f["log_channel_subscriber_count"] = np.log1p(subs)
    f["log_channel_view_count"] = np.log1p(ch_views)
    f["log_channel_video_count"] = np.log1p(ch_videos)
    f["log_channel_avg_views_per_video"] = np.log1p(avg_views)

    return f


def learn_country_groups(features: pd.DataFrame) -> dict:
    """Keep the top-N countries by frequency; everything else becomes 'Other'.

    109 trending / 106 channel countries over ~6.7k rows would otherwise blow up
    into hundreds of near-empty one-hot columns.
    """
    return {
        col: features[col].value_counts().head(TOP_N_COUNTRIES).index.tolist()
        for col in ["trending_country", "channel_country"]
    }


def apply_country_grouping(features: pd.DataFrame, groups: dict) -> pd.DataFrame:
    f = features.copy()
    for col in ["trending_country", "channel_country"]:
        keep = groups[col]
        f[f"{col}_grouped"] = f[col].where(f[col].isin(keep), RARE_CATEGORY_LABEL)
    return f
