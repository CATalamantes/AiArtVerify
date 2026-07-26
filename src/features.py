"""
Feature engineering for the cleaned canerkonuk train/val dataframes
(output of src/data_prep.py).

NOTE: canerkonuk has no dislike-count column (unlike datasnaek) — YouTube's
public API dropped dislike counts in Dec 2021, and this dataset postdates
that. like_ratio is therefore not computable and is intentionally omitted.
"""

import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "data", "processed", "train.parquet")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------------------------------
    # Timing features — from the video's publish timestamp, not
    # trending_date (when it trended, not when it was published).
    # -----------------------------------------------------------------
    publish_dt = pd.to_datetime(df["video_published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["publish_hour"] = publish_dt.dt.hour
    df["publish_dayofweek"] = publish_dt.dt.dayofweek  # 0=Mon

    df["days_to_trend"] = (df["trending_date"] - publish_dt).dt.total_seconds() / 86400.0

    # -----------------------------------------------------------------
    # Tags
    # -----------------------------------------------------------------
    tags = df["video_tags"].fillna("")
    df["tag_count"] = tags.apply(lambda s: len([t for t in s.split(",") if t.strip()]))

    # -----------------------------------------------------------------
    # Title — rough clickbait signals
    # -----------------------------------------------------------------
    title = df["video_title"].fillna("")
    df["title_length"] = title.str.len()
    df["title_has_caps_word"] = title.apply(
        lambda s: int(any(w.isupper() and len(w) > 1 for w in s.split()))
    )

    # -----------------------------------------------------------------
    # like_ratio — SKIPPED. canerkonuk has no dislike-count column, so
    # video_like_count / (video_like_count + dislikes) is not computable.
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # comment_rate — guard against divide-by-zero. Clipped at 1.0: a
    # handful of low-traffic videos show comment_count > view_count
    # (view-count reporting lag), which isn't a data error worth
    # dropping rows over, just bounding the derived feature.
    # -----------------------------------------------------------------
    view_count = df["video_view_count"].astype("Float64")
    df["comment_rate"] = (
        df["video_comment_count"].astype("Float64") / view_count.where(view_count > 0)
    ).clip(upper=1.0)

    # -----------------------------------------------------------------
    # channel_engagement_rate — guard against divide-by-zero / null subs
    # -----------------------------------------------------------------
    subscriber_count = df["channel_subscriber_count"].astype("Float64")
    df["channel_engagement_rate"] = (
        df["channel_view_count"].astype("Float64") / subscriber_count.where(subscriber_count > 0)
    )

    # -----------------------------------------------------------------
    # video_category_id — kept as-is, just marked categorical.
    # Not encoded yet; one-hot vs. target encoding is a modeling-stage choice.
    # -----------------------------------------------------------------
    df["video_category_id"] = df["video_category_id"].astype("category")

    return df


if __name__ == "__main__":
    df = pd.read_parquet(TRAIN_PATH)
    print(f"Loaded {len(df):,} rows from {TRAIN_PATH}")

    df = build_features(df)

    new_columns = [
        "publish_hour",
        "publish_dayofweek",
        "days_to_trend",
        "tag_count",
        "title_length",
        "title_has_caps_word",
        "comment_rate",
        "channel_engagement_rate",
    ]

    print("\nlike_ratio: SKIPPED — canerkonuk has no dislike-count column.")
    print("\n.describe() on new feature columns:")
    print(df[new_columns].describe())
