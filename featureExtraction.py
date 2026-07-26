import pandas as pd
import numpy as np
import re

INPUT_PATH = "cleaned_data.csv"
OUTPUT_PATH = "features.csv"

print("Loading data...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

####Extracting Features :)

df["video_published_at"] = pd.to_datetime(df["video_published_at"], utc=True, format="mixed")
df["video_trending__date"] = pd.to_datetime(df["video_trending__date"], utc=True, format="mixed")
df["channel_published_at"] = pd.to_datetime(df["channel_published_at"], utc=True, format="mixed")

features = pd.DataFrame(index=df.index)
features["video_id"] = df["video_id"]

# ---------------------------------------------------------------------------
# 1. TIMING FEATURES
# ---------------------------------------------------------------------------
features["publish_hour"] = df["video_published_at"].dt.hour
features["publish_day_of_week"] = df["video_published_at"].dt.dayofweek  # 0=Mon
features["publish_month"] = df["video_published_at"].dt.month
features["publish_is_weekend"] = features["publish_day_of_week"].isin([5, 6]).astype(int)

# Time from publish to hitting trending (hours) — strong signal for "how fast it took off"
features["hours_to_trending"] = (
        (df["video_trending__date"] - df["video_published_at"]).dt.total_seconds() / 3600
).clip(lower=0)

# Channel age at time of publish (days) — proxy for channel maturity/authority
features["channel_age_days_at_publish"] = (
        (df["video_published_at"] - df["channel_published_at"]).dt.total_seconds() / 86400
).clip(lower=0)


# ---------------------------------------------------------------------------
# 2. VIDEO DURATION
# ---------------------------------------------------------------------------
# Format looks like "0 days 00:04:48" -> convert to seconds
def parse_duration_to_seconds(val):
    try:
        td = pd.Timedelta(val)
        return td.total_seconds()
    except (ValueError, TypeError):
        return np.nan


features["duration_seconds"] = df["video_duration"].apply(parse_duration_to_seconds)
features["is_short"] = (features["duration_seconds"] <= 60).astype(int)  # YouTube Shorts-length

# ---------------------------------------------------------------------------
# 3. TEXT FEATURES — TITLE
# ---------------------------------------------------------------------------
title = df["video_title"].fillna("")
features["title_length_chars"] = title.str.len()
features["title_word_count"] = title.str.split().apply(len)
features["title_has_capitalized_word"] = title.apply(
    lambda s: int(any(w.isupper() and len(w) > 1 for w in s.split()))
)
features["title_uppercase_ratio"] = title.apply(
    lambda s: sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
)
features["title_exclamation_count"] = title.str.count("!")
features["title_question_count"] = title.str.count(r"\?")
features["title_has_number"] = title.str.contains(r"\d", regex=True).astype(int)
features["title_has_bracket"] = title.str.contains(r"[\[\(]", regex=True).astype(int)
features["title_emoji_count"] = title.apply(
    lambda s: len(re.findall(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", s
    ))
)

# ---------------------------------------------------------------------------
# 4. TEXT FEATURES — DESCRIPTION
# ---------------------------------------------------------------------------
desc = df["video_description"].fillna("")
features["description_length_chars"] = desc.str.len()
features["description_word_count"] = desc.str.split().apply(len)
features["description_has_url"] = desc.str.contains(
    r"https?://", regex=True, case=False
).astype(int)
features["description_url_count"] = desc.str.count(r"https?://")
features["description_line_count"] = desc.str.count("\n") + 1
features["has_description"] = (df["video_description"].notna() & (desc.str.len() > 0)).astype(int)



# ------------------------------------------------------------------------
# 5. TAGS
# ---------------------------------------------------------------------------
def count_tags(val):
    if pd.isna(val) or val == "":
        return 0
    # tags are typically pipe or comma separated
    parts = re.split(r"[|,]", str(val))
    return len([p for p in parts if p.strip()])


features["tag_count"] = df["video_tags"].apply(count_tags)
features["has_tags"] = (features["tag_count"] > 0).astype(int)

# ---------------------------------------------------------------------------
# 6. CATEGORICAL / METADATA
# ---------------------------------------------------------------------------
features["video_category"] = df["video_category_id"]  # already human-readable; one-hot at model time
features["video_definition_hd"] = (df["video_definition"] == "hd").astype(int)
features["video_licensed_content"] = df["video_licensed_content"].astype(int)
features["trending_country"] = df["video_trending_country"]
features["channel_country"] = df["channel_country"].fillna("Unknown")

# ---------------------------------------------------------------------------
# 7. CHANNEL AUTHORITY / SIZE FEATURES (pre-existing channel stats — not leakage,
#    since these describe the channel's general standing, not this video's outcome)
# ---------------------------------------------------------------------------
features["channel_subscriber_count"] = df["channel_subscriber_count"]
features["channel_view_count"] = df["channel_view_count"]
features["channel_video_count"] = df["channel_video_count"]
features["channel_have_hidden_subscribers"] = df["channel_have_hidden_subscribers"].astype(int)

# Average views per video for the channel historically — proxy for channel "pull"
features["channel_avg_views_per_video"] = (
        df["channel_view_count"] / df["channel_video_count"].replace(0, np.nan)
).fillna(0)

# Subscriber-to-view efficiency (loyal vs. viral-driven channel)
features["channel_sub_to_view_ratio"] = (
        df["channel_subscriber_count"] / df["channel_view_count"].replace(0, np.nan)
).fillna(0)

features["channel_description_length"] = df["channel_description"].fillna("").str.len()
features["channel_has_custom_url"] = df["channel_custom_url"].notna().astype(int)

# ---------------------------------------------------------------------------
# 8. TARGET / LABEL COLUMNS (kept separate — build your target from these,
#    do NOT use them as model inputs)
# ---------------------------------------------------------------------------
features["_target_view_count"] = df["video_view_count"]
features["_target_like_count"] = df["video_like_count"]
features["_target_comment_count"] = df["video_comment_count"]
# Example composite engagement rate you might use as a single target:
features["_target_engagement_rate"] = (
        (df["video_like_count"] + df["video_comment_count"]) / df["video_view_count"].replace(0, np.nan)
).fillna(0)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
features.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {features.shape[1]} columns x {features.shape[0]:,} rows -> {OUTPUT_PATH}")
print("\nFeature columns created:")
for c in features.columns:
    if c not in ("video_id",) and not c.startswith("_target"):
        print(f"  - {c}")
print("\nTarget columns (prefixed _target_, exclude from model inputs):")
for c in features.columns:
    if c.startswith("_target"):
        print(f"  - {c}")