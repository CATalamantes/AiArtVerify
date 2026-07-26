import sys
sys.path.append('/Users/prabesharyal/ai4all/src')

import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import build_features

df = pd.read_parquet('/Users/prabesharyal/ai4all/data/processed/train.parquet')
df = build_features(df)

FEATURE_COLUMNS_RAW = [
    'video_category_id', 'tag_count', 'title_length', 'title_has_caps_word',
    'publish_hour', 'publish_dayofweek', 'video_trending_country',
]
TARGET_COLUMN = 'video_view_count'
model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN] + ['video_id', 'trending_date']].dropna(
    subset=FEATURE_COLUMNS_RAW + [TARGET_COLUMN]
).copy()

deduped_df = model_df.sort_values(['video_id', 'trending_date', 'video_trending_country']).drop_duplicates(
    subset=['video_id', 'trending_date'], keep='first'
)

groups = deduped_df['video_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(deduped_df, deduped_df[TARGET_COLUMN], groups=groups))
train_df = deduped_df.iloc[train_idx]

print(f"Train rows: {len(train_df):,} (should match 04_dedup_group_split.ipynb: 3,343,308)")

stats = {
    "tag_count_mean": float(train_df["tag_count"].mean()),
    "title_length_mean": float(train_df["title_length"].mean()),
    "title_has_caps_word_mode": int(train_df["title_has_caps_word"].mode().iloc[0]),
    "publish_hour_mean": float(train_df["publish_hour"].mean()),
    "publish_dayofweek_mean": float(train_df["publish_dayofweek"].mean()),
    "video_category_id_mode": str(train_df["video_category_id"].mode().iloc[0]),
}
print(json.dumps(stats, indent=2))

with open('/Users/prabesharyal/ai4all/models/baseline_feature_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved to models/baseline_feature_stats.json")
