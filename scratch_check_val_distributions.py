import sys
sys.path.append('/Users/prabesharyal/ai4all/src')

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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

# --- OLD: row-random split (leaked) ---
y_all = model_df[TARGET_COLUMN].astype('float64')
_, y_val_leaked = train_test_split(y_all, test_size=0.2, random_state=42)
print("=== y_val — OLD row-random (leaked) split ===")
print(y_val_leaked.describe())
print()

# --- NEW: dedup + group split ---
deduped_df = model_df.sort_values(['video_id', 'trending_date', 'video_trending_country']).drop_duplicates(
    subset=['video_id', 'trending_date'], keep='first'
)
y_dedup = deduped_df[TARGET_COLUMN].astype('float64')
groups = deduped_df['video_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(deduped_df, y_dedup, groups=groups))
y_val_grouped = y_dedup.iloc[val_idx]
print("=== y_val — NEW dedup + group split ===")
print(y_val_grouped.describe())
