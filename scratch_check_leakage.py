import sys
sys.path.append('/Users/prabesharyal/ai4all/src')

import pandas as pd
from sklearn.model_selection import train_test_split

from features import build_features

df = pd.read_parquet('/Users/prabesharyal/ai4all/data/processed/train.parquet')
df = build_features(df)

FEATURE_COLUMNS_RAW = [
    'video_category_id', 'tag_count', 'title_length', 'title_has_caps_word',
    'publish_hour', 'publish_dayofweek', 'video_trending_country',
]
TARGET_COLUMN = 'video_view_count'
model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN] + ['video_id']].dropna(subset=FEATURE_COLUMNS_RAW + [TARGET_COLUMN]).copy()

print(f"Total rows: {len(model_df):,}")
print(f"Unique video_id: {model_df['video_id'].nunique():,}")
print(f"Rows per video_id (mean): {len(model_df) / model_df['video_id'].nunique():.2f}")
print()

v2_cols = [c for c in FEATURE_COLUMNS_RAW if c != 'video_trending_country']

# Exact duplicate rows in the v2 feature space (ignoring country/video_id), including target
dupe_mask_with_y = model_df.duplicated(subset=v2_cols + [TARGET_COLUMN], keep=False)
print(f"Rows that are exact duplicates (v2 features + target) of >=1 other row: {dupe_mask_with_y.sum():,} "
      f"({dupe_mask_with_y.mean():.1%})")

dupe_mask_x_only = model_df.duplicated(subset=v2_cols, keep=False)
print(f"Rows that are exact duplicates on v2 FEATURES ONLY (X, ignoring y): {dupe_mask_x_only.sum():,} "
      f"({dupe_mask_x_only.mean():.1%})")

# For rows sharing identical X, how much does y vary? (same-X groups)
same_x = model_df.groupby(v2_cols)[TARGET_COLUMN]
group_sizes = same_x.size()
print(f"Number of distinct v2-feature X combos: {len(group_sizes):,}")
print(f"Combos with >1 row: {(group_sizes > 1).sum():,}")

# Now replicate the actual train/val split (same random_state=42) used in the RF notebook and check leakage
X_raw_v2 = model_df[v2_cols]
y = model_df[TARGET_COLUMN].astype('float64')
X_encoded = pd.get_dummies(X_raw_v2, columns=['video_category_id'], drop_first=True)

idx_train, idx_val = train_test_split(model_df.index, test_size=0.2, random_state=42)

train_vid = set(model_df.loc[idx_train, 'video_id'])
val_vid = set(model_df.loc[idx_val, 'video_id'])
overlap = train_vid & val_vid
print()
print(f"Unique videos in train: {len(train_vid):,}")
print(f"Unique videos in val:   {len(val_vid):,}")
print(f"Videos appearing in BOTH train and val (same video_id split across the row-random split): {len(overlap):,} "
      f"({len(overlap) / len(val_vid):.1%} of val's unique videos)")

# Of validation rows, what fraction have their video_id also present in train?
val_df = model_df.loc[idx_val]
frac_val_rows_leaked = val_df['video_id'].isin(train_vid).mean()
print(f"Fraction of VALIDATION ROWS whose video_id also appears somewhere in train: {frac_val_rows_leaked:.1%}")
