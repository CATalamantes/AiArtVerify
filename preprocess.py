import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

INPUT_PATH = "features.csv"

print("Loading features...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 1. DEFINE TARGET
# ---------------------------------------------------------------------------
# view_count is heavily right-skewed (a few viral videos dominate) — log-transform it
# so linear regression isn't distorted by outliers.
df["log_target_view_count"] = np.log1p(df["_target_view_count"])
target_col = "log_target_view_count"

# ---------------------------------------------------------------------------
# 2. GROUP HIGH-CARDINALITY CATEGORICALS BEFORE ONE-HOT ENCODING
#    (109 / 106 unique countries would create too many sparse columns
#     for ~7k rows — keep only the most common, bucket the rest as "Other")
# ---------------------------------------------------------------------------
def group_rare_categories(series, top_n=15):
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), "Other")

df["trending_country_grouped"] = group_rare_categories(df["trending_country"], top_n=15)
df["channel_country_grouped"] = group_rare_categories(df["channel_country"], top_n=15)

# ---------------------------------------------------------------------------
# 3. SELECT FEATURE COLUMNS
#    - drop video_id (identifier, not a feature)
#    - drop raw trending_country / channel_country (replaced by grouped + encoded versions)
#    - drop raw channel_subscriber_count / channel_view_count / channel_video_count /
#      channel_avg_views_per_video (replaced by their log_ versions — heavily skewed otherwise)
#    - drop everything prefixed _target_ (these ARE the label, never model inputs)
# ---------------------------------------------------------------------------
drop_cols = [
    "video_id", "trending_country", "channel_country",
    "channel_subscriber_count", "channel_view_count",
    "channel_video_count", "channel_avg_views_per_video",
    "_target_view_count", "_target_like_count", "_target_comment_count",
    "_target_engagement_rate", "log_target_view_count",
]
feature_df = df.drop(columns=drop_cols)

categorical_cols = ["video_category", "trending_country_grouped", "channel_country_grouped"]
numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

print(f"\nNumeric features ({len(numeric_cols)}):", numeric_cols)
print(f"Categorical features ({len(categorical_cols)}):", categorical_cols)

# ---------------------------------------------------------------------------
# 4. ONE-HOT ENCODE CATEGORICALS
# ---------------------------------------------------------------------------
encoded = pd.get_dummies(feature_df[categorical_cols], drop_first=True)
X = pd.concat([feature_df[numeric_cols], encoded], axis=1)
y = df[target_col]

print(f"\nFinal feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")

# ---------------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT (do this BEFORE scaling to avoid leaking test data
#    statistics into the scaler)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

# ---------------------------------------------------------------------------
# 6. SCALE NUMERIC FEATURES
#    Fit the scaler on TRAIN ONLY, then apply the same transform to test.
#    One-hot columns (0/1) don't need scaling, but scaling them too is harmless
#    for plain linear regression and keeps things simple.
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

print("\nSample of scaled training data:")
print(X_train_scaled.head(3))

# ---------------------------------------------------------------------------
# 7. SAVE EVERYTHING FOR THE MODELING STEP
# ---------------------------------------------------------------------------
with open("preprocessed.pkl", "wb") as f:
    pickle.dump(
        {
            "X_train": X_train_scaled, "X_test": X_test_scaled,
            "y_train": y_train, "y_test": y_test,
            "scaler": scaler, "feature_names": list(X.columns),
        },
        f,
    )
print("\nSaved preprocessed.pkl")
