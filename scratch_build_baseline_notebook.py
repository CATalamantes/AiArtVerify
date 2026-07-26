import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 02 — Baseline: Linear Regression (pre-publish features only)\n"
    "\n"
    "Predicts `video_view_count` using only metadata that would be available "
    "**before a video is published** — no channel-size features "
    "(`channel_subscriber_count`, `channel_view_count`, `channel_engagement_rate`), "
    "since those reflect channel authority/history, not the video itself, and "
    "no cross-country trending-overlap features (e.g. `global_hit_share` from "
    "the EDA notebook), since those are retroactive — you can't know how many "
    "countries a video will trend in before it's published.\n"
    "\n"
    "This is trained on an 80/20 split **within** `train.parquet` only. "
    "`val.parquet` (the 2026-06-11 cutoff holdout) stays untouched for later, "
    "final comparison."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('../src')\n"
    "\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "import joblib\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.metrics import mean_squared_error, r2_score\n"
    "from sklearn.model_selection import train_test_split\n"
    "\n"
    "from features import build_features\n"
    "\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.width', 200)\n"
    "\n"
    "MODELS_DIR = Path('../models')\n"
    "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "DATA_PATH = Path('../data/processed/train.parquet')\n"
    "df = pd.read_parquet(DATA_PATH)\n"
    "df = build_features(df)\n"
    "print(f\"Loaded and featurized {len(df):,} rows\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Feature set — pre-publish metadata only\n"
    "\n"
    "**Included**: `video_category_id`, `tag_count`, `title_length`, "
    "`title_has_caps_word`, `publish_hour`, `publish_dayofweek`, "
    "`video_trending_country`.\n"
    "\n"
    "**Explicitly excluded**: `channel_subscriber_count`, `channel_view_count`, "
    "`channel_engagement_rate` (channel-size confound — describes the channel's "
    "standing, not this video, and isn't fixed at publish time either), and "
    "anything derived from cross-country trending overlap (e.g. "
    "`global_hit_share`, `other_countries` from the EDA notebook — retroactive, "
    "only knowable after a video has already trended everywhere it's going to).\n"
    "\n"
    "**Target**: `video_view_count` (raw, untransformed — this is the actual "
    "baseline test of a plain linear model against the real skewed target, not "
    "a log-smoothed version of the problem)."
))

cells.append(nbf.v4.new_code_cell(
    "FEATURE_COLUMNS_RAW = [\n"
    "    'video_category_id',\n"
    "    'tag_count',\n"
    "    'title_length',\n"
    "    'title_has_caps_word',\n"
    "    'publish_hour',\n"
    "    'publish_dayofweek',\n"
    "    'video_trending_country',\n"
    "]\n"
    "TARGET_COLUMN = 'video_view_count'\n"
    "\n"
    "model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN]].dropna().copy()\n"
    "print(f\"Rows after dropping any remaining nulls in the feature/target set: {len(model_df):,} \"\n"
    "      f\"(from {len(df):,})\")\n"
    "\n"
    "X_raw = model_df[FEATURE_COLUMNS_RAW]\n"
    "y = model_df[TARGET_COLUMN].astype('float64')\n"
    "X_raw.head()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## One-hot encode `video_category_id` and `video_trending_country`\n"
    "\n"
    "Using `drop_first=True` so each dummy set has a well-defined baseline "
    "category (avoids the dummy-variable trap / perfect multicollinearity, "
    "which also keeps the linear regression coefficients uniquely identified "
    "and interpretable relative to that baseline)."
))

cells.append(nbf.v4.new_code_cell(
    "categorical_cols = ['video_category_id', 'video_trending_country']\n"
    "numeric_cols = ['tag_count', 'title_length', 'title_has_caps_word', 'publish_hour', 'publish_dayofweek']\n"
    "\n"
    "X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)\n"
    "FEATURE_COLUMNS = list(X_encoded.columns)\n"
    "\n"
    "dropped_category_baseline = sorted(X_raw['video_category_id'].astype(str).unique())[0]\n"
    "dropped_country_baseline = sorted(X_raw['video_trending_country'].astype(str).unique())[0]\n"
    "\n"
    "print(f\"Feature matrix shape after one-hot encoding: {X_encoded.shape}\")\n"
    "print(f\"Total features: {len(FEATURE_COLUMNS)} \"\n"
    "      f\"({len(numeric_cols)} numeric + {len(FEATURE_COLUMNS) - len(numeric_cols)} one-hot dummies)\")\n"
    "print(f\"Dropped (baseline) category: {dropped_category_baseline!r}\")\n"
    "print(f\"Dropped (baseline) country: {dropped_country_baseline!r}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train/validation split — 80/20 WITHIN train.parquet\n"
    "\n"
    "`random_state=42`. This is separate from `val.parquet` (the time-based "
    "holdout), which stays untouched for later."
))

cells.append(nbf.v4.new_code_cell(
    "X_train, X_val, y_train, y_val = train_test_split(\n"
    "    X_encoded, y, test_size=0.2, random_state=42\n"
    ")\n"
    "print(f\"Train: {X_train.shape}\")\n"
    "print(f\"Val:   {X_val.shape}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train LinearRegression, evaluate on the validation split"
))

cells.append(nbf.v4.new_code_cell(
    "model = LinearRegression()\n"
    "model.fit(X_train, y_train)\n"
    "\n"
    "y_pred = model.predict(X_val)\n"
    "\n"
    "rmse = np.sqrt(mean_squared_error(y_val, y_pred))\n"
    "r2 = r2_score(y_val, y_pred)\n"
    "\n"
    "print('=== Baseline Linear Regression — validation results ===')\n"
    "print(f'RMSE: {rmse:,.2f} views')\n"
    "print(f'R^2:  {r2:.4f}')\n"
    "print()\n"
    "print(f'For reference — y_val stats:')\n"
    "print(y_val.describe())"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Coefficients\n"
    "\n"
    "**Caveat on \"magnitude\"**: features here are on very different natural "
    "scales — one-hot dummies are 0/1, while `tag_count`/`title_length`/"
    "`publish_hour`/`publish_dayofweek` range over tens of units — and none "
    "were standardized. So a raw coefficient is \"predicted view-count change "
    "per one-unit change in that feature\" (or per switching that dummy on), "
    "**not** a standardized effect size. Comparing a one-hot country coefficient "
    "against a per-hour or per-tag coefficient directly is apples-to-oranges; "
    "read magnitude within each feature type, not blindly across all of them."
))

cells.append(nbf.v4.new_code_cell(
    "coefs = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values(ascending=False)\n"
    "\n"
    "print('Intercept:', f'{model.intercept_:,.2f}')\n"
    "print()\n"
    "print('=== Top 15 POSITIVE coefficients ===')\n"
    "print(coefs.head(15))\n"
    "print()\n"
    "print('=== Top 15 NEGATIVE coefficients ===')\n"
    "print(coefs.tail(15).sort_values())\n"
    "print()\n"
    "\n"
    "top5_overall = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(5)\n"
    "print('=== Top 5 overall, by |coefficient| ===')\n"
    "print(top5_overall)"
))

cells.append(nbf.v4.new_markdown_cell(
    "**Plain-language interpretation of the top 5 (placeholder — filled in after running)**"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Save model and feature column list"
))

cells.append(nbf.v4.new_code_cell(
    "MODEL_PATH = MODELS_DIR / 'baseline_linear_regression.pkl'\n"
    "FEATURES_PATH = MODELS_DIR / 'baseline_feature_columns.json'\n"
    "\n"
    "joblib.dump(model, MODEL_PATH)\n"
    "with open(FEATURES_PATH, 'w') as f:\n"
    "    json.dump(FEATURE_COLUMNS, f, indent=2)\n"
    "\n"
    "print(f'Saved model to {MODEL_PATH}')\n"
    "print(f'Saved {len(FEATURE_COLUMNS)} feature columns to {FEATURES_PATH}')"
))

nb['cells'] = cells

with open('/Users/prabesharyal/ai4all/notebooks/02_baseline_linear_regression.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
