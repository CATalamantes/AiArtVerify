import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 05 — Adding `days_to_trend` to the corrected (dedup + grouped) feature set\n"
    "\n"
    "`04_dedup_group_split.ipynb` fixed the video-identity leakage but flagged a "
    "remaining issue: after dedup, a video still contributes one row per "
    "distinct `trending_date`, all sharing identical *pre-publish* features "
    "(`video_category_id`, `tag_count`, `title_length`, `title_has_caps_word`, "
    "`publish_hour`, `publish_dayofweek`) but a target that climbs across those "
    "rows as the video accumulates views during its trending run (e.g. the "
    "Kendrick Lamar Super Bowl video: 16.3M views on day 1 of trending → 94.2M "
    "by day ~21). With no feature distinguishing an early-trending snapshot "
    "from a late one, that's irreducible label noise for the v2 feature set.\n"
    "\n"
    "`build_features` in `src/features.py` already computes `days_to_trend` — "
    "`(trending_date - video_published_at)` in days — which is exactly that "
    "missing signal: how long the video had been public by the time this "
    "particular row's view-count snapshot was taken. Adding it directly tests "
    "the theory: if most of the remaining unexplained variance really is "
    "\"which day of the trending run is this,\" `days_to_trend` should pick up "
    "a large share of it.\n"
    "\n"
    "Same dedup (`video_id` + `trending_date`) and `GroupShuffleSplit` by "
    "`video_id` (`test_size=0.2, random_state=42`) as `04_dedup_group_split.ipynb` "
    "— only the feature set changes, so the comparison isolates the effect of "
    "`days_to_trend`."
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
    "from sklearn.ensemble import RandomForestRegressor\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.metrics import mean_squared_error, r2_score\n"
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "\n"
    "from features import build_features\n"
    "\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.width', 200)\n"
    "\n"
    "MODELS_DIR = Path('../models')\n"
    "\n"
    "DATA_PATH = Path('../data/processed/train.parquet')\n"
    "df = pd.read_parquet(DATA_PATH)\n"
    "df = build_features(df)\n"
    "print(f\"Loaded and featurized {len(df):,} rows\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Feature set — v2 pre-publish features + `days_to_trend`"
))

cells.append(nbf.v4.new_code_cell(
    "FEATURE_COLUMNS_RAW = [\n"
    "    'video_category_id',\n"
    "    'tag_count',\n"
    "    'title_length',\n"
    "    'title_has_caps_word',\n"
    "    'publish_hour',\n"
    "    'publish_dayofweek',\n"
    "    'days_to_trend',\n"
    "    'video_trending_country',\n"
    "]\n"
    "FEATURE_COLUMNS_RAW_V3 = [c for c in FEATURE_COLUMNS_RAW if c != 'video_trending_country']\n"
    "TARGET_COLUMN = 'video_view_count'\n"
    "KEY_COLUMNS = ['video_id', 'trending_date']\n"
    "\n"
    "model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN] + KEY_COLUMNS].dropna(\n"
    "    subset=FEATURE_COLUMNS_RAW + [TARGET_COLUMN]\n"
    ").copy()\n"
    "print(f\"Rows before dedup: {len(model_df):,} (from {len(df):,})\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Dedup: one row per `(video_id, trending_date)`\n"
    "\n"
    "Same procedure as `04_dedup_group_split.ipynb`. Note `days_to_trend` is "
    "itself a function of `(video_id, trending_date)`, so it does NOT collapse "
    "away in this dedup step the way country did — rows for the same video on "
    "different trending dates now have genuinely different `days_to_trend` "
    "values, not identical features."
))

cells.append(nbf.v4.new_code_cell(
    "model_df_sorted = model_df.sort_values(['video_id', 'trending_date', 'video_trending_country'])\n"
    "deduped_df = model_df_sorted.drop_duplicates(subset=['video_id', 'trending_date'], keep='first').copy()\n"
    "\n"
    "print(f\"Rows before dedup: {len(model_df):,}\")\n"
    "print(f\"Rows after dedup:  {len(deduped_df):,}\")\n"
    "print(f\"Unique video_id in deduped data: {deduped_df['video_id'].nunique():,}\")\n"
    "print(f\"Mean rows per video after dedup: {len(deduped_df) / deduped_df['video_id'].nunique():.2f}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Encode features (v2 categorical one-hot + `days_to_trend` as a plain numeric)"
))

cells.append(nbf.v4.new_code_cell(
    "X_raw_v3 = deduped_df[FEATURE_COLUMNS_RAW_V3]\n"
    "y = deduped_df[TARGET_COLUMN].astype('float64')\n"
    "groups = deduped_df['video_id']\n"
    "\n"
    "X_encoded = pd.get_dummies(X_raw_v3, columns=['video_category_id'], drop_first=True)\n"
    "FEATURE_COLUMNS = list(X_encoded.columns)\n"
    "print(f\"Feature matrix shape: {X_encoded.shape}\")\n"
    "print(f\"Total features: {len(FEATURE_COLUMNS)} (v2 dedup baseline had 19; +1 for days_to_trend)\")\n"
    "print(FEATURE_COLUMNS)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## GroupShuffleSplit by `video_id` — identical split logic to `04_dedup_group_split.ipynb`"
))

cells.append(nbf.v4.new_code_cell(
    "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
    "train_idx, val_idx = next(gss.split(X_encoded, y, groups=groups))\n"
    "\n"
    "X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]\n"
    "y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]\n"
    "groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]\n"
    "\n"
    "print(f\"Train: {X_train.shape} ({groups_train.nunique():,} unique videos)\")\n"
    "print(f\"Val:   {X_val.shape} ({groups_val.nunique():,} unique videos)\")\n"
    "print(f\"Video overlap between train and val: {len(set(groups_train) & set(groups_val)):,} (should be 0)\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train Linear Regression"
))

cells.append(nbf.v4.new_code_cell(
    "lr = LinearRegression()\n"
    "lr.fit(X_train, y_train)\n"
    "y_pred_lr = lr.predict(X_val)\n"
    "\n"
    "rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))\n"
    "r2_lr = r2_score(y_val, y_pred_lr)\n"
    "\n"
    "print('=== Linear Regression (+ days_to_trend) — validation results ===')\n"
    "print(f'RMSE: {rmse_lr:,.2f} views')\n"
    "print(f'R^2:  {r2_lr:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train Random Forest\n"
    "\n"
    "Same hyperparameters as `04_dedup_group_split.ipynb` "
    "(`n_estimators=200, max_depth=15, n_jobs=-1, random_state=42`)."
))

cells.append(nbf.v4.new_code_cell(
    "rf = RandomForestRegressor(\n"
    "    n_estimators=200,\n"
    "    max_depth=15,\n"
    "    n_jobs=-1,\n"
    "    random_state=42,\n"
    ")\n"
    "rf.fit(X_train, y_train)\n"
    "y_pred_rf = rf.predict(X_val)\n"
    "\n"
    "rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))\n"
    "r2_rf = r2_score(y_val, y_pred_rf)\n"
    "\n"
    "print('=== Random Forest (+ days_to_trend) — validation results ===')\n"
    "print(f'RMSE: {rmse_rf:,.2f} views')\n"
    "print(f'R^2:  {r2_rf:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Comparison against the dedup-only (no `days_to_trend`) baseline from `04_dedup_group_split.ipynb`"
))

cells.append(nbf.v4.new_code_cell(
    "rmse_lr_v2 = 7_152_346.60\n"
    "r2_lr_v2 = 0.0458\n"
    "rmse_rf_v2 = 7_011_337.01\n"
    "r2_rf_v2 = 0.0831\n"
    "\n"
    "print(f'{\"\":38s} {\"RMSE\":>18s} {\"R^2\":>10s}')\n"
    "print(f'{\"Linear Regression — v2 (no days_to_trend)\":38s} {rmse_lr_v2:>18,.2f} {r2_lr_v2:>10.4f}')\n"
    "print(f'{\"Linear Regression — + days_to_trend\":38s} {rmse_lr:>18,.2f} {r2_lr:>10.4f}')\n"
    "print(f'{\"Random Forest — v2 (no days_to_trend)\":38s} {rmse_rf_v2:>18,.2f} {r2_rf_v2:>10.4f}')\n"
    "print(f'{\"Random Forest — + days_to_trend\":38s} {rmse_rf:>18,.2f} {r2_rf:>10.4f}')\n"
    "print()\n"
    "print(f'Linear RMSE change: {rmse_lr - rmse_lr_v2:+,.2f} ({(rmse_lr - rmse_lr_v2) / rmse_lr_v2:+.2%})')\n"
    "print(f'Linear R^2 change:  {r2_lr - r2_lr_v2:+.4f} ({(r2_lr - r2_lr_v2) / r2_lr_v2:+.2%} relative)')\n"
    "print(f'RF RMSE change:     {rmse_rf - rmse_rf_v2:+,.2f} ({(rmse_rf - rmse_rf_v2) / rmse_rf_v2:+.2%})')\n"
    "print(f'RF R^2 change:      {r2_rf - r2_rf_v2:+.4f} ({(r2_rf - r2_rf_v2) / r2_rf_v2:+.2%} relative)')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## `days_to_trend`'s contribution — coefficient (linear) and importance (RF)"
))

cells.append(nbf.v4.new_code_cell(
    "coefs = pd.Series(lr.coef_, index=FEATURE_COLUMNS).sort_values(ascending=False)\n"
    "coef_rank_by_abs = coefs.abs().sort_values(ascending=False).index.get_loc('days_to_trend') + 1\n"
    "\n"
    "print('=== Linear Regression coefficient for days_to_trend ===')\n"
    "print(f\"days_to_trend coefficient: {coefs['days_to_trend']:,.2f} views per additional day trending\")\n"
    "print(f\"Rank by |coefficient| among all {len(FEATURE_COLUMNS)} features: #{coef_rank_by_abs}\")\n"
    "print()\n"
    "print('Top 5 by |coefficient|:')\n"
    "print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(5))\n"
    "print()\n"
    "\n"
    "importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)\n"
    "importance_rank = list(importances.index).index('days_to_trend') + 1\n"
    "\n"
    "print('=== Random Forest importance for days_to_trend ===')\n"
    "print(f\"days_to_trend importance: {importances['days_to_trend']:.4f} \"\n"
    "      f\"({importances['days_to_trend'] / importances.sum():.1%} of total)\")\n"
    "print(f\"Rank among all {len(FEATURE_COLUMNS)} features: #{importance_rank}\")\n"
    "print()\n"
    "print('Top 5 by importance:')\n"
    "print(importances.head(5))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Save the models"
))

cells.append(nbf.v4.new_code_cell(
    "LR_PATH = MODELS_DIR / 'linear_regression_days_to_trend.pkl'\n"
    "RF_PATH = MODELS_DIR / 'random_forest_days_to_trend.pkl'\n"
    "FEATURES_PATH = MODELS_DIR / 'days_to_trend_feature_columns.json'\n"
    "\n"
    "joblib.dump(lr, LR_PATH)\n"
    "joblib.dump(rf, RF_PATH)\n"
    "with open(FEATURES_PATH, 'w') as f:\n"
    "    json.dump(FEATURE_COLUMNS, f, indent=2)\n"
    "\n"
    "print(f'Saved Linear Regression to {LR_PATH}')\n"
    "print(f'Saved Random Forest to {RF_PATH}')\n"
    "print(f'Saved {len(FEATURE_COLUMNS)} feature columns to {FEATURES_PATH}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Summary (placeholder — filled in after running)"
))

nb['cells'] = cells

with open('/Users/prabesharyal/ai4all/notebooks/05_days_to_trend.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
