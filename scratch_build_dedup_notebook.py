import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 04 — Fixing video-level leakage: dedup + GroupShuffleSplit\n"
    "\n"
    "**Why this notebook exists**: `03_random_forest.ipynb` trained on a plain "
    "row-random 80/20 split and got a suspiciously large jump over the linear "
    "baseline (R² 0.097 → 0.776). Investigation found the row-random split was "
    "leaking: this dataset has one row per `(video_id, video_trending_country, "
    "trending_date)`, but the v2 feature set (`video_category_id`, `tag_count`, "
    "`title_length`, `title_has_caps_word`, `publish_hour`, `publish_dayofweek`) "
    "is entirely video-level — none of it varies by country or date. That means:\n"
    "\n"
    "1. **Same video, same `trending_date`, different countries → identical "
    "`video_view_count`.** Confirmed directly: `ekr2nIex040` (\"ROSÉ & Bruno "
    "Mars - APT.\") shows the exact same view count (64,890,495) across all 91 "
    "countries it trended in on 2024-10-20 — it's a single global daily "
    "snapshot broadcast per country, not a country-specific count. These rows "
    "are 100% redundant given the v2 feature set (country isn't even a "
    "feature) — pure duplication, zero information.\n"
    "2. **Same video, different `trending_date` → `video_view_count` genuinely "
    "differs**, climbing as the video accumulates more views during its "
    "trending run (e.g. the same APT. video: 143.7M on 2024-10-26 → 407.5M by "
    "2024-11-23). These rows share identical v2 features but different "
    "targets — a row-random split scatters a video's early- and late-trending "
    "snapshots across both train and val, so the model can partly \"recognize\" "
    "a video from train and interpolate its val-set target, rather than "
    "generalizing to unseen videos.\n"
    "\n"
    "**The fix, in two steps**:\n"
    "1. **Dedup**: collapse same-day country duplicates down to one row per "
    "`(video_id, trending_date)` — no information lost, since those rows were "
    "exact duplicates in the v2 feature+target space.\n"
    "2. **Group-split**: on what remains, use `GroupShuffleSplit` grouped by "
    "`video_id` so all of a given video's remaining (date-level) rows land "
    "entirely in train or entirely in val, never split across both.\n"
    "\n"
    "Then rerun both the Linear Regression baseline and the Random Forest on "
    "the corrected data, same v2 feature set, for a clean comparison."
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
    "## Build model_df (same raw columns as before, plus `video_id` and "
    "`trending_date` for dedup/grouping)"
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
    "FEATURE_COLUMNS_RAW_V2 = [c for c in FEATURE_COLUMNS_RAW if c != 'video_trending_country']\n"
    "TARGET_COLUMN = 'video_view_count'\n"
    "KEY_COLUMNS = ['video_id', 'trending_date']\n"
    "\n"
    "model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN] + KEY_COLUMNS].dropna(\n"
    "    subset=FEATURE_COLUMNS_RAW + [TARGET_COLUMN]\n"
    ").copy()\n"
    "print(f\"Rows before dedup: {len(model_df):,} (from {len(df):,})\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Step 1 — Dedup: one row per `(video_id, trending_date)`\n"
    "\n"
    "Sorting by `video_trending_country` before `drop_duplicates(keep='first')` "
    "just makes the kept row deterministic/reproducible — since same-day rows "
    "across countries are exact duplicates in the v2 feature+target space, "
    "which country happens to be kept doesn't matter."
))

cells.append(nbf.v4.new_code_cell(
    "model_df_sorted = model_df.sort_values(['video_id', 'trending_date', 'video_trending_country'])\n"
    "deduped_df = model_df_sorted.drop_duplicates(subset=['video_id', 'trending_date'], keep='first').copy()\n"
    "\n"
    "print(f\"Rows before dedup: {len(model_df):,}\")\n"
    "print(f\"Rows after dedup:  {len(deduped_df):,}\")\n"
    "print(f\"Dropped as country-duplicate rows: {len(model_df) - len(deduped_df):,} \"\n"
    "      f\"({(len(model_df) - len(deduped_df)) / len(model_df):.1%})\")\n"
    "print(f\"Unique video_id in deduped data: {deduped_df['video_id'].nunique():,}\")\n"
    "print(f\"Mean rows per video after dedup (i.e. mean distinct trending dates/video): \"\n"
    "      f\"{len(deduped_df) / deduped_df['video_id'].nunique():.2f}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Encode v2 features — identical column set to the earlier baseline/RF"
))

cells.append(nbf.v4.new_code_cell(
    "X_raw_v2 = deduped_df[FEATURE_COLUMNS_RAW_V2]\n"
    "y = deduped_df[TARGET_COLUMN].astype('float64')\n"
    "groups = deduped_df['video_id']\n"
    "\n"
    "X_encoded = pd.get_dummies(X_raw_v2, columns=['video_category_id'], drop_first=True)\n"
    "FEATURE_COLUMNS = list(X_encoded.columns)\n"
    "print(f\"Feature matrix shape: {X_encoded.shape}\")\n"
    "\n"
    "FEATURES_PATH = MODELS_DIR / 'baseline_feature_columns.json'\n"
    "with open(FEATURES_PATH) as f:\n"
    "    saved_v2_columns = json.load(f)\n"
    "assert FEATURE_COLUMNS == saved_v2_columns, 'Feature column mismatch vs. saved v2 baseline'\n"
    "print('Confirmed: identical column set and order to the v2 baseline.')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Step 2 — GroupShuffleSplit by `video_id`\n"
    "\n"
    "`test_size=0.2, random_state=42` — same proportions as before, but now no "
    "`video_id` appears on both sides of the split."
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
    "## Train Linear Regression on the corrected split"
))

cells.append(nbf.v4.new_code_cell(
    "lr = LinearRegression()\n"
    "lr.fit(X_train, y_train)\n"
    "y_pred_lr = lr.predict(X_val)\n"
    "\n"
    "rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))\n"
    "r2_lr = r2_score(y_val, y_pred_lr)\n"
    "\n"
    "print('=== Linear Regression (deduped + grouped) — validation results ===')\n"
    "print(f'RMSE: {rmse_lr:,.2f} views')\n"
    "print(f'R^2:  {r2_lr:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train Random Forest on the corrected split\n"
    "\n"
    "Same hyperparameters as the leaked run in `03_random_forest.ipynb` "
    "(`n_estimators=200, max_depth=15, n_jobs=-1, random_state=42`) — no "
    "retuning, so the before/after comparison isolates the effect of fixing "
    "the split, not a hyperparameter change."
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
    "print('=== Random Forest (deduped + grouped) — validation results ===')\n"
    "print(f'RMSE: {rmse_rf:,.2f} views')\n"
    "print(f'R^2:  {r2_rf:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Full comparison: leaked (row-random) vs. corrected (dedup + group-split)"
))

cells.append(nbf.v4.new_code_cell(
    "rmse_lr_leaked = 18_932_915.0\n"
    "r2_lr_leaked = 0.0972\n"
    "rmse_rf_leaked = 9_429_240.85\n"
    "r2_rf_leaked = 0.7761\n"
    "\n"
    "print(f'{\"\":42s} {\"RMSE\":>18s} {\"R^2\":>10s}')\n"
    "print(f'{\"Linear Regression — row-random (leaked)\":42s} {rmse_lr_leaked:>18,.2f} {r2_lr_leaked:>10.4f}')\n"
    "print(f'{\"Linear Regression — dedup + grouped\":42s} {rmse_lr:>18,.2f} {r2_lr:>10.4f}')\n"
    "print(f'{\"Random Forest — row-random (leaked)\":42s} {rmse_rf_leaked:>18,.2f} {r2_rf_leaked:>10.4f}')\n"
    "print(f'{\"Random Forest — dedup + grouped\":42s} {rmse_rf:>18,.2f} {r2_rf:>10.4f}')\n"
    "print()\n"
    "print(f'Linear RMSE change: {rmse_lr - rmse_lr_leaked:+,.2f} ({(rmse_lr - rmse_lr_leaked) / rmse_lr_leaked:+.2%})')\n"
    "print(f'Linear R^2 change:  {r2_lr - r2_lr_leaked:+.4f}')\n"
    "print(f'RF RMSE change:     {rmse_rf - rmse_rf_leaked:+,.2f} ({(rmse_rf - rmse_rf_leaked) / rmse_rf_leaked:+.2%})')\n"
    "print(f'RF R^2 change:      {r2_rf - r2_rf_leaked:+.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Save the corrected models"
))

cells.append(nbf.v4.new_code_cell(
    "LR_PATH = MODELS_DIR / 'baseline_linear_regression_v2_dedup_grouped.pkl'\n"
    "RF_PATH = MODELS_DIR / 'random_forest_dedup_grouped.pkl'\n"
    "\n"
    "joblib.dump(lr, LR_PATH)\n"
    "joblib.dump(rf, RF_PATH)\n"
    "print(f'Saved Linear Regression to {LR_PATH}')\n"
    "print(f'Saved Random Forest to {RF_PATH}')\n"
    "print()\n"
    "print('NOTE: models/baseline_linear_regression_v2.pkl and models/random_forest.pkl '\n"
    "      '(the row-random, leaked-split versions) are left in place for reference/comparison, '\n"
    "      'not overwritten. baseline_feature_columns.json is unchanged — same 19 v2 columns '\n"
    "      'apply to both the leaked and corrected models.')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Summary (placeholder — filled in after running)"
))

nb['cells'] = cells

with open('/Users/prabesharyal/ai4all/notebooks/04_dedup_group_split.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
