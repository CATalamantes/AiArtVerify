import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 03 — Random Forest (pre-publish features only, v2 feature set)\n"
    "\n"
    "Same target and same pre-publish feature set as the v2 baseline ablation "
    "in `02_baseline_linear_regression.ipynb` — `video_category_id`, "
    "`tag_count`, `title_length`, `title_has_caps_word`, `publish_hour`, "
    "`publish_dayofweek`. `video_trending_country` stays excluded: the v2 "
    "ablation showed most of its apparent predictive power was the small-market "
    "composition artifact (Malta/Luxembourg/Slovenia trending lists ~30-49% "
    "globally-viral padding), not real signal, and RMSE barely moved (+2.9%) "
    "when it was dropped.\n"
    "\n"
    "**Goal**: test whether a non-linear model (which can capture interactions "
    "and non-linear effects a plain `LinearRegression` can't) does meaningfully "
    "better than the v2 linear baseline (RMSE 18,932,915 / R² 0.0972) on the "
    "exact same features and the exact same train/validation split."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('../src')\n"
    "\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "import joblib\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.ensemble import RandomForestRegressor\n"
    "from sklearn.metrics import mean_squared_error, r2_score\n"
    "from sklearn.model_selection import train_test_split\n"
    "\n"
    "from features import build_features\n"
    "\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.width', 200)\n"
    "\n"
    "MODELS_DIR = Path('../models')\n"
    "FIGURES_DIR = Path('../reports/figures')\n"
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "DATA_PATH = Path('../data/processed/train.parquet')\n"
    "df = pd.read_parquet(DATA_PATH)\n"
    "df = build_features(df)\n"
    "print(f\"Loaded and featurized {len(df):,} rows\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Feature set — identical to the v2 baseline\n"
    "\n"
    "Rebuilt the same way as `02_baseline_linear_regression.ipynb`: start from "
    "the same raw column set (including `video_trending_country`) so the "
    "`dropna()` row filter is identical, then drop country before encoding — "
    "this reproduces the exact same `model_df` row set as the v2 ablation, "
    "which is what makes the train/validation split below reproducible.\n"
    "\n"
    "**Target**: `video_view_count` (raw, untransformed, same as the baseline)."
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
    "\n"
    "model_df = df[FEATURE_COLUMNS_RAW + [TARGET_COLUMN]].dropna().copy()\n"
    "print(f\"Rows after dropping any remaining nulls in the feature/target set: {len(model_df):,} \"\n"
    "      f\"(from {len(df):,})\")\n"
    "\n"
    "X_raw_v2 = model_df[FEATURE_COLUMNS_RAW_V2]\n"
    "y = model_df[TARGET_COLUMN].astype('float64')\n"
    "\n"
    "X_encoded = pd.get_dummies(X_raw_v2, columns=['video_category_id'], drop_first=True)\n"
    "FEATURE_COLUMNS = list(X_encoded.columns)\n"
    "print(f\"Feature matrix shape after one-hot encoding: {X_encoded.shape}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Confirm this matches the saved v2 baseline feature list\n"
    "\n"
    "`models/baseline_feature_columns.json` was overwritten at the end of the "
    "baseline notebook to hold the v2 (no-country) column list. Loading it "
    "here and comparing against the columns just built confirms we're using "
    "the exact same 19 features, in a way that doesn't silently drift if "
    "`build_features` or the raw feature list ever changes."
))

cells.append(nbf.v4.new_code_cell(
    "FEATURES_PATH = MODELS_DIR / 'baseline_feature_columns.json'\n"
    "with open(FEATURES_PATH) as f:\n"
    "    saved_v2_columns = json.load(f)\n"
    "\n"
    "print(f\"Saved v2 baseline feature list: {len(saved_v2_columns)} columns\")\n"
    "print(f\"Freshly built feature list:     {len(FEATURE_COLUMNS)} columns\")\n"
    "assert FEATURE_COLUMNS == saved_v2_columns, (\n"
    "    'Feature column mismatch vs. saved v2 baseline — cannot guarantee an apples-to-apples comparison'\n"
    ")\n"
    "print('Confirmed: identical column set and order to the v2 baseline.')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train/validation split — identical to the v2 baseline\n"
    "\n"
    "Same `model_df` (same `dropna()` row filter, same row order), same "
    "`test_size=0.2, random_state=42` — reproduces the exact same "
    "train/validation row split as the v2 baseline, so RMSE/R² are directly "
    "comparable."
))

cells.append(nbf.v4.new_code_cell(
    "X_train, X_val, y_train, y_val = train_test_split(\n"
    "    X_encoded, y, test_size=0.2, random_state=42\n"
    ")\n"
    "print(f\"Train: {X_train.shape}\")\n"
    "print(f\"Val:   {X_val.shape}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Train RandomForestRegressor\n"
    "\n"
    "No hyperparameter tuning yet — `n_estimators=200` and a modest "
    "`max_depth=15` just to get a real number on 7.9M training rows without "
    "an open-ended training time (a benchmark run of 10 trees at this depth "
    "took ~19s with `n_jobs=-1` on this machine, so 200 trees is a few minutes, "
    "not hours). Revisit depth/estimator count/`min_samples_leaf` once we know "
    "whether the non-linear model is even worth tuning further."
))

cells.append(nbf.v4.new_code_cell(
    "rf = RandomForestRegressor(\n"
    "    n_estimators=200,\n"
    "    max_depth=15,\n"
    "    n_jobs=-1,\n"
    "    random_state=42,\n"
    ")\n"
    "rf.fit(X_train, y_train)\n"
    "print('Trained.')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Evaluate on the validation split, compare against the v2 linear baseline"
))

cells.append(nbf.v4.new_code_cell(
    "y_pred = rf.predict(X_val)\n"
    "\n"
    "rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred))\n"
    "r2_rf = r2_score(y_val, y_pred)\n"
    "\n"
    "rmse_v2_baseline = 18_932_915.0\n"
    "r2_v2_baseline = 0.0972\n"
    "\n"
    "print('=== Random Forest — validation results ===')\n"
    "print(f'RMSE: {rmse_rf:,.2f} views')\n"
    "print(f'R^2:  {r2_rf:.4f}')\n"
    "print()\n"
    "print('=== Comparison: v2 Linear Regression baseline vs. Random Forest ===')\n"
    "print(f'{\"\":24s} {\"RMSE\":>18s} {\"R^2\":>10s}')\n"
    "print(f'{\"v2 Linear Regression\":24s} {rmse_v2_baseline:>18,.2f} {r2_v2_baseline:>10.4f}')\n"
    "print(f'{\"Random Forest\":24s} {rmse_rf:>18,.2f} {r2_rf:>10.4f}')\n"
    "print()\n"
    "print(f'RMSE change: {rmse_rf - rmse_v2_baseline:+,.2f} '\n"
    "      f'({(rmse_rf - rmse_v2_baseline) / rmse_v2_baseline:+.2%})')\n"
    "print(f'R^2 change:  {r2_rf - r2_v2_baseline:+.4f} '\n"
    "      f'({(r2_rf - r2_v2_baseline) / r2_v2_baseline:+.2%} relative)')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Feature importances\n"
    "\n"
    "Translating the one-hot dummy column names into readable labels before "
    "plotting — `video_category_id_Pets & Animals` becomes "
    "`Category: Pets & Animals`, the five plain numeric/binary features get "
    "short human labels."
))

cells.append(nbf.v4.new_code_cell(
    "NUMERIC_LABELS = {\n"
    "    'tag_count': 'Tag count',\n"
    "    'title_length': 'Title length',\n"
    "    'title_has_caps_word': 'Title has ALL-CAPS word',\n"
    "    'publish_hour': 'Publish hour',\n"
    "    'publish_dayofweek': 'Publish day of week',\n"
    "}\n"
    "\n"
    "\n"
    "def readable_label(col: str) -> str:\n"
    "    if col in NUMERIC_LABELS:\n"
    "        return NUMERIC_LABELS[col]\n"
    "    if col.startswith('video_category_id_'):\n"
    "        return f\"Category: {col[len('video_category_id_'):]}\"\n"
    "    return col\n"
    "\n"
    "\n"
    "importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)\n"
    "importances.index = [readable_label(c) for c in importances.index]\n"
    "importances = importances.sort_values(ascending=False)\n"
    "\n"
    "print('=== Feature importances (all 19, descending) ===')\n"
    "print(importances.to_string(float_format=lambda v: f'{v:.4f}'))"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(9, 7))\n"
    "importances.sort_values().plot(kind='barh', ax=ax, color='#4C72B0')\n"
    "ax.set_xlabel('Feature importance (mean decrease in impurity)')\n"
    "ax.set_title('Random Forest — feature importances (v2 pre-publish feature set)')\n"
    "fig.tight_layout()\n"
    "\n"
    "fig_path = FIGURES_DIR / 'rf_feature_importance.png'\n"
    "fig.savefig(fig_path, dpi=150)\n"
    "print(f'Saved figure to {fig_path}')\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Save the trained model"
))

cells.append(nbf.v4.new_code_cell(
    "MODEL_PATH = MODELS_DIR / 'random_forest.pkl'\n"
    "joblib.dump(rf, MODEL_PATH)\n"
    "print(f'Saved model to {MODEL_PATH}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Summary (placeholder — filled in after running)"
))

nb['cells'] = cells

with open('/Users/prabesharyal/ai4all/notebooks/03_random_forest.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
