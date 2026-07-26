import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 01 — Exploratory Data Analysis: canerkonuk (train)\n"
    "\n"
    "Loads the cleaned `data/processed/train.parquet` (canerkonuk, our sole training "
    "source), applies `build_features` from `src/features.py`, and looks at "
    "distributions, correlations with `video_view_count`, and category/country "
    "breakdowns. The category and country charts are also saved to "
    "`reports/figures/` for the geographic bias write-up."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('../src')\n"
    "\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "\n"
    "from features import build_features\n"
    "\n"
    "pd.set_option('display.max_columns', None)\n"
    "sns.set_theme(style='whitegrid')\n"
    "\n"
    "FIG_DIR = Path('../reports/figures')\n"
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "DATA_PATH = Path('../data/processed/train.parquet')\n"
    "df = pd.read_parquet(DATA_PATH)\n"
    "df = build_features(df)\n"
    "print(f\"Loaded and featurized {len(df):,} rows\")\n"
    "df.head(3)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Distribution of view / like / comment counts (log scale)\n"
    "\n"
    "These are heavily right-skewed — a log10 scale is needed to see the shape "
    "(consistent with the extreme long tail we already saw in "
    "`channel_engagement_rate` during feature engineering)."
))

cells.append(nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
    "count_cols = ['video_view_count', 'video_like_count', 'video_comment_count']\n"
    "titles = ['Views', 'Likes', 'Comments']\n"
    "\n"
    "for ax, col, title in zip(axes, count_cols, titles):\n"
    "    values = df[col].dropna().astype('float64')\n"
    "    values = values[values > 0]\n"
    "    ax.hist(np.log10(values), bins=80, color='#4C72B0', edgecolor='none')\n"
    "    ax.set_title(f'{title} distribution (log10 scale)')\n"
    "    ax.set_xlabel(f'log10({title.lower()})')\n"
    "    ax.set_ylabel('video-trending-day rows')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'count_distributions_log.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Correlation of features with `video_view_count`"
))

cells.append(nbf.v4.new_code_cell(
    "corr_cols = [\n"
    "    'video_view_count', 'video_like_count', 'video_comment_count',\n"
    "    'days_to_trend', 'tag_count', 'title_length', 'comment_rate',\n"
    "    'channel_engagement_rate', 'publish_hour', 'publish_dayofweek',\n"
    "]\n"
    "corr_matrix = df[corr_cols].astype('float64').corr()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 7))\n"
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)\n"
    "ax.set_title('Correlation matrix — engineered features vs. video_view_count')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print('Correlation with video_view_count, sorted:')\n"
    "print(corr_matrix['video_view_count'].sort_values(ascending=False))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Views by `video_category_id`\n"
    "\n"
    "Category is already a human-readable string in this dataset (not a numeric "
    "code), so labels are directly usable. Using median-per-category bars rather "
    "than a raw boxplot — with ~10M rows, medians aggregate cleanly and are more "
    "legible than an overplotted boxplot."
))

cells.append(nbf.v4.new_code_cell(
    "category_medians = (\n"
    "    df.groupby('video_category_id', observed=True)['video_view_count']\n"
    "    .median()\n"
    "    .sort_values(ascending=False)\n"
    ")\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, max(6, len(category_medians) * 0.3)))\n"
    "category_medians.plot(kind='barh', ax=ax, color='#55A868')\n"
    "ax.invert_yaxis()\n"
    "ax.set_xlabel('Median video_view_count')\n"
    "ax.set_ylabel('video_category_id')\n"
    "ax.set_title('Median views by category (sorted descending)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'views_by_category.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "category_medians.head(10)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Views by `video_trending_country`\n"
    "\n"
    "Saved as a standalone figure for the geographic bias write-up."
))

cells.append(nbf.v4.new_code_cell(
    "country_medians = (\n"
    "    df.groupby('video_trending_country', observed=True)['video_view_count']\n"
    "    .median()\n"
    "    .sort_values(ascending=False)\n"
    ")\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, max(8, len(country_medians) * 0.18)))\n"
    "country_medians.plot(kind='barh', ax=ax, color='#C44E52')\n"
    "ax.invert_yaxis()\n"
    "ax.set_xlabel('Median video_view_count')\n"
    "ax.set_ylabel('video_trending_country')\n"
    "ax.set_title('Median views by trending country (sorted descending)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'views_by_country.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "country_medians.head(10)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. `days_to_trend` distribution\n"
    "\n"
    "How fast videos trend after publishing, and whether the small negative tail "
    "we flagged during feature engineering (clock skew, min ≈ -0.77 days) is a "
    "meaningful bump or genuinely negligible."
))

cells.append(nbf.v4.new_code_cell(
    "values = df['days_to_trend'].dropna()\n"
    "pct_negative = (values < 0).mean()\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "\n"
    "axes[0].hist(values, bins=100, range=(values.quantile(0.001), values.quantile(0.999)), color='#8172B2')\n"
    "axes[0].set_title('days_to_trend distribution (0.1%-99.9% range)')\n"
    "axes[0].set_xlabel('days_to_trend')\n"
    "axes[0].set_ylabel('rows')\n"
    "\n"
    "negative_tail = values[values < 0]\n"
    "if len(negative_tail):\n"
    "    axes[1].hist(negative_tail, bins=40, color='#CCB974')\n"
    "axes[1].set_title(f'Negative tail only (n={len(negative_tail):,}, {pct_negative:.4%} of rows)')\n"
    "axes[1].set_xlabel('days_to_trend (negative only)')\n"
    "axes[1].set_ylabel('rows')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'days_to_trend_distribution.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f\"Rows with negative days_to_trend: {len(negative_tail):,} ({pct_negative:.4%} of {len(values):,})\")\n"
    "print(f\"Min: {values.min():.3f}  Max: {values.max():.3f}  Median: {values.median():.3f}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. `publish_hour` and `publish_dayofweek` distributions\n"
    "\n"
    "Is there a visible pattern in when trending videos get published, or does "
    "it look roughly uniform?"
))

cells.append(nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "\n"
    "hour_counts = df['publish_hour'].value_counts().sort_index()\n"
    "axes[0].bar(hour_counts.index, hour_counts.values, color='#4C72B0')\n"
    "axes[0].set_title('Publish hour (UTC) of trending videos')\n"
    "axes[0].set_xlabel('Hour of day')\n"
    "axes[0].set_ylabel('rows')\n"
    "axes[0].set_xticks(range(0, 24, 2))\n"
    "\n"
    "day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
    "day_counts = df['publish_dayofweek'].value_counts().sort_index()\n"
    "axes[1].bar([day_labels[i] for i in day_counts.index], day_counts.values, color='#55A868')\n"
    "axes[1].set_title('Publish day-of-week of trending videos')\n"
    "axes[1].set_xlabel('Day of week')\n"
    "axes[1].set_ylabel('rows')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print('Publish hour value counts (top 5):')\n"
    "print(hour_counts.sort_values(ascending=False).head())\n"
    "print()\n"
    "print('Publish day-of-week value counts:')\n"
    "print(day_counts.rename(index=dict(enumerate(day_labels))))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. Summary — key numbers for the write-up\n"
    "\n"
    "(Computed here so the plain-language summary below is drawn from actual "
    "output, not estimated.)"
))

cells.append(nbf.v4.new_code_cell(
    "print('Top 3 categories by median views:')\n"
    "print(category_medians.head(3))\n"
    "print()\n"
    "print('Bottom 3 categories by median views:')\n"
    "print(category_medians.tail(3))\n"
    "print()\n"
    "print('Top 5 countries by median views:')\n"
    "print(country_medians.head(5))\n"
    "print()\n"
    "print('Bottom 5 countries by median views:')\n"
    "print(country_medians.tail(5))\n"
    "print()\n"
    "corr_no_self = corr_matrix['video_view_count'].drop('video_view_count')\n"
    "print('Correlation with views, ranked by |correlation|:')\n"
    "print(corr_no_self.reindex(corr_no_self.abs().sort_values(ascending=False).index))\n"
    "print()\n"
    "print(f'% negative days_to_trend: {pct_negative:.4%}')\n"
    "print(f'Median days_to_trend: {values.median():.2f}')\n"
    "print(f'Peak publish hour: {hour_counts.idxmax()} ({hour_counts.max():,} rows)')\n"
    "print(f'Peak publish day: {day_labels[day_counts.idxmax()]} ({day_counts.max():,} rows)')\n"
    "print(f'Views/category max-to-min ratio: {category_medians.max() / category_medians.min():.1f}x')\n"
    "print(f'Views/country max-to-min ratio: {country_medians.max() / country_medians.min():.1f}x')"
))

# Placeholder — filled in with real numbers after execution.
cells.append(nbf.v4.new_markdown_cell("## Key takeaways\n\n_(placeholder — filled in after running the notebook)_"))

nb['cells'] = cells

with open('/Users/prabesharyal/ai4all/notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
