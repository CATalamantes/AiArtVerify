"""
STEP 1 — Inspection only. No merging, no cleaning.

Inspects the two raw YouTube datasets:
  - data_datasnaek  /  (per-country CSVs, legacy, ~2017-2018)
  - data_canerkonuk    /  (parquet, global, more recent)

Prints schemas, dtypes, sample rows, date ranges, country coverage,
and basic data-quality flags (dupes, nulls, encoding issues).
"""

import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE = os.path.dirname(os.path.abspath(__file__))
# NOTE: these directory names have trailing spaces as created on disk.
DATASNAEK_DIR = os.path.join(BASE, "data_datasnaek  ")
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")

TARGET_COUNTRIES = {
    "US", "GB", "DE", "CA", "FR", "RU", "MX", "KR", "JP", "IN"
}
TARGET_COUNTRIES_HUMAN = (
    "US (United States), GB (UK), DE (Germany), CA (Canada), FR (France), "
    "RU (Russia), MX (Mexico), KR (South Korea), JP (Japan), IN (India)"
)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# DATASNAEK — list files
# ---------------------------------------------------------------------------
section("DATASNAEK — directory listing")
datasnaek_files = sorted(os.listdir(DATASNAEK_DIR))
for f in datasnaek_files:
    path = os.path.join(DATASNAEK_DIR, f)
    print(f"  {f:30s} {os.path.getsize(path):,} bytes")

csv_files = sorted(f for f in datasnaek_files if f.endswith(".csv"))
json_files = sorted(f for f in datasnaek_files if f.endswith(".json"))
print(f"\n{len(csv_files)} per-country CSVs, {len(json_files)} category JSON files")

# ---------------------------------------------------------------------------
# DATASNAEK — full read of one CSV (US, likely most standard)
# ---------------------------------------------------------------------------
section("DATASNAEK — USvideos.csv — full read, columns/dtypes/sample")

us_path = os.path.join(DATASNAEK_DIR, "USvideos.csv")
# datasnaek CSVs are known to have latin-1/mixed encoding issues in titles/tags
try:
    us_df = pd.read_csv(us_path, encoding="utf-8")
    us_encoding_used = "utf-8"
except UnicodeDecodeError as e:
    print(f"utf-8 failed ({e}); retrying with latin-1")
    us_df = pd.read_csv(us_path, encoding="latin-1")
    us_encoding_used = "latin-1"

print(f"Encoding used: {us_encoding_used}")
print(f"Shape: {us_df.shape}")
print("\nColumns and dtypes:")
print(us_df.dtypes)
print("\nSample rows:")
print(us_df.head(3).to_string())

# ---------------------------------------------------------------------------
# DATASNAEK — spot-check schema consistency across a few other countries
# ---------------------------------------------------------------------------
section("DATASNAEK — schema spot-check across countries")

spot_check = ["GB", "DE", "JP"]
us_cols = list(us_df.columns)
for cc in spot_check:
    path = os.path.join(DATASNAEK_DIR, f"{cc}videos.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8", nrows=5000)
        enc = "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", nrows=5000)
        enc = "latin-1"
    same = list(df.columns) == us_cols
    print(f"  {cc}videos.csv  encoding={enc:8s} columns_match_US={same}")
    if not same:
        print(f"    US cols:  {us_cols}")
        print(f"    {cc} cols: {list(df.columns)}")

# ---------------------------------------------------------------------------
# DATASNAEK — date range (across ALL countries, using trending_date col)
# ---------------------------------------------------------------------------
section("DATASNAEK — date range across all countries")

date_col_candidates = [c for c in us_df.columns if "date" in c.lower()]
print(f"Date-like columns found: {date_col_candidates}")

all_dates = []
country_video_ids = {}
null_report = {}
for f in csv_files:
    cc = f.replace("videos.csv", "")
    path = os.path.join(DATASNAEK_DIR, f)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # trending_date in this dataset is typically format yy.dd.mm
    if "trending_date" in df.columns:
        parsed = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")
        n_bad = parsed.isna().sum()
        all_dates.append((cc, parsed.min(), parsed.max(), n_bad, len(df)))

    if "video_id" in df.columns:
        dup_count = df["video_id"].duplicated().sum()
        country_video_ids[cc] = (len(df), df["video_id"].nunique(), dup_count)

    key_fields = [c for c in ["video_id", "title", "views", "likes", "dislikes",
                               "comment_count", "trending_date"] if c in df.columns]
    null_report[cc] = {c: int(df[c].isna().sum()) for c in key_fields}

print("\nPer-country trending_date range (parsed with format %y.%d.%m):")
for cc, mn, mx, n_bad, n_rows in all_dates:
    print(f"  {cc}: {mn.date() if pd.notna(mn) else 'NaT'} -> {mx.date() if pd.notna(mx) else 'NaT'}  "
          f"(unparseable dates: {n_bad}/{n_rows})")

overall_min = min(mn for _, mn, _, _, _ in all_dates if pd.notna(mn))
overall_max = max(mx for _, _, mx, _, _ in all_dates if pd.notna(mx))
print(f"\nDATASNAEK overall date range: {overall_min.date()} -> {overall_max.date()}")

# ---------------------------------------------------------------------------
# DATASNAEK — duplicate video IDs
# ---------------------------------------------------------------------------
section("DATASNAEK — duplicate video_id within each country file")
for cc, (n_rows, n_unique, n_dup) in country_video_ids.items():
    print(f"  {cc}: rows={n_rows:,}  unique_video_id={n_unique:,}  duplicated_rows={n_dup:,}")
print("\n(Note: a video trending on multiple days legitimately produces repeat "
      "video_ids with different trending_date — that is expected, not a bug. "
      "True duplicates would be identical video_id + trending_date pairs.)")

for cc in country_video_ids:
    path = os.path.join(DATASNAEK_DIR, f"{cc}videos.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if "video_id" in df.columns and "trending_date" in df.columns:
        exact_dupes = df.duplicated(subset=["video_id", "trending_date"]).sum()
        print(f"  {cc}: exact (video_id, trending_date) duplicate rows = {exact_dupes:,}")

# ---------------------------------------------------------------------------
# DATASNAEK — null report on key fields
# ---------------------------------------------------------------------------
section("DATASNAEK — null counts in key fields, per country")
null_df = pd.DataFrame(null_report).T
print(null_df)

# ---------------------------------------------------------------------------
# DATASNAEK — category id JSON structure (sample)
# ---------------------------------------------------------------------------
section("DATASNAEK — category JSON structure (US sample)")
with open(os.path.join(DATASNAEK_DIR, "US_category_id.json"), "r", encoding="utf-8") as fh:
    cat_json = json.load(fh)
print(f"Top-level keys: {list(cat_json.keys())}")
print(f"Number of category items: {len(cat_json.get('items', []))}")
if cat_json.get("items"):
    print("Sample item:")
    print(json.dumps(cat_json["items"][0], indent=2))

# ---------------------------------------------------------------------------
# DATASNAEK — country values present (derived from filenames, since there's
# no explicit country column in the per-country files)
# ---------------------------------------------------------------------------
section("DATASNAEK — country coverage")
datasnaek_countries = set(cc for cc in country_video_ids)
print(f"Countries present (from filenames): {sorted(datasnaek_countries)}")
print(f"Target 10 countries: {TARGET_COUNTRIES_HUMAN}")
missing = TARGET_COUNTRIES - datasnaek_countries
extra = datasnaek_countries - TARGET_COUNTRIES
print(f"Missing from target list: {sorted(missing) if missing else 'none'}")
print(f"Present but not in target list: {sorted(extra) if extra else 'none'}")


# ===========================================================================
# CANERKONUK — parquet inspection (schema/metadata only, then small sample)
# ===========================================================================
section("CANERKONUK — directory listing")
canerkonuk_files = sorted(os.listdir(CANERKONUK_DIR))
for f in canerkonuk_files:
    path = os.path.join(CANERKONUK_DIR, f)
    print(f"  {f:45s} {os.path.getsize(path):,} bytes")

daily_path = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global_daily.parquet")
full_path = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")

for label, path in [("DAILY (smaller)", daily_path), ("FULL (3.8GB)", full_path)]:
    section(f"CANERKONUK — {label} — schema/metadata only (no full load)")
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"File: {os.path.basename(path)}")
    print(f"Num row groups: {meta.num_row_groups}")
    print(f"Total rows: {meta.num_rows:,}")
    print(f"Total columns: {meta.num_columns}")
    print("\nSchema:")
    print(pf.schema_arrow)

section("CANERKONUK — DAILY file — small sample read (first row group)")
pf_daily = pq.ParquetFile(daily_path)
first_batch = next(pf_daily.iter_batches(batch_size=2000))
sample_df = first_batch.to_pandas()
print(f"Sample shape: {sample_df.shape}")
print("\nColumns and dtypes:")
print(sample_df.dtypes)
print("\nSample rows:")
print(sample_df.head(3).to_string())

# ---------------------------------------------------------------------------
# CANERKONUK — date range (full daily file — it's only ~13MB so a full read
# is cheap; we avoid full-loading only the 3.8GB file)
# ---------------------------------------------------------------------------
section("CANERKONUK — DAILY file — full read for date range / country / dupes / nulls")
daily_df = pd.read_parquet(daily_path)
print(f"Full shape: {daily_df.shape}")

date_cols = [c for c in daily_df.columns if "date" in c.lower() or "published" in c.lower() or "trending" in c.lower()]
print(f"Date-like columns found: {date_cols}")

if "video_trending__date" in daily_df.columns:
    trending_dates = pd.to_datetime(daily_df["video_trending__date"], errors="coerce", utc=True)
    print(f"\nvideo_trending__date range: {trending_dates.min()} -> {trending_dates.max()}")
    print(f"Unparseable trending dates: {trending_dates.isna().sum()}")

if "video_published_at" in daily_df.columns:
    pub_dates = pd.to_datetime(daily_df["video_published_at"], errors="coerce", utc=True)
    print(f"video_published_at range: {pub_dates.min()} -> {pub_dates.max()}")

# ---------------------------------------------------------------------------
# CANERKONUK — country coverage
# ---------------------------------------------------------------------------
section("CANERKONUK — country coverage")
country_cols = [c for c in daily_df.columns if "country" in c.lower()]
print(f"Country-like columns found: {country_cols}")
for c in country_cols:
    vals = sorted(daily_df[c].dropna().unique().tolist())
    print(f"\n{c} — {len(vals)} unique values:")
    print(f"  {vals}")
    if c == "video_trending_country":
        vals_set = set(vals)
        missing = TARGET_COUNTRIES - vals_set
        extra = vals_set - TARGET_COUNTRIES
        print(f"  Missing from target 10: {sorted(missing) if missing else 'none'}")
        print(f"  Present but not in target 10: {sorted(extra) if extra else 'none'}")

# ---------------------------------------------------------------------------
# CANERKONUK — duplicate video IDs
# ---------------------------------------------------------------------------
section("CANERKONUK — duplicate video_id checks")
if "video_id" in daily_df.columns:
    n_rows = len(daily_df)
    n_unique = daily_df["video_id"].nunique()
    n_dup_rows = daily_df["video_id"].duplicated().sum()
    print(f"rows={n_rows:,}  unique_video_id={n_unique:,}  duplicated_rows={n_dup_rows:,}")
    subset_cols = [c for c in ["video_id", "video_trending__date", "video_trending_country"] if c in daily_df.columns]
    if len(subset_cols) > 1:
        exact_dupes = daily_df.duplicated(subset=subset_cols).sum()
        print(f"Exact duplicate rows on {subset_cols}: {exact_dupes:,}")

# ---------------------------------------------------------------------------
# CANERKONUK — null report on key fields
# ---------------------------------------------------------------------------
section("CANERKONUK — null counts in key fields")
key_fields = [c for c in [
    "video_id", "video_title", "video_view_count", "video_like_count",
    "video_comment_count", "video_published_at", "video_trending__date",
    "video_trending_country", "video_category_id", "video_duration",
    "channel_view_count", "channel_subscriber_count"
] if c in daily_df.columns]
print(daily_df[key_fields].isna().sum())

print("\n\nDONE — inspection complete. No merge/cleaning performed.")
