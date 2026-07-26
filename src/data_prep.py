"""
Cleaning / prep pipeline for the canerkonuk dataset (our sole training source;
datasnaek is set aside, not merged).

Steps: load -> filter to legitimate countries -> cast numeric columns ->
drop nulls in profiled fields -> parse trending date -> time-based train/val
split at the 2026-06-11 cutoff -> write data/processed/{train,val}.parquet.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: this directory name has trailing spaces as created on disk.
INPUT_PATH = os.path.join(ROOT, "data_canerkonuk    ", "youtube_trending_videos_global.parquet")
OUTPUT_DIR = os.path.join(ROOT, "data", "processed")
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train.parquet")
VAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "val.parquet")

DATE_COLUMN = "video_trending__date"

# Allowlist of legitimate country/region names, from the full-file country
# scan (113 distinct values minus the 2 corrupted ones: "Music", "People &
# Blogs", which came from the row-shift bug traced to row groups 39/55/56).
VALID_COUNTRIES = [
    "Algeria", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahrain", "Bangladesh", "Belarus", "Belgium", "Bolivia",
    "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Cambodia", "Canada",
    "Chile", "Colombia", "Costa Rica", "Croatia", "Cyprus", "Czechia",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Estonia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece",
    "Guatemala", "Honduras", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kuwait", "Laos", "Latvia", "Lebanon",
    "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Malaysia", "Malta",
    "Mexico", "Moldova", "Montenegro", "Morocco", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Nigeria", "North Macedonia", "Norway",
    "Oman", "Pakistan", "Panama", "Papua New Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania",
    "Russia", "Saudi Arabia", "Senegal", "Serbia", "Singapore", "Slovakia",
    "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka",
    "Sweden", "Switzerland", "Taiwan", "Tanzania", "Thailand", "Tunisia",
    "Turkey", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Venezuela", "Vietnam", "Yemen", "Zimbabwe",
]

# Numeric-looking columns that are string-typed in the parquet schema.
# video_category_id is deliberately excluded: it holds string category
# labels (e.g. "Music", "Gaming"), not numeric codes, in this dataset.
NUMERIC_COLUMNS = [
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "channel_view_count",
    "channel_subscriber_count",
    "channel_video_count",
]

# Fields profiled earlier (all confirmed <1% null) — drop rather than impute.
NULL_CHECK_COLUMNS = [
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "video_category_id",
    "channel_view_count",
    "channel_subscriber_count",
]

TRAIN_VAL_CUTOFF = pd.Timestamp("2026-06-11")


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():
    row_counts = {}

    section("1. Load")
    df = pd.read_parquet(INPUT_PATH)
    row_counts["0_loaded"] = len(df)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {INPUT_PATH}")

    section("2. Filter video_trending_country to allowlist")
    before = len(df)
    df = df[df["video_trending_country"].isin(VALID_COUNTRIES)]
    row_counts["1_after_country_filter"] = len(df)
    print(f"Rows before: {before:,}  ->  after: {len(df):,}  (dropped {before - len(df):,})")

    section("3. Cast numeric-looking string columns")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        print(f"  cast {col} -> Int64")
    row_counts["2_after_numeric_cast"] = len(df)

    section("4. Drop rows with nulls in profiled fields")
    before = len(df)
    for col in NULL_CHECK_COLUMNS:
        n_null = int(df[col].isna().sum())
        print(f"  {col}: {n_null:,} nulls")
    df = df.dropna(subset=NULL_CHECK_COLUMNS)
    row_counts["3_after_null_drop"] = len(df)
    print(f"Rows before: {before:,}  ->  after: {len(df):,}  (dropped {before - len(df):,})")

    section("5. Parse trending date")
    before = len(df)
    parsed = pd.to_datetime(df[DATE_COLUMN], errors="coerce", format="mixed", utc=True)
    df["trending_date"] = parsed.dt.tz_convert(None).dt.normalize()
    n_unparseable = int(df["trending_date"].isna().sum())
    print(f"Unparseable dates: {n_unparseable:,}")
    df = df.dropna(subset=["trending_date"])
    row_counts["4_after_date_parse"] = len(df)
    print(f"Rows before: {before:,}  ->  after: {len(df):,}  (dropped {before - len(df):,})")

    section("6. Time-based train/val split")
    train_df = df[df["trending_date"] < TRAIN_VAL_CUTOFF].copy()
    val_df = df[df["trending_date"] >= TRAIN_VAL_CUTOFF].copy()
    row_counts["5_train"] = len(train_df)
    row_counts["5_val"] = len(val_df)
    print(f"Cutoff: {TRAIN_VAL_CUTOFF.date()}")
    print(f"Train rows (< cutoff): {len(train_df):,}")
    print(f"Val rows (>= cutoff):  {len(val_df):,}")

    section("7. Write output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_parquet(VAL_OUTPUT_PATH, index=False)
    print(f"Wrote {TRAIN_OUTPUT_PATH}")
    print(f"Wrote {VAL_OUTPUT_PATH}")

    section("8. Final summary")
    print("Row counts at each step:")
    for label, count in row_counts.items():
        print(f"  {label:28s} {count:>12,}")

    print(f"\nFinal train rows: {len(train_df):,}")
    print(f"Final val rows:   {len(val_df):,}")

    print(f"\nTrain date range: {train_df['trending_date'].min().date()} -> {train_df['trending_date'].max().date()}")
    print(f"Val date range:   {val_df['trending_date'].min().date()} -> {val_df['trending_date'].max().date()}")

    expected_train_max = TRAIN_VAL_CUTOFF - pd.Timedelta(days=1)
    train_ok = train_df["trending_date"].max() <= expected_train_max
    val_ok = val_df["trending_date"].min() >= TRAIN_VAL_CUTOFF
    print(f"\nTrain max date <= {expected_train_max.date()} (day before cutoff): {train_ok}")
    print(f"Val min date >= {TRAIN_VAL_CUTOFF.date()} (cutoff): {val_ok}")
    print(f"Split boundary confirmed clean: {train_ok and val_ok}")


if __name__ == "__main__":
    main()
