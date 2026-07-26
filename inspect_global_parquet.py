"""
Schema-safe scan of the large canerkonuk file (youtube_trending_videos_global.parquet, ~3.8GB).

Does NOT load the full file into memory: reads only the two columns we need
(video_trending__date, video_trending_country) via row-group-wise column
projection, which pyarrow pushes down at the I/O level.

Reports:
  1. Full date range (min/max)
  2. Number of unique dates (continuous history vs scattered snapshots)
  3. Row count per year / per few-month bucket
  4. Country values, compared against the daily file's country set
"""

import os

import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")
FULL_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")
DAILY_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global_daily.parquet")

COLS = ["video_trending__date", "video_trending_country"]


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


section("FULL file — metadata")
pf = pq.ParquetFile(FULL_PATH)
meta = pf.metadata
print(f"Row groups: {meta.num_row_groups}")
print(f"Total rows: {meta.num_rows:,}")

section(f"FULL file — reading only columns {COLS} via row-group projection")

dates_all = []
countries_all = []
row_group_sizes = []

for i in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(i, columns=COLS)
    df = tbl.to_pandas()
    row_group_sizes.append(len(df))
    dates_all.append(df["video_trending__date"])
    countries_all.append(df["video_trending_country"])
    if (i + 1) % 10 == 0 or (i + 1) == pf.metadata.num_row_groups:
        print(f"  read row group {i + 1}/{pf.metadata.num_row_groups} "
              f"({sum(row_group_sizes):,} rows so far)")

dates = pd.concat(dates_all, ignore_index=True)
countries = pd.concat(countries_all, ignore_index=True)
del dates_all, countries_all

print(f"\nTotal rows read: {len(dates):,}")
print(f"Row group sizes: min={min(row_group_sizes):,} max={max(row_group_sizes):,} "
      f"mean={sum(row_group_sizes) / len(row_group_sizes):,.0f}")

# ---------------------------------------------------------------------------
# 1. Date range
# ---------------------------------------------------------------------------
section("1. Date range")
parsed_dates = pd.to_datetime(dates, errors="coerce", format="mixed", utc=True)
n_bad = parsed_dates.isna().sum()
print(f"Unparseable dates: {n_bad:,} / {len(parsed_dates):,}")
print(f"Min date: {parsed_dates.min()}")
print(f"Max date: {parsed_dates.max()}")
print(f"Span: {(parsed_dates.max() - parsed_dates.min()).days:,} days")

# ---------------------------------------------------------------------------
# 2. Unique dates
# ---------------------------------------------------------------------------
section("2. Unique dates present")
unique_dates = parsed_dates.dt.date.dropna().unique()
unique_dates_sorted = sorted(unique_dates)
n_unique = len(unique_dates_sorted)
span_days = (unique_dates_sorted[-1] - unique_dates_sorted[0]).days + 1 if n_unique else 0
print(f"Number of unique calendar dates: {n_unique:,}")
print(f"Calendar span (first to last, inclusive): {span_days:,} days")
print(f"Coverage ratio (unique_dates / calendar_span): {n_unique / span_days:.2%}" if span_days else "N/A")

if n_unique <= 40:
    print("\nAll unique dates (scattered-snapshot check):")
    for d in unique_dates_sorted:
        print(f"  {d}")
else:
    print(f"\nFirst 10 unique dates: {unique_dates_sorted[:10]}")
    print(f"Last 10 unique dates:  {unique_dates_sorted[-10:]}")

    # gap analysis — find missing-date runs to see if it's continuous or lumpy
    full_range = pd.date_range(unique_dates_sorted[0], unique_dates_sorted[-1], freq="D").date
    missing = sorted(set(full_range) - set(unique_dates_sorted))
    print(f"\nMissing calendar days within the span: {len(missing):,} / {len(full_range):,}")
    if missing:
        print(f"First 10 missing days: {missing[:10]}")
        print(f"Last 10 missing days:  {missing[-10:]}")

# ---------------------------------------------------------------------------
# 3. Row count per year / per few-month bucket
# ---------------------------------------------------------------------------
section("3. Row count per year")
year_counts = parsed_dates.dt.year.value_counts().sort_index()
print(year_counts.to_string())

section("3b. Row count per quarter (year-Q)")
quarter = parsed_dates.dt.to_period("Q")
quarter_counts = quarter.value_counts().sort_index()
print(quarter_counts.to_string())

section("3c. Row count per month (last 24 months of coverage, if applicable)")
month = parsed_dates.dt.to_period("M")
month_counts = month.value_counts().sort_index()
print(month_counts.tail(24).to_string())
print(f"\n(Total distinct year-month buckets: {len(month_counts)})")

# ---------------------------------------------------------------------------
# 4. Country coverage — compare to daily file
# ---------------------------------------------------------------------------
section("4. Country coverage in FULL file")
full_countries = set(countries.dropna().unique())
print(f"Unique countries in FULL file: {len(full_countries)}")
print(sorted(full_countries))

section("4b. Country coverage in DAILY file (re-check for comparison)")
daily_df = pd.read_parquet(DAILY_PATH, columns=["video_trending_country"])
daily_countries = set(daily_df["video_trending_country"].dropna().unique())
print(f"Unique countries in DAILY file: {len(daily_countries)}")
print(sorted(daily_countries))

section("4c. Diff between FULL and DAILY country sets")
only_in_full = full_countries - daily_countries
only_in_daily = daily_countries - full_countries
print(f"In FULL but not DAILY ({len(only_in_full)}): {sorted(only_in_full)}")
print(f"In DAILY but not FULL ({len(only_in_daily)}): {sorted(only_in_daily)}")
print(f"Match: {'YES — identical sets' if not only_in_full and not only_in_daily else 'NO — sets differ'}")

TARGET_MAP = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany", "CA": "Canada",
    "FR": "France", "RU": "Russia", "MX": "Mexico", "KR": "South Korea",
    "JP": "Japan", "IN": "India",
}
section("4d. Target 10 countries — present in FULL file?")
for code, name in TARGET_MAP.items():
    present = name in full_countries
    print(f"  {code} ({name}): {'present' if present else 'MISSING'}")

print("\n\nDONE — scan complete. No merge/cleaning performed.")
