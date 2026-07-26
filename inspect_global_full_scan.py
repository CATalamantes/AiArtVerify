"""
Schema-safe, column-projected scan of data_canerkonuk/_global.parquet (3.8GB).
Does NOT load the full 28-column file into memory — reads only the 9 columns
needed for this report, row-group by row-group.

Reports:
  1. Date range, unique date count, row count per month, total rows, unique video_id count
  2. Null counts at full scale for the 6 fields flagged earlier from the daily sample
  3. Distinct country/region values in the full file
"""

import os

import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")
FULL_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")

COLS = [
    "video_id",
    "video_trending__date",
    "video_trending_country",
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "video_category_id",
    "channel_view_count",
    "channel_subscriber_count",
]


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


section("FULL file — metadata")
pf = pq.ParquetFile(FULL_PATH)
meta = pf.metadata
n_row_groups = meta.num_row_groups
print(f"Row groups: {n_row_groups}")
print(f"Total rows (from metadata): {meta.num_rows:,}")

section(f"Reading {len(COLS)}/{meta.num_columns} columns via row-group projection: {COLS}")

video_id_set = set()
null_counts = {c: 0 for c in COLS if c not in ("video_id", "video_trending__date", "video_trending_country")}
total_rows = 0
date_chunks = []
country_chunks = []

for i in range(n_row_groups):
    tbl = pf.read_row_group(i, columns=COLS)
    df = tbl.to_pandas()
    total_rows += len(df)

    video_id_set.update(df["video_id"].dropna().tolist())

    for c in null_counts:
        null_counts[c] += int(df[c].isna().sum())

    date_chunks.append(df["video_trending__date"])
    country_chunks.append(df["video_trending_country"])

    if (i + 1) % 10 == 0 or (i + 1) == n_row_groups:
        print(f"  read row group {i + 1}/{n_row_groups}  "
              f"(rows so far: {total_rows:,}, unique video_id so far: {len(video_id_set):,})")

dates = pd.concat(date_chunks, ignore_index=True)
countries = pd.concat(country_chunks, ignore_index=True)
del date_chunks, country_chunks

# ---------------------------------------------------------------------------
# 1. Date range / unique dates / per-month counts / row & video_id totals
# ---------------------------------------------------------------------------
section("1. Date range, unique dates, monthly row counts, totals")

parsed_dates = pd.to_datetime(dates, errors="coerce", format="mixed", utc=True)
n_bad = int(parsed_dates.isna().sum())
print(f"Unparseable dates: {n_bad:,} / {len(parsed_dates):,}")
print(f"Min date: {parsed_dates.min()}")
print(f"Max date: {parsed_dates.max()}")

unique_dates = sorted(parsed_dates.dt.date.dropna().unique())
print(f"\nNumber of unique calendar dates: {len(unique_dates):,}")
print(f"First 5 unique dates: {unique_dates[:5]}")
print(f"Last 5 unique dates:  {unique_dates[-5:]}")

section("1b. Row count per month (full history)")
month = parsed_dates.dt.tz_localize(None).dt.to_period("M")
month_counts = month.value_counts().sort_index()
print(month_counts.to_string())
print(f"\nTotal distinct year-month buckets: {len(month_counts)}")

section("1c. Totals")
print(f"Total row count: {total_rows:,}")
print(f"Total unique video_id count: {len(video_id_set):,}")
print(f"Duplicate rows (by video_id, includes legit multi-day/multi-country trends): "
      f"{total_rows - len(video_id_set):,}")

# ---------------------------------------------------------------------------
# 2. Null counts at full scale
# ---------------------------------------------------------------------------
section("2. Null counts at full scale (vs. daily-sample-file figures reported earlier)")
prior_daily_sample_counts = {
    "video_view_count": 1,
    "video_like_count": 236,
    "video_comment_count": 59,
    "video_category_id": 1,
    "channel_view_count": 49,
    "channel_subscriber_count": 49,
}
print(f"{'column':30s} {'full-file nulls':>18s} {'daily-sample nulls':>20s}")
for c in null_counts:
    print(f"{c:30s} {null_counts[c]:>18,} {prior_daily_sample_counts.get(c, 'n/a'):>20}")

# ---------------------------------------------------------------------------
# 3. Distinct countries
# ---------------------------------------------------------------------------
section("3. Distinct country/region values in FULL file")
distinct_countries = sorted(countries.dropna().unique().tolist())
print(f"Count: {len(distinct_countries)}")
for c in distinct_countries:
    print(f"  {c}")

print("\n\nDONE — scan complete. No cleaning/merge performed.")
