"""
Schema-safe scan (column-projected, row-group by row-group) of
data_canerkonuk/_global.parquet to get daily row counts near the tail of the
dataset, to evaluate candidate time-based train/validation cutoffs.
"""

import os

import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")
FULL_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")

pf = pq.ParquetFile(FULL_PATH)
n_row_groups = pf.metadata.num_row_groups

chunks = []
for i in range(n_row_groups):
    tbl = pf.read_row_group(i, columns=["video_trending__date"])
    chunks.append(tbl.to_pandas()["video_trending__date"])

dates = pd.concat(chunks, ignore_index=True)
parsed = pd.to_datetime(dates, errors="coerce", format="mixed", utc=True)
day = parsed.dt.date
day_valid = day.dropna()

daily_counts = day.value_counts().sort_index()
daily_counts = daily_counts[(daily_counts.index >= pd.Timestamp("2024-10-01").date())]

print("=" * 90)
print("Daily row counts, last 10 weeks of coverage")
print("=" * 90)
print(daily_counts.tail(70).to_string())

total_rows = len(dates)
max_date = day_valid.max()
print(f"\nMax date in data: {max_date}")
print(f"Total rows (all time, including junk): {total_rows:,}")

print("\n" + "=" * 90)
print("Candidate cutoffs — rows before vs. on/after cutoff")
print("=" * 90)

candidates = {
    "6 weeks back (cutoff 2026-06-04)": pd.Timestamp("2026-06-04").date(),
    "5 weeks back (cutoff 2026-06-11)": pd.Timestamp("2026-06-11").date(),
    "4 weeks back (cutoff 2026-06-18)": pd.Timestamp("2026-06-18").date(),
}

for label, cutoff in candidates.items():
    train_rows = int((day_valid < cutoff).sum())
    val_rows = int((day_valid >= cutoff).sum())
    val_days = (max_date - cutoff).days + 1
    print(f"\n{label}")
    print(f"  train rows (< {cutoff}): {train_rows:,}  ({train_rows / total_rows:.1%})")
    print(f"  val rows   (>= {cutoff} through {max_date}): {val_rows:,}  ({val_rows / total_rows:.1%})")
    print(f"  val window: {val_days} days")
    print(f"  val rows/day avg: {val_rows / val_days:,.0f}")

print("\n\nDONE.")
