"""
Trace the video_trending_country == 'Music' / 'People & Blogs' anomaly found
in data_canerkonuk/_global.parquet.

Reads ALL columns, but row-group by row-group (never the whole 3.8GB file at
once), filtering to only the affected rows so memory stays small.

Reports:
  - Full-row dump of 20-30 affected rows (every column) to inspect for a
    field-swap or row-shift pattern
  - Total affected row count across the full file
  - Whether affected rows cluster by row group (proxy for ingestion batch)
    and/or by trending date
"""

import os

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 240)

BASE = os.path.dirname(os.path.abspath(__file__))
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")
FULL_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")

BAD_VALUES = {"Music", "People & Blogs"}


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


pf = pq.ParquetFile(FULL_PATH)
n_row_groups = pf.metadata.num_row_groups
all_cols = list(pf.schema_arrow.names)

section(f"Scanning all {n_row_groups} row groups, all {len(all_cols)} columns, "
        f"filtering video_trending_country in {BAD_VALUES}")

matched_frames = []
per_rowgroup_counts = []
total_rows_scanned = 0

for i in range(n_row_groups):
    tbl = pf.read_row_group(i)
    df = tbl.to_pandas()
    total_rows_scanned += len(df)
    mask = df["video_trending_country"].isin(BAD_VALUES)
    n_match = int(mask.sum())
    per_rowgroup_counts.append(n_match)
    if n_match:
        matched = df.loc[mask].copy()
        matched["__row_group"] = i
        matched_frames.append(matched)
    if (i + 1) % 10 == 0 or (i + 1) == n_row_groups:
        running_total = sum(per_rowgroup_counts)
        print(f"  row group {i + 1}/{n_row_groups}  "
              f"(scanned {total_rows_scanned:,} rows, affected so far: {running_total:,})")

affected = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame(columns=all_cols)

# ---------------------------------------------------------------------------
# Total affected count
# ---------------------------------------------------------------------------
section("Total affected rows across full file")
print(f"Total rows scanned: {total_rows_scanned:,}")
print(f"Total affected rows: {len(affected):,}")
print(f"Affected fraction: {len(affected) / total_rows_scanned:.5%}")
print("\nBreakdown by bad value:")
print(affected["video_trending_country"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Full-row dump of 20-30 affected rows, every column
# ---------------------------------------------------------------------------
section("Full-row dump — up to 30 affected rows, ALL columns")
sample = affected.head(30)
for idx, row in sample.iterrows():
    print(f"\n--- affected row {idx} (row_group={row['__row_group']}) ---")
    for col in all_cols:
        val = row[col]
        val_str = str(val)
        if len(val_str) > 150:
            val_str = val_str[:150] + "...[truncated]"
        print(f"  {col:32s} = {val_str}")

# ---------------------------------------------------------------------------
# Clustering by row group (ingestion-batch proxy)
# ---------------------------------------------------------------------------
section("Clustering — affected row count per row group")
rg_series = pd.Series(per_rowgroup_counts, name="affected_rows")
nonzero = rg_series[rg_series > 0]
print(f"Row groups with at least 1 affected row: {len(nonzero)} / {n_row_groups}")
print(nonzero.to_string())

# ---------------------------------------------------------------------------
# Clustering by trending date
# ---------------------------------------------------------------------------
section("Clustering — affected rows by video_trending__date")
if "video_trending__date" in affected.columns and len(affected):
    parsed = pd.to_datetime(affected["video_trending__date"], errors="coerce", format="mixed", utc=True)
    date_counts = parsed.dt.date.value_counts().sort_index()
    print(date_counts.to_string())
    print(f"\nUnique dates among affected rows: {parsed.dt.date.nunique()}")
    print(f"Date range among affected rows: {parsed.min()} -> {parsed.max()}")

# ---------------------------------------------------------------------------
# Shift-pattern check: does video_category_id look like a country in these rows?
# ---------------------------------------------------------------------------
section("Shift-pattern check — video_category_id values in affected rows")
if "video_category_id" in affected.columns:
    print(affected["video_category_id"].value_counts().to_string())

section("Shift-pattern check — channel_country values in affected rows (compare to video_trending_country)")
if "channel_country" in affected.columns:
    print(affected[["video_trending_country", "channel_country", "video_category_id"]].head(30).to_string())

print("\n\nDONE — trace complete. No cleaning/merge performed.")
