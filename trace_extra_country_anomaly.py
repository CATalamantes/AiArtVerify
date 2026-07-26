"""
Trace the additional rows dropped by the country allowlist filter beyond the
7 already-traced "Music" / "People & Blogs" rows.

Reads ALL columns, row-group by row-group (never the whole 3.8GB file at
once), keeping only rows whose video_trending_country is NOT in the allowlist
AND NOT one of the 2 previously-traced bad values.
"""

import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_prep import VALID_COUNTRIES  # noqa: E402

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 240)

BASE = os.path.dirname(os.path.abspath(__file__))
CANERKONUK_DIR = os.path.join(BASE, "data_canerkonuk    ")
FULL_PATH = os.path.join(CANERKONUK_DIR, "youtube_trending_videos_global.parquet")

PREVIOUSLY_TRACED = {"Music", "People & Blogs"}
VALID_SET = set(VALID_COUNTRIES)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


pf = pq.ParquetFile(FULL_PATH)
n_row_groups = pf.metadata.num_row_groups
all_cols = list(pf.schema_arrow.names)

section(f"Scanning all {n_row_groups} row groups for invalid countries NOT already traced")

matched_frames = []
per_rowgroup_counts = []
total_rows_scanned = 0
all_bad_values_seen = {}

for i in range(n_row_groups):
    tbl = pf.read_row_group(i)
    df = tbl.to_pandas()
    total_rows_scanned += len(df)

    invalid_mask = ~df["video_trending_country"].isin(VALID_SET)
    new_mask = invalid_mask & ~df["video_trending_country"].isin(PREVIOUSLY_TRACED)
    n_match = int(new_mask.sum())
    per_rowgroup_counts.append(n_match)

    if invalid_mask.any():
        for val, cnt in df.loc[invalid_mask, "video_trending_country"].value_counts().items():
            all_bad_values_seen[val] = all_bad_values_seen.get(val, 0) + int(cnt)

    if n_match:
        matched = df.loc[new_mask].copy()
        matched["__row_group"] = i
        matched_frames.append(matched)

    if (i + 1) % 10 == 0 or (i + 1) == n_row_groups:
        running_total = sum(per_rowgroup_counts)
        print(f"  row group {i + 1}/{n_row_groups}  "
              f"(scanned {total_rows_scanned:,} rows, new-anomaly rows so far: {running_total:,})")

affected = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame(columns=all_cols)

section("All invalid country values seen (should total 15: 7 previously traced + N new)")
for val, cnt in sorted(all_bad_values_seen.items(), key=lambda x: -x[1]):
    tag = "PREVIOUSLY TRACED" if val in PREVIOUSLY_TRACED else "NEW"
    print(f"  {val!r}: {cnt} rows  [{tag}]")

section("Total new (previously untraced) affected rows")
print(f"Total: {len(affected):,}")

section("Full-row dump — all new affected rows, ALL columns")
for idx, row in affected.iterrows():
    print(f"\n--- affected row {idx} (row_group={row['__row_group']}) ---")
    for col in all_cols:
        val = row[col]
        val_str = str(val)
        if len(val_str) > 150:
            val_str = val_str[:150] + "...[truncated]"
        print(f"  {col:32s} = {val_str}")

section("Clustering — new affected rows per row group")
rg_series = pd.Series(per_rowgroup_counts, name="affected_rows")
nonzero = rg_series[rg_series > 0]
print(nonzero.to_string())
print(f"\nCompare to previously traced row groups: {{39, 55, 56}}")
print(f"Overlap with previously traced row groups: {set(nonzero.index) & {39, 55, 56}}")
print(f"New row groups: {set(nonzero.index) - {39, 55, 56}}")

section("Shift-pattern check — video_id values (are these the same 2 source videos?)")
if len(affected):
    print(affected["video_id"].value_counts().to_string())

print("\n\nDONE — trace complete.")
