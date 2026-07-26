import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 60)

cols = [
    'video_id', 'video_title', 'video_trending_country', 'trending_date',
    'video_view_count', 'video_like_count', 'video_comment_count',
]
df = pd.read_parquet('/Users/prabesharyal/ai4all/data/processed/train.parquet', columns=cols)
print(f"Loaded {len(df):,} rows")

# Find videos that trended in the most countries/rows, to get a clear multi-row example
counts_per_video = df.groupby('video_id').size().sort_values(ascending=False)
print()
print("=== Top 10 videos by row count (most country-appearances) ===")
print(counts_per_video.head(10))

top_video_id = counts_per_video.index[0]
print()
print(f"=== All rows for top video_id={top_video_id!r} ===")
sub = df[df['video_id'] == top_video_id].sort_values(['video_trending_country', 'trending_date'])
print(sub.to_string(index=False))

# Also specifically look for a "ROSÉ"/"Bruno Mars"/"APT" style global mega-hit by searching titles
mask = df['video_title'].str.contains('APT', case=False, na=False) & df['video_title'].str.contains('ROS', case=False, na=False)
matches = df.loc[mask, 'video_id'].unique()
print()
print(f"=== Titles matching APT/ROS(É) search: {len(matches)} unique video_id(s) ===")
if len(matches) > 0:
    vid = matches[0]
    sub2 = df[df['video_id'] == vid].sort_values(['video_trending_country', 'trending_date'])
    print(f"video_id={vid!r}, title={sub2['video_title'].iloc[0]!r}")
    print(f"Rows for this video: {len(sub2)}")
    print(sub2.to_string(index=False))

# General stat: within each video_id, how often does video_view_count actually vary across rows?
print()
print("=== Does video_view_count vary within a video_id, across its rows? ===")
nunique_view = df.groupby('video_id')['video_view_count'].nunique()
multi_row_videos = counts_per_video[counts_per_video > 1].index
nunique_view_multi = nunique_view.loc[multi_row_videos]
print(f"Videos with >1 row: {len(multi_row_videos):,}")
print(f"Of those, videos where ALL rows share the exact same video_view_count: "
      f"{(nunique_view_multi == 1).sum():,} ({(nunique_view_multi == 1).mean():.1%})")
print(f"Of those, videos where video_view_count DIFFERS across rows: "
      f"{(nunique_view_multi > 1).sum():,} ({(nunique_view_multi > 1).mean():.1%})")
