from pathlib import Path
import pandas as pd

filePath = Path('.') / 'data' / 'youtube_trending_videos_global_daily.parquet'
data = pd.read_parquet(filePath)
print(data.head())
print('Data successfully read!')
print("\n")
################DATA CLEANING

print("==== DATA CLEANING ====")

#####Dropping rows with missing values (ignoring channel_custom_url, which we don't need)
data_clean = data.dropna(subset=[c for c in data.columns if c != 'channel_custom_url'])
print(f"Original rows: {data.shape[0]} | Clean rows: {data_clean.shape[0]}")

#####Converting video_trending__date to datetime FIRST — we need real dates to sort on
#####before we can reliably pick each video's earliest trending appearance.
data_clean['video_trending__date'] = pd.to_datetime(data_clean['video_trending__date'])

#####Collapsing to one row per video: keep only the FIRST day each video trended.
#####A video can appear once per country per day, so multiple rows can share the same
#####video_id (different countries/days, different accumulated stats). We care about
#####"how well did it perform when it became trending" — so we sort by date ascending
#####and keep the earliest row per video_id. Explicit sort is required: without it,
#####keep='first' would just keep whatever row happens to appear first in the file,
#####which is not guaranteed to be the earliest date.
before_dedup = data_clean.shape[0]
data_clean = data_clean.sort_values('video_trending__date')
data_clean = data_clean.drop_duplicates(subset='video_id', keep='first')
print(f"Rows before dedup: {before_dedup} | Rows after dedup: {data_clean.shape[0]} | Removed: {before_dedup - data_clean.shape[0]}")

#####Converting remaining date strings to datetime objects
data_clean['video_published_at'] = pd.to_datetime(data_clean['video_published_at'])
data_clean['channel_published_at'] = pd.to_datetime(
    data_clean['channel_published_at'],
    format='mixed',
    utc=True
)
data_clean['video_duration'] = pd.to_timedelta(
    data_clean['video_duration']
        .str.replace('PT', '', regex=False)
        .str.replace('H', 'h', regex=False)
        .str.replace('M', 'm', regex=False)
        .str.replace('S', 's', regex=False))

#####Converting strings to ints
data_clean['video_like_count'] = data_clean['video_like_count'].astype(int)
data_clean['video_view_count'] = data_clean['video_view_count'].astype(int)
data_clean['video_comment_count'] = data_clean['video_comment_count'].astype(int)
data_clean['channel_view_count'] = data_clean['channel_view_count'].astype(int)
data_clean['channel_subscriber_count'] = data_clean['channel_subscriber_count'].astype(int)
data_clean['channel_video_count'] = data_clean['channel_video_count'].astype(int)

#####Saving cleaned data so downstream scripts (featureExtraction.py) can pick it up.
OUTPUT_PATH = "cleaned_data.csv"
data_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {data_clean.shape[0]:,} rows x {data_clean.shape[1]} columns -> {OUTPUT_PATH}")