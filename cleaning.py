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

#####Removing duplicate videos (same video trending in multiple countries on the same day
#####has identical stats copied across rows — keep only the first occurrence per video)
before_dedup = data_clean.shape[0]
data_clean = data_clean.drop_duplicates(subset='video_id', keep='first')
print(f"Rows before dedup: {before_dedup} | Rows after dedup: {data_clean.shape[0]} | Removed: {before_dedup - data_clean.shape[0]}")

#####Converting date strings to datetime object
data_clean['video_published_at'] = pd.to_datetime(data_clean['video_published_at'])
data_clean['channel_published_at'] = pd.to_datetime(
    data_clean['channel_published_at'],
    format='mixed',
    utc=True
)
data_clean['video_trending__date'] = pd.to_datetime(data_clean['video_trending__date'])
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


