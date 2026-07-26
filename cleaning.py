from pathlib import Path
import pandas as pd

filePath = Path('.') / 'data' / 'youtube_trending_videos_global_daily.parquet'
data = pd.read_parquet(filePath)
print(data.head())
print('Data successfully read!')
print("\n")
################DATA CLEANING

print("==== DATA CLEANING ====")
data_clean = data.dropna()
print(f"Original rows: {data.shape[0]} | Clean rows: {data_clean.shape[0]}")

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
data_clean['video_comment_count'] =data_clean['video_comment_count'].astype(int)
data_clean['channel_view_count'] = data_clean['channel_view_count'].astype(int)
data_clean['channel_subscriber_count'] = data_clean['channel_subscriber_count'].astype(int)
data_clean['channel_video_count'] = data_clean['channel_video_count'].astype(int)

data_clean.to_csv(r"C:\Users\rahma\OneDrive\Documents\YT\cleaned_data.csv", index=False)
