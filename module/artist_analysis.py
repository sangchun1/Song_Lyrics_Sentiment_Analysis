import pandas as pd

# 아티스트별 Count
def get_artist_emotion_count(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    top_artists = df['artist'].value_counts().head(top_n).index.tolist()
    df_artist = df[df['artist'].isin(top_artists)]
    return df_artist.groupby('artist')[emotion_count_cols].mean()

# 아티스트별 Ratio
def get_artist_emotion_ratio(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    count = get_artist_emotion_count(df, top_n)
    return count.div(count.sum(axis=1), axis=0)

# 아티스트별 Score
def get_artist_emotion_score(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    top_artists = df['artist'].value_counts().head(top_n).index.tolist()
    df_artist = df[df['artist'].isin(top_artists)]
    return df_artist.groupby('artist')[['emotion_score', 'normalized_emotion_score']].mean()

# 아티스트별 Top-N Words
def get_artist_top_words(df: pd.DataFrame, top_n=10, artist_n=20) -> pd.Series:
    top_artists = df['artist'].value_counts().head(artist_n).index.tolist()
    df_artist = df[df['artist'].isin(top_artists)]
    words = df_artist.explode('emotion_words')
    return words.groupby('artist')['emotion_words'].value_counts().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True)

# 특정 아티스트 감정 변화 추이 (연도별)
def get_artist_emotion_trend(df: pd.DataFrame, artist_name: str, is_year=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    artist_df = df[df['artist'] == artist_name]
    if is_year:
        count = artist_df.groupby('year')[emotion_count_cols].sum()
        score = artist_df.groupby('year')[['emotion_score', 'normalized_emotion_score']].mean()
    else:
        count = artist_df.groupby('decade')[emotion_count_cols].sum()
        score = artist_df.groupby('decade')[['emotion_score', 'normalized_emotion_score']].mean()
    return count, score
