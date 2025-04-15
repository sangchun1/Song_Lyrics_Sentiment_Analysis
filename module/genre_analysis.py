import pandas as pd

# 장르별 Count
def get_genre_emotion_count(df: pd.DataFrame) -> pd.DataFrame:
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    return df.groupby('genre')[emotion_count_cols].sum()

# 장르별 Ratio
def get_genre_emotion_ratio(df: pd.DataFrame) -> pd.DataFrame:
    count = get_genre_emotion_count(df)
    return count.div(count.sum(axis=1), axis=0)

# 장르별 Score
def get_genre_emotion_score(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('genre')[['emotion_score', 'normalized_emotion_score']].mean()

# 장르별 Top-N Words
def get_genre_top_words(df: pd.DataFrame, top_n=10) -> pd.Series:
    words = df.explode('emotion_words')
    return words.groupby('genre')['emotion_words'].value_counts().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True)