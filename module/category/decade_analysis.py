import pandas as pd

# 시대별 Count
def get_decade_emotion_count(df: pd.DataFrame, is_year=False) -> pd.DataFrame:
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    if is_year:
        return df.groupby('year')[emotion_count_cols].sum()
    return df.groupby('decade')[emotion_count_cols].sum()

# 시대별 Ratio
def get_decade_emotion_ratio(df: pd.DataFrame, is_year=False) -> pd.DataFrame:
    count = get_decade_emotion_count(df)
    if is_year:
        count = get_decade_emotion_count(df, is_year=True)
    return count.div(count.sum(axis=1), axis=0)

# 시대별 Score
def get_decade_emotion_score(df: pd.DataFrame, is_year=False) -> pd.DataFrame:
    if is_year:
        return df.groupby('year')[['emotion_score', 'normalized_emotion_score']].mean()
    return df.groupby('decade')[['emotion_score', 'normalized_emotion_score']].mean()

# 시대별 Top-N Words
def get_decade_top_words(df: pd.DataFrame, top_n=10, is_year=False) -> pd.Series:
    words = df.explode('emotion_words')
    if is_year:
        return words.groupby('year')['emotion_words'].value_counts().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True)
    return words.groupby('decade')['emotion_words'].value_counts().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True)