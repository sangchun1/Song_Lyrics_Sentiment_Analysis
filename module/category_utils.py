import pandas as pd

def load_emotion_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df['decade'] = (df['year'] // 10) * 10
    df['emotion_words'] = df['emotion_words'].str.split(', ')
    df = df[df['genre'] != 'Unknown']
    return df