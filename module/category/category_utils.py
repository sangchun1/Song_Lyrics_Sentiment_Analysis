import pandas as pd

def load_emotion_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df[df['genre'] != 'Unknown']
    return df