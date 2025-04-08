import pandas as pd
import re
import nltk
import fasttext
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# ========================
# NLTK 리소스 다운로드
# ========================
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# ========================
# 사용자 정의 불용어 로딩
# ========================
def load_filler_words(filepath="filler_words.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

custom_stopwords = set(stopwords.words("english"))
custom_stopwords.update(load_filler_words())

# ========================
# 형태소 분석기, lemmatizer
# ========================
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# ========================
# Lemmatization 함수
# ========================
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# ========================
# 형태소 분석 + 불용어 제거 + lemmatization 적용
# ========================
def tokenize_and_remove_stopwords(text):
    if pd.isna(text):
        return []
    tokens = tokenizer.tokenize(text)
    filtered = [w for w in tokens if w.lower() not in custom_stopwords]
    lemmatized = lemmatize_tokens(filtered)
    return lemmatized

# ========================
# fastText 모델 로드
# ========================
lang_model = fasttext.load_model("lid.176.bin")

def is_english_fasttext(text):
    try:
        prediction = lang_model.predict(text.replace('\n', ' '), k=1)[0][0]
        return prediction == '__label__en'
    except:
        return False

def parallel_language_filter(df, text_column, n_jobs=6):
    with Pool(n_jobs) as pool:
        results = pool.map(is_english_fasttext, df[text_column])
    return results

def filter_by_year_range(df, start_year=1980, end_year=2024):
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]

# ========================
# 전체 가사 전처리 함수 (라인 제거, 정제, 속어 치환, 긴 단어 제거)
# ========================
def preprocess_lyrics(text):
    if pd.isna(text):
        return ""
    text = remove_repeated_lines(text, keep_repeats=1)
    text = clean_lyrics(text)
    text = normalize_slang(text)
    text = ' '.join([w for w in text.split() if len(w) < 25])
    return text

# ========================
# 공통 유틸 함수들
# ========================
def clean_lyrics(text):
    if pd.isna(text):
        return ""
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r'\[.*?\]', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def normalize_slang(text):
    slang_dict = {
        "gonna": "going to",
        "wanna": "want to",
        "ain't": "is not",
        "lemme": "let me",
        "gotta": "got to",
        "'til": "until",
        "y'all": "you all",
        "imma": "i am going to",
        "kinda": "kind of",
        "outta": "out of",
        "lotta": "lot of",
        "dunno": "do not know",
        "wassup": "what is up",
        "yo": "you",
        "cuz": "because",
        "cause": "because"
    }
    for slang, standard in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', standard, text)
    return text

def remove_repeated_lines(text, keep_repeats=1):
    if pd.isna(text):
        return ""
    lines = text.splitlines()
    seen = {}
    unique_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            seen[stripped] = seen.get(stripped, 0) + 1
            if seen[stripped] <= keep_repeats:
                unique_lines.append(stripped)
    return ' '.join(unique_lines)

# ========================
# Genius Lyrics Dataset 전처리
# ========================
def process_genius_translations(df):
    mask = df['artist'] == 'Genius English Translations'
    df.loc[mask, 'artist'] = df.loc[mask, 'title'].str.split(' - ').str[0]
    df.loc[mask, 'title'] = df.loc[mask, 'title'].str.replace(r'English Translation', '', regex=True)

    for idx in df[mask].index:
        artist_pattern = re.escape(df.loc[idx, 'artist']) + r'\s*\-\s*'
        df.at[idx, 'title'] = re.sub('^' + artist_pattern, '', df.loc[idx, 'title']).strip()

    return df

def preprocess_genius_dataset(df):
    print("1. 컬럼 정리 및 결측치 제거 중...")
    df = df.rename(columns={
        'title': 'title',
        'artist': 'artist',
        'tag': 'genre',
        'lyrics': 'lyrics',
        'year': 'year'
    })
    df = df.dropna(subset=['lyrics', 'year', 'language'])
    df = df.drop(columns=[col for col in df.columns if col not in ['title', 'artist', 'genre', 'views', 'year', 'lyrics', 'language']])
    df = df[['title', 'artist', 'genre', 'year', 'views', 'lyrics', 'language']]

    print("2. 연도 필터링 중...")
    df = filter_by_year_range(df, 1980, 2024)

    print("3. 영어 가사 필터링 중...")
    df = df[df['language'] == 'en']
    df['is_english'] = parallel_language_filter(df, 'lyrics')
    df = df[df['is_english']].drop(columns=['language', 'is_english'])

    print("4. 제목 및 아티스트 정리 중...")
    df = process_genius_translations(df)

    print("5. 가사 정제 중...")
    df['lyrics'] = df['lyrics'].apply(preprocess_lyrics)

    print("6. 형태소 분석 및 불용어 제거 중...")
    df['lyrics_tokens'] = df['lyrics'].apply(tokenize_and_remove_stopwords)

    print("7. 빈 토큰 제거 및 리셋 중...")
    df = df[df['lyrics_tokens'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    print("전처리 완료.")
    return df.reset_index(drop=True)

# ========================
# Top 100 Songs Dataset 전처리
# ========================
def preprocess_top100_dataset(df):
    print("1. 컬럼 정리 및 결측치 제거 중...")
    df = df.rename(columns={
        'Song Title': 'title',
        'Artist': 'artist',
        'Release Date': 'release date',
        'Year': 'year',
        'Rank': 'rank',
        'Lyrics': 'lyrics'
    })
    df = df.dropna(subset=['lyrics', 'year'])
    df = df[[col for col in ['title', 'artist', 'release date', 'year', 'rank', 'lyrics'] if col in df.columns]]

    print("2. 연도 필터링 중...")
    df = filter_by_year_range(df, 1980, 2024)

    print("3. 가사 정제 중...")
    df['lyrics'] = df['lyrics'].apply(preprocess_lyrics)

    print("4. 형태소 분석 및 불용어 제거 중...")
    df['lyrics_tokens'] = df['lyrics'].apply(tokenize_and_remove_stopwords)

    print("5. 빈 토큰 제거 및 리셋 중...")
    df = df[df['lyrics_tokens'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    print("전처리 완료.")
    return df.reset_index(drop=True)