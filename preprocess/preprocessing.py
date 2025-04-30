import pandas as pd
import re
import fasttext
from multiprocessing import Pool
import spacy

# ========================
# 필요 리소스 로드
# ========================
lang_model = fasttext.load_model("../data/lid.176.bin") # FastText 모델
nlp = spacy.load("en_core_web_sm") # SpaCy tokenizer
filler_words_path = '../data/filler_words.txt'
def load_filler_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    return set(word.strip().lower() for word in words if word.strip())
custom_filler_words = load_filler_words(filler_words_path)
stop_words = nlp.Defaults.stop_words.union(custom_filler_words)

def filter_by_year_range(df, start_year=1980, end_year=2024):
    '''
    연도 필터링 함수

        Parameters:
        - df: DataFrame, 원본 데이터
        - start_year: int, 시작 연도
        - end_year: int, 종료 연도

        처리과정:
        1. 데이터 프레임에서 시작 연도와 종료 연도 사이에 있는 값들만 필터링

        Return:
        - DataFrame, 필터링된 데이터 프레임
    '''
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64') # 'year' 컬럼을 int로 변환
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]    # 연도 범위 필터링한 데이터프레임 반환

def is_english_fasttext(text):
    '''
    FastText 모델을 사용하여 텍스트가 영어인지 확인하는 함수

        Parameters:
        - text: str, 텍스트

        처리과정:
        1. FastText 모델을 사용하여 텍스트의 언어를 예측
        2. 예측된 언어가 영어(__label__en)인지 확인
        3. 예측된 언어가 영어이면 True, 아니면 False 반환

        Return:
        - bool, 영어 여부
    '''
    try:
        prediction = lang_model.predict(text.replace('\n', ' '), k=1)[0][0]
        return prediction == '__label__en'
    except:
        return False

def parallel_language_filter(df, text_column, n_jobs=6):
    '''
    FastText 모델을 사용하여 영어만 필터링하는 함수

        Parameters:
        - df: DataFrame, 원본 데이터
        - text_column: str, 텍스트 컬럼 이름
        - n_jobs: int, 병렬 처리할 프로세스 수

        처리과정:
        1. multiprocessing.Pool을 사용하여 병렬 처리
        2. 각 텍스트에 대해 is_english_fasttext 함수를 호출하여 영어 여부 확인
        3. 결과를 리스트로 반환

        Return:
        - list, 영어 여부 리스트
    '''
    with Pool(n_jobs) as pool:
        results = pool.map(is_english_fasttext, df[text_column])
    return results

def process_genius_translations(df):
    '''
    Genius 번역 노래 제목 및 아티스트 처리 함수

        Parameters:
        - df: DataFrame, Genius 데이터프레임

        처리 과정:
        1. 'Genius English Translations' 아티스트 필터링
        2. 아티스트 이름을 제목에서 추출
        3. 제목에서 'English Translation' 제거
        4. 제목에서 아티스트 이름 제거
        6. 아티스트 이름과 제목의 중복 제거

        Return:
        - DataFrame, 처리된 데이터프레임
    '''
    mask = df['artist'] == 'Genius English Translations'    # 'Genius English Translations' 아티스트 필터링
    df.loc[mask, 'artist'] = df.loc[mask, 'title'].str.split(' - ').str[0]  # 아티스트 이름을 제목에서 추출
    df.loc[mask, 'title'] = df.loc[mask, 'title'].str.replace(r'English Translation', '', regex=True) # 제목에서 'English Translation' 제거

    for idx in df[mask].index:
        artist_pattern = re.escape(df.loc[idx, 'artist']) + r'\s*\-\s*' # 
        df.at[idx, 'title'] = re.sub('^' + artist_pattern, '', df.loc[idx, 'title']).strip()    #

    return df

def expand_multi_artist_rows(df):
    '''
    여러 아티스트가 있는 행을 분리하는 함수

        Parameters:
        - df: DataFrame, 원본 데이터

        처리과정:
        1. 'artist' 컬럼에서 여러 아티스트가 있는 행을 분리
        2. 각 아티스트에 대해 새로운 행 생성
        3. 원본 데이터프레임에서 여러 아티스트가 있는 행을 제거
        4. 새로운 행을 원본 데이터프레임에 추가
        5. 최종 데이터프레임 반환

        Return:
        - DataFrame, 분리된 데이터프레임
    '''
    expanded_rows = []

    for _, row in df.iterrows():
        artists = re.split(r'\s*(?:&|,|feat\.|Feat\.|FEAT\.|featuring|Featuring| X | x )\s*', row["artist"])
        artists = [a.strip() for a in artists if a.strip()]

        if len(artists) > 1:
            for artist in artists:
                new_row = row.copy()
                new_row["artist"] = artist
                expanded_rows.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    df_cleaned = df[~df['artist'].str.contains(" & ", na=False)]
    final_df = pd.concat([df_cleaned, expanded_df], ignore_index=True)

    return final_df

def preprocess_lyrics(text, remove_repeat=True):
    '''
    전체 가사 전처리 함수
        Parameters:
        - text: str, 가사

        처리 과정:
        1. 중복된 라인 제거(반복되는 후렴구 등)
        2. 가사 정제(불필요한 문자 제거)
        3. 줄임말 및 속어 치환
        4. 길이가 긴 단어 제거

        Return:
        - str, 전처리된 가사
    '''
    if pd.isna(text):
        return ""
    
    # 중복된 라인 제거
    if remove_repeat:
        lines = text.splitlines()
        seen = {}
        unique_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                seen[stripped] = seen.get(stripped, 0) + 1
                if seen[stripped] <= 1:
                    unique_lines.append(stripped)
        text = ' '.join(unique_lines)

    # 가사 정제(불필요한 문자 제거)
    text = text.strip().lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"(\w+)in'", r"\1ing", text)

    # 특수문자 제거
    text = re.sub(r'\[.*?\]', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    text = re.sub(r'\s+', ' ', text)

    # 속어 치환
    slang_contractions = {
        "gonna": "going to",
        "wanna": "want to",
        "lemme": "let me",
        "gotta": "got to",
        "y'all": "you all",
        "imma": "i am going to",
        "kinda": "kind of",
        "outta": "out of",
        "lotta": "lot of",
        "dunno": "do not know",
        "wassup": "what is up",
        "yo": "you",
        "cuz": "because",
        "cause": "because",
        "gimme": "give me",
        "bruh": "brother",
        "ya": "you",
        "nah": "no",
        "bet": "okay",
        "finna": "going to",
        "tryna": "trying to",
        "sorta": "sort of",
        "'em": "them",
        "'cause": "because",
        "'til": "until",
        "'round": "around",
        "'bout": "about",
        "ain't": "is not",
        "would've": "would have",
        "could've": "could have",
        "should've": "should have",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "can't": "cannot",
        "won't": "will not",
        "goin": "going",
        "nothin": "nothing",
        "lovin": "loving",
        "lookin": "looking",
        "feelin": "feeling",
        "gettin": "getting",
        "somethin": "something",
        "doin": "doing",
        "comin": "coming",
        "tryin": "trying",
        "talkin": "talking",
        "livin": "living",
        "thinkin": "thinking",
        "makin": "making",
        "runnin": "running",
        "fuckin": "fucking",
        "sayin": "saying",
        "dancin": "dancing",
        "flowin": "flowing"
    }
    for short, full in {**slang_contractions}.items():
        text = re.sub(r"\b" + re.escape(short) + r"\b", full, text)
    
    # 반복 문자 줄이기
    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    # 길이가 긴 단어 제거
    text = ' '.join([w for w in text.split() if len(w) < 25])

    return text

def tokenize_lemmatize(texts, n_process=-1):
    '''
    텍스트를 spaCy로 토큰화하고, 불용어 제거 및 표제어화(lemmatization)를 적용하는 함수

        Parameters:
        - text: str, 텍스트

        처리과정:
        1. spaCy로 토큰화
        2. 불용어 제거
        3. spaCy의 lemmatization을 사용하여 표준화

        Return:
        - list, 토큰화된 단어 리스트
    '''

    results = []

    for doc in nlp.pipe(texts, batch_size=50, n_process=n_process):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.text.lower() not in stop_words
        ]
        results.append(tokens)
        
    return results
