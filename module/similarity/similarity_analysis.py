from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def generate_tfidf_matrix(df, emotion_word_col='emotion_words', save=False, save_path=None):
    """
    감정 단어 리스트를 기반으로 TF-IDF 행렬 생성

    Parameters:
    - df: DataFrame, 감정 분석된 전체 데이터
    - emotion_word_col: str, 감정 단어 리스트가 들어 있는 컬럼명
    - save: bool, TF-IDF 벡터 및 벡터라이저 저장 여부
    - save_path: str or None, TF-IDF 벡터 및 벡터라이저 저장 경로 (선택)

    Returns:
    - tfidf_matrix: sparse matrix, TF-IDF 행렬
    - vectorizer: TfidfVectorizer 객체
    """
    # 리스트 형태를 공백 기준 문자열로 변환
    documents = df[emotion_word_col].apply(lambda x: ' '.join(x.split(', '))).tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 저장 옵션
    if save:
        if save_path is None:
            save_path = '../data'
        with open(f"{save_path}/tfidf_matrix.pkl", "wb") as f:
            pickle.dump(tfidf_matrix, f)
        with open(f"{save_path}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    return tfidf_matrix, vectorizer

def compute_cosine_similarity(tfidf_matrix, save=False, save_path=None):
    """
    TF-IDF 행렬을 기반으로 cosine 유사도 행렬 계산

    Parameters:
    - tfidf_matrix: sparse matrix, TF-IDF 행렬
    - save: bool, 유사도 행렬 저장 여부
    - save_path: str or None, 유사도 행렬 저장 경로 (선택)

    Returns:
    - similarity_matrix: ndarray, cosine 유사도 행렬
    """
    similarity_matrix = cosine_similarity(tfidf_matrix)

    if save:
        if save_path is None:
            save_path = '../data'
        with open(f"{save_path}/similarity_matrix.pkl", "wb") as f:
            pickle.dump(similarity_matrix, f)

    return similarity_matrix