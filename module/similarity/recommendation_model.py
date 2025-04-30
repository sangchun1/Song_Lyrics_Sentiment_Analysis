import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_user_emotion_profile(user_songs_df, emotion_columns):
    """
    사용자 감정 성향 벡터 계산

    Parameters:
    - user_songs_df: 사용자가 좋아한 노래들의 데이터프레임
    - emotion_columns: 감정군 컬럼 리스트

    Returns:
    - user_profile: 1D numpy array
    """
    return user_songs_df[emotion_columns].mean().values.reshape(1, -1)

def recommend_songs_by_profile(df, user_profile, emotion_columns, cluster_col='cluster', top_n=10):
    """
    사용자 감정 프로필 기반 추천 시스템

    Parameters:
    - df: 전체 데이터프레임
    - user_profile: 사용자 감정 벡터 (1D array)
    - emotion_columns: 감정군 컬럼 리스트
    - cluster_col: 클러스터 컬럼
    - top_n: 추천 곡 수

    Returns:
    - 추천 결과 DataFrame
    """
    # 전체 클러스터별 평균 감정 프로필 계산
    cluster_profiles = df.groupby(cluster_col)[emotion_columns].mean().values
    cluster_ids = df[cluster_col].unique()

    # 사용자와 각 클러스터 평균의 cosine 유사도 계산
    similarities = cosine_similarity(user_profile, cluster_profiles)[0]
    best_cluster = cluster_ids[np.argmax(similarities)]

    # 선택된 클러스터 내에서 각 노래와의 유사도 측정
    candidate_df = df[df[cluster_col] == best_cluster].copy()
    candidate_vectors = candidate_df[emotion_columns].values
    song_similarities = cosine_similarity(user_profile, candidate_vectors)[0]

    candidate_df['similarity'] = song_similarities
    recommendations = candidate_df.sort_values(by='similarity', ascending=False).head(top_n)

    return recommendations
