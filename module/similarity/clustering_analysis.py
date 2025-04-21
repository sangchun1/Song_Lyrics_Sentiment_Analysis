from sklearn.cluster import KMeans, AgglomerativeClustering
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter

def apply_kmeans(tfidf_matrix, n_clusters=5, save=False, save_path=None):
    """
    TF-IDF 벡터에 대해 KMeans 클러스터링 수행

    Parameters:
    - tfidf_matrix: sparse matrix
    - n_clusters: int, 군집 수
    - save_path: str or None, 클러스터 라벨 저장 경로

    Returns:
    - labels: ndarray, 각 문서의 클러스터 라벨
    - model: KMeans 객체
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(tfidf_matrix)

    if save:
        if save_path is None:
            save_path = '../data'
        with open(f"{save_path}/kmeans_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

    return labels, model

def apply_hierarchical_clustering(tfidf_matrix, n_clusters=5, save=False, save_path=None):
    """
    TF-IDF 벡터에 대해 Agglomerative Clustering 수행

    Parameters:
    - tfidf_matrix: sparse matrix
    - n_clusters: int
    - save_path: str or None

    Returns:
    - labels: ndarray
    - model: AgglomerativeClustering 객체
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = model.fit_predict(tfidf_matrix.toarray())  # toarray 필요

    if save:
        if save_path is None:
            save_path = '../data'
        with open(f"{save_path}/hierarchical_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

    return labels, model

def analyze_and_visualize_clusters(df, labels, emotion_columns, cluster_col='cluster'):
    """
    클러스터별 감정 통계 분석 및 시각화 (막대그래프 + 워드클라우드)

    Parameters:
    - df: DataFrame, 전체 감정 분석 데이터
    - labels: ndarray, 클러스터 라벨
    - emotion_columns: list of str, 감정군 컬럼명 리스트
    - cluster_col: str, 클러스터 번호 저장 컬럼명
    - save_dir: str, 이미지 저장 경로
    """
    # 클러스터 컬럼 추가
    df[cluster_col] = labels

    # 클러스터 개수
    n_clusters = len(set(labels))

    for cluster_id in range(n_clusters):
        cluster_df = df[df[cluster_col] == cluster_id]

        # 1. 감정군 평균 값 barplot
        mean_scores = cluster_df[emotion_columns].mean().sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=mean_scores.index, y=mean_scores.values)
        plt.title(f'Cluster {cluster_id} - Average Emotion Scores')
        plt.ylabel('Mean Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()

        # 2. 클러스터 내 전체 감정 단어 워드클라우드
        all_words = [word for sublist in cluster_df['emotion_words'] for word in sublist]
        word_freq = dict(Counter(all_words))
        wordcloud = WordCloud(width=800, height=400, collocations=False, background_color='white').generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {cluster_id} - Emotion WordCloud')
        plt.tight_layout()
        plt.show()
        plt.close()

def analyze_cluster_genre_year_distribution(df, cluster_col='cluster', genre_col='genre', year_col='year', save_dir='results/plots/cluster_trends'):
    """
    클러스터별 장르 및 연도 분포 시각화

    Parameters:
    - df: DataFrame
    - cluster_col: str, 클러스터 번호가 저장된 컬럼
    - genre_col: str, 장르 컬럼명
    - year_col: str, 연도 컬럼명
    - save_dir: str, 저장 폴더
    """
    os.makedirs(save_dir, exist_ok=True)
    n_clusters = df[cluster_col].nunique()

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]

        # 1. 장르 분포 (countplot)
        plt.figure(figsize=(10, 5))
        sns.countplot(data=cluster_df, y=genre_col, order=cluster_df[genre_col].value_counts().index)
        plt.title(f'Cluster {cluster_id} - Genre Distribution')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cluster_{cluster_id}_genre_distribution.png")
        plt.close()

        # 2. 시대별 분포 (10년 단위로 묶기)
        cluster_df['decade'] = (cluster_df[year_col] // 10) * 10
        decade_dist = cluster_df['decade'].value_counts().sort_index()

        plt.figure(figsize=(8, 4))
        sns.barplot(x=decade_dist.index.astype(int), y=decade_dist.values)
        plt.title(f'Cluster {cluster_id} - Decade Distribution')
        plt.xlabel('Decade')
        plt.ylabel('Number of Songs')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cluster_{cluster_id}_decade_distribution.png")
        plt.close()