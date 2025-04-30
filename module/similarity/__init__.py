# TF-IDF 및 유사도 계산 함수
from .similarity_analysis import (
    generate_tfidf_matrix,
    compute_cosine_similarity
)

# 클러스터링 함수
from .clustering_analysis import (
    apply_kmeans,
    apply_hierarchical_clustering,
    analyze_and_visualize_clusters,
    analyze_cluster_genre_year_distribution
)

# 추천 엔진 함수
from .recommendation_model import (
    get_user_emotion_profile,
    recommend_songs_by_profile
)

# 외부에 노출할 함수들 정의
__all__ = [
    "generate_tfidf_matrix",
    "compute_cosine_similarity",
    "apply_kmeans",
    "apply_hierarchical_clustering",
    "analyze_and_visualize_clusters",
    "analyze_cluster_genre_year_distribution",
    "get_user_emotion_profile",
    "recommend_songs_by_profile"
]
