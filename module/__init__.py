# 공통 처리
from .category_utils import load_emotion_data

# 장르별 분석
from .genre_analysis import (
    get_genre_emotion_count,
    get_genre_emotion_ratio,
    get_genre_emotion_score,
    get_genre_top_words,
)

# 연도별 분석
from .decade_analysis import (
    get_decade_emotion_count,
    get_decade_emotion_ratio,
    get_decade_emotion_score,
    get_decade_top_words,
)

# 아티스트별 분석
from .artist_analysis import (
    get_artist_emotion_count,
    get_artist_emotion_ratio,
    get_artist_emotion_score,
    get_artist_top_words,
    get_artist_emotion_trend,
)