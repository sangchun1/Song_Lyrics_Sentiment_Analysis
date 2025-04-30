import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import linregress
from collections import defaultdict
from better_profanity import profanity
import os

#####################################################################################################################################
profanity.load_censor_words() # profanity 라이브러리 불러오기
#####################################################################################################################################
def load_emotion_lexicon(lexicon_path = "../data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 
                         custom_emotion_map = {"joy": "love", "trust": "love", "sadness": "sadness", "fear": "fear",
                                               "disgust":"disgust", "anger": "anger", "anticipation": "hope", "surprise": "hope",}):
    '''
    감정 사전 불러오는 함수

        Parameters:
        - lexicon_path: str, 감정 사전 파일 경로
        - custom_emotion_map: dict, 감정 사전에서 사용할 감정군 매핑

        처리과정:
        1. 감정 사전 파일을 로드
        2. 각 단어에 대해 감정군을 매핑
        3. association이 1인 단어만 필터링
        4. 단어와 감정군을 매핑하여 딕셔너리 형태로 저장
        5. 최종적으로 단어와 감정군 매핑을 반환

        Returns:
        - word_to_emotions: dict, 단어와 감정군 매핑
    '''
    word_to_emotions = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, association = line.strip().split("\t")
            if int(association) != 1:
                continue  # association이 1이 아닌 경우 스킵
            if emotion not in custom_emotion_map:
                continue  # 매핑이 없는 감정은 스킵
            mapped_emotion = custom_emotion_map[emotion]
            word_to_emotions.setdefault(word, set()).add(mapped_emotion)
            
    return word_to_emotions

def censor_word_list(word_list):
    '''
    욕설 단어 리스트를 censor 처리하는 함수

        Parameters:
        - word_list: list, censor 처리할 단어 리스트

        처리과정:
        1. 각 단어에 대해 profanity 라이브러리로 욕설 여부 확인
        2. 욕설인 경우, 단어 길이에 따라 censor 처리
        3. censor 처리된 단어를 리스트에 추가
        4. 최종적으로 censor 처리된 단어 리스트를 반환

        Returns:
        - list, censor 처리된 단어 리스트
    '''
    censored = []
    for word in word_list:
        if profanity.contains_profanity(word):
            if len(word) <= 2:
                # 앞글자 + *들
                censored_word = word[0] + "*" * (len(word) - 1)
            else:
                # 앞+*+뒤 남기기
                censored_word = word[0] + "*" * (len(word) - 2) + word[-1]
        else:
            censored_word = word
        censored.append(censored_word)

    return censored
#####################################################################################################################################
def plot_emotion_bar(df, emotion_list, value_type="count", category=None, category_value=None,
                     color_palette="Set2", title=None, save=False, save_path=None):
    '''
    감정군별 감정 단어 수 또는 비율을 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 단어 수 또는 비율 데이터프레임
        - emotion_list: list, 감정군 리스트
        - value_type: str, "count" 또는 "ratio", 감정 단어 수 또는 비율
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - color_palette: str, seaborn 색상 팔레트
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로

        처리과정:
        1. 카테고리 필터링
        2. 감정 단어 수 또는 비율 계산
        3. 제목 설정
        4. 시각화
        5. 그래프 저장

        Returns:
        - None
    '''
    # 경고
    assert value_type in ["count", "ratio"], "value_type은 'count' 또는 'ratio'만 가능합니다."

    # 데이터프레임 복사 및 카테고리 필터링
    plot_df = df.copy()
    if category and category_value:
        plot_df = plot_df[plot_df[category] == category_value]

    # 값 계산
    if value_type == "count":   # 단어 수라면 .sum()
        values = {emo.title(): plot_df[f"{value_type}_{emo}"].sum() for emo in emotion_list}
    else:   # 비율이라면 .mean()
        values = {emo.title(): plot_df[f"{value_type}_{emo}"].mean() for emo in emotion_list}
    
    # 제목 설정
    if not title:
        base_title = "감정군 감정 단어 수 합계" if value_type == "count" else "감정군 감정 비율 (평균)"
        if category and category_value:
            title = f"{category_value.title()} {base_title}"
        else:
            title = base_title

    # 시각화
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(values.keys()), y=list(values.values()), palette=color_palette)
    value_list = list(values.values())
    for i, v in enumerate(value_list):  # 막대 위에 수치 표시
        plt.text(i, v + (max(value_list) * 0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("단어 수" if value_type == "count" else "평균 비율 (%)")
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category or "total"}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def generate_topN_wordcloud(df, value_type="count", mode="category", category=None, category_value=None,
                            selected_emotion=None, top_n=100, title=None, censor_profanity=False,
                            color_map="Set2", save=False, save_path=None):
    '''
    감정 단어 워드클라우드를 생성하는 함수

        Parameters:
        - df: DataFrame, 감정 단어 데이터프레임
        - value_type: str, "count" 또는 "tfidf", 감정 단어 수 또는 tfidf
        - mode: str, "category" 또는 "grouped", 카테고리 모드 또는 감정군 그룹 모드
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - selected_emotion: str, 선택된 감정군
        - top_n: int, 상위 N개 단어
        - censor_profanity: bool, 욕설 단어 censor 처리 여부
        - title: str, 워드클라우드 제목
        - color_map: str, 워드클라우드 색상 맵
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로

        처리과정:
        1. 카테고리 필터링
        2. 감정군 그룹 모드 또는 전체 모드 선택
        3. 감정 단어 빈도수 계산
        4. 욕설 단어 censor 처리
        5. 워드클라우드 생성
        6. 제목 설정
        7. 시각화
        8. 그래프 저장

        Returns:
        - None
    '''
    plot_df = df.copy() # 데이터프레임 복사

    # 필터링
    if mode == "category" and category and category_value:
        plot_df = plot_df[plot_df[category] == category_value]

    # 감정군 그룹 모드
    if mode == "grouped":
        if selected_emotion is None:
            print("선택된 감정군이 필요합니다.")
            return

        word_to_emotions = load_emotion_lexicon()   # 감정 사전 로드

        if value_type == "count":   # 단어 수 기반 
            word_freq = defaultdict(int)    # 감정 단어 빈도수 저장
            for word_list in plot_df["emotion_words"]:  # 감정 단어 리스트에서 감정군에 해당하는 단어 빈도수 계산
                for word in word_list:
                    if word in word_to_emotions and selected_emotion in word_to_emotions[word]:
                        word_freq[word] += 1
        else:   # tfidf 기반
            selected_words = []
            for word_list in plot_df["emotion_words"]:
                for word in word_list:
                    if word in word_to_emotions and selected_emotion in word_to_emotions[word]:
                        selected_words.append(word)
            
            if selected_words:
                documents = [" ".join(selected_words)]  # 하나의 문서
                vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
                tfidf_matrix = vectorizer.fit_transform(documents)
                tfidf_scores = tfidf_matrix.toarray().flatten()
                feature_names = vectorizer.get_feature_names_out()
                word_freq = dict(zip(feature_names, tfidf_scores))
            else:
                word_freq = {}

        if censor_profanity:    # 욕설 단어 censor 처리
            word_freq = {censor_word_list([word])[0]: freq for word, freq in word_freq.items()}

        text = " ".join([word for word in word_freq for _ in range(word_freq[word])])
        wc = WordCloud(width=800, height=400, collocations=False, background_color="white", 
                       colormap=color_map, regexp=r"\S+").generate(text)

        # 제목 설정
        if title is None:
            title = f"{selected_emotion.title()} 감정군 WordCloud"
            if value_type == "tfidf":
                title += "(TF-IDF)"
            if censor_profanity:
                title += "(비속어 처리)"

    # 전체 모드
    else:
        if value_type == "count":
            word_freq = defaultdict(int)    # 감정 단어 빈도수 저장
            for word_list in plot_df["emotion_words"]:  # 감정 단어 리스트에서 감정군에 해당하는 단어 빈도수 계산
                for word in word_list:
                    word_freq[word] += 1
        else:
            docs = plot_df["emotion_words"].apply(lambda x: ' '.join(x) if isinstance(x, list) else "").tolist()
            vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
            tfidf_matrix = vectorizer.fit_transform(docs)
            tfidf_mean = tfidf_matrix.mean(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            word_freq = dict(zip(feature_names, tfidf_mean))
        
        top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])

        if censor_profanity:    # 욕설 단어 censor 처리
            top_words = {censor_word_list([word])[0]: freq for word, freq in top_words.items()}

        text = " ".join([word for word, freq in top_words.items() for _ in range(freq)])
        wc = WordCloud(width=800, height=400, collocations=False, background_color="white", 
                       colormap=color_map, regexp=r"\S+").generate(text)

        # 제목 설정
        if title is None:
            if category and category_value:
                title = f"{category_value.title()} Top {top_n} 감정 단어 WordCloud"
            else:
                title = f"Top {top_n} 감정 단어 WordCloud"
            if value_type == "tfidf":
                title += "(TF-IDF)"
            if censor_profanity:
                title += "(비속어 처리)"

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category or "total"}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_emotion_correlation_heatmap(df, emotion_list, value_type="count", category=None, category_value=None, 
                                     title=None, save=False, save_path=None):
    '''
    감정군 간 상관관계 히트맵을 시각화하는 함수
        
        Parameters:
        - df: DataFrame, 감정 단어 수 또는 비율 데이터프레임
        - emotion_list: list, 감정군 리스트
        - value_type: str, "count" 또는 "ratio", 감정 단어 수 또는 비율
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 카테고리 필터링
        2. 감정 단어 수 또는 비율 컬럼 리스트 생성
        3. 상관관계 계산
        4. 제목 설정
        5. 시각화
        6. 그래프 저장
            
        Returns:
        - None
    '''
    # 감정 단어 수 또는 비율 컬럼 리스트
    col_prefix = f"{value_type}_"
    target_cols = [f"{col_prefix}{emo}" for emo in emotion_list]

    # 필터링
    plot_df = df.copy()
    if category and category_value:
        plot_df = plot_df[plot_df[category] == category_value]

    # 상관관계 계산
    corr = plot_df[target_cols].corr()
    corr.index = [emo.title() for emo in emotion_list]
    corr.columns = [emo.title() for emo in emotion_list]

    # 제목 설정
    if not title:
        base_title = f"감정군 {value_type} 상관관계 히트맵"
        title = f"{category_value.title()} {base_title}" if category and category_value else base_title

    # 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category or 'total'}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
#####################################################################################################################################
def plot_emotion_score_histogram(df, score_column="emotion_score", category=None, category_value=None,
                                              normalized=False, exclude_outliers=False, show_kde=False, bins=30, 
                                              title=None, save=False, save_path=None):
    '''
    감정 점수 분포 히스토그램을 시각화하는 함수
        
        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - score_column: str, 감정 점수 컬럼 이름
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - normalized: bool, 정규화 여부
        - exclude_outliers: bool, 이상치 제외 여부
        - show_kde: bool, 커널 밀도 추정선 표시 여부
        - bins: int, 히스토그램 bin 개수
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 카테고리 필터링
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 제목 설정
        5. 시각화
        6. 그래프 저장
        
        Returns:
        - None
    '''
    plot_df = df.copy() # 데이터프레임 복사

    # 필터링
    if category and category_value:
        plot_df = plot_df[plot_df[category] == category_value]

    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    
    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"

    # 제목 설정
    if not title:
        base = "감정 점수 분포 히스토그램" if not normalized else "정규화 감정 점수 히스토그램"
        if category and category_value:
            title = f"{category_value.title()} {base}"
        else:
            title = base
        if exclude_outliers:
            title += " (이상치 제외)"

    # 시각화
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df[score_column], bins=bins, kde=show_kde, color="skyblue")
    plt.title(title)
    plt.xlabel("감정 점수" if not normalized else "정규화 감정 점수")
    plt.ylabel("곡 수")
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category or 'total'}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_emotion_ratio_vs_score(df, emotion_name, score_column="emotion_score", normalized=False, exclude_outliers=False, 
                                title=None, save=False, save_path=None):
    '''
    감정군 비율과 감정 점수 상관관계를 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - emotion_name: str, 감정군 이름
        - score_column: str, 감정 점수 컬럼 이름
        - normalized: bool, 정규화 여부
        - exclude_outliers: bool, 이상치 제외 여부
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 데이터프레임 복사
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 제목 설정
        5. 상관계수 계산
        6. 시각화
        7. 그래프 저장
            
        Returns:
        - None
    '''
    # 데이터프레임 복사
    plot_df = df.copy()

    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]

    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"
    
    # 제목 설정
    if not title:
        title = f"{emotion_name.title()} 비율 vs 감정 점수" if not normalized else f"{emotion_name.title()} 비율 vs 정규화 감정 점수"
    if exclude_outliers:
            title += " (이상치 제외)"

    # 상관계수 계산
    x = plot_df[score_column]
    y = plot_df[f"ratio_{emotion_name}"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    
    # 시각화
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.plot([], [], ' ', label=f"$r$ = {r_value:.2f}")
    plt.plot([], [], ' ', label=f"$R^2$ = {r_squared:.2f}")
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel(f"{emotion_name.title()} 비율 (%)")
    plt.legend(loc="best")
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/total/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_top_songs_by_emotion_score(df, score_column="emotion_score", top_n=10, is_bottom=False, 
                                    normalized=False, exclude_outliers=False, save=False, save_path=None,):
    '''
    감정 점수 상위/하위 N곡을 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - score_column: str, 감정 점수 컬럼 이름
        - top_n: int, 상위/하위 N곡 수
        - is_bottom: bool, 하위 N곡 여부
        - normalized: bool, 정규화 여부
        - exclude_outliers: bool, 이상치 제외 여부
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 데이터프레임 복사
        2. title - artist 형식으로 새로운 컬럼 생성
        3. 이상치 제거
        4. 정규화 점수로 교체
        5. 상위/하위 N곡 추출
        6. 시각화
        7. 그래프 저장
            
        Returns:
        - None
    '''
    # 데이터프레임 복사
    plot_df = df.copy()

    # title - artist 형식으로 새로운 컬럼 생성
    plot_df["title_artist"] = plot_df["title"] + " - " + plot_df["artist"]
    
    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    
    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"

    # 상위/하위 N곡 추출
    if is_bottom == False:
        top_songs = plot_df.nlargest(top_n, "emotion_score")
        title = f"감정 점수 상위 {top_n}곡" if not normalized else f"정규화 감정 점수 상위 {top_n}곡"
    else:
        top_songs = plot_df.nsmallest(top_n, "emotion_score")
        title = f"감정 점수 하위 {top_n}곡" if not normalized else f"정규화 감정 점수 하위 {top_n}곡"

    # 시각화
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="emotion_score", y="title_artist", data=top_songs, palette="Greens_d")
    for i, v in enumerate(top_songs[score_column].values):  # 수치 표시 추가
        ax.text(i, v + (max(top_songs[score_column].values) * 0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 제목 - 아티스트")
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/total/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
#####################################################################################################################################
def plot_genre_avg_emotion_score(df, score_column="emotion_score", title="장르별 평균 감정 점수", 
                                 exclude_outliers=False, normalized=False, save=False, save_path=None):
    '''
    장르별 평균 감정 점수를 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - score_column: str, 감정 점수 컬럼 이름
        - title: str, 그래프 제목
        - exclude_outliers: bool, 이상치 제외 여부
        - normalized: bool, 정규화 여부
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 데이터프레임 복사
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 장르별 평균 감정 점수 계산
        5. 시각화
        6. 그래프 저장
        
        Returns:
        - None
    '''
    # 데이터프레임 복사
    plot_df = df.copy()

    # 이상치 제거 
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    
    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    # 장르별 평균 감정 점수 계산
    genre_score = plot_df.groupby("genre")[score_column].mean()

    # 시각화
    plt.figure(figsize=(10, 5))
    sns.barplot(x=genre_score.index, y=genre_score.values, palette="muted")
    plt.title(title)
    plt.xlabel("장르")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/category/genre/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_genre_emotion_score_heatmap(df, score_column="emotion_score", title=None,
                                     exclude_outliers=False, normalized=False, save=False, save_path=None):
    '''
    장르별 감정 점수 분포 상관관계 히트맵을 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - score_column: str, 감정 점수 컬럼 이름
        - title: str, 그래프 제목
        - exclude_outliers: bool, 이상치 제외 여부
        - normalized: bool, 정규화 여부
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로

        처리과정:
        1. 데이터프레임 복사
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 장르별 상관관계 계산
        5. 제목 설정
        6. 시각화
        7. 그래프 저장

        Returns:
        - None
    '''
    
    # 데이터프레임 복사
    plot_df = df.copy()

    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    
    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"
    
    # 장르별 상관관계 계산
    pivot_df = plot_df.pivot_table(index="title", columns="genre", values=score_column) # 장르별 감정 점수 피벗 테이블 생성
    corr = pivot_df.corr()  # 상관관계 계산

    # 제목 설정
    if not title:
        title = "장르별 감정 점수 분포 상관관계 히트맵" if not normalized else "장르별 정규화 감정 점수 분포 상관관계 히트맵"
        if exclude_outliers:
            title += "(이상치 제외)"

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/category/genre/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_emotion_ratio_stacked_bar(df, emotion_list, category, title=None, save=False, save_path=None):
    '''
    감정군 비율 스택 바차트를 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 단어 수 또는 비율 데이터프레임
        - emotion_list: list, 감정군 리스트
        - category: str, 카테고리 이름
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 카테고리 필터링
        2. 감정 단어 수 또는 비율 컬럼 리스트 생성
        3. 그룹핑하여 평균 계산
        4. 제목 설정
        5. 시각화
        6. 그래프 저장
            
        Returns:
        - None
    '''
    # 경고
    assert category in df.columns, f"{category} 컬럼이 데이터에 존재하지 않습니다."

    # 감정군 비율 컬럼 리스트
    col_names = [f"ratio_{emo}" for emo in emotion_list]

    # 그룹핑
    grouped = df.groupby(category)[col_names].mean()
    grouped.columns = [emo.title() for emo in emotion_list]

    # 제목 설정
    if not title:
        title = f"{category.title()}별 감정군 비율 스택 바차트"

    # 시각화
    bottom = None
    plt.figure(figsize=(12, max(6, 0.4 * len(grouped))))
    for emo in grouped.columns:
        plt.bar(grouped.index, grouped[emo], bottom=bottom, label=emo)
        bottom = grouped[emo] if bottom is None else bottom + grouped[emo]
    plt.title(title)
    plt.xlabel(category.title())
    plt.ylabel("비율 (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
#####################################################################################################################################
def plot_emotion_trend(df, emotion_list, category="year", category_value=None, value_type="count", title=None, save=False, save_path=None):
    '''
    감정군 비율 변화 추이를 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 단어 수 또는 비율 데이터프레임
        - emotion_list: list, 감정군 리스트
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - value_type: str, "ratio" 또는 "count", 감정 단어 수 또는 비율
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 카테고리 필터링
        2. 감정 단어 수 또는 비율 컬럼 리스트 생성
        3. 그룹핑하여 평균 계산
        4. 제목 설정
        5. 시각화
        6. 그래프 저장
            
        Returns:
        - None
    '''
    # 경고
    assert value_type in ["count", "ratio"], "value_type는 'count' 또는 'ratio'만 가능합니다."

    # 감정 단어 수 또는 비율 컬럼 리스트
    col_prefix = f"{value_type}_"
    target_cols = [f"{col_prefix}{emo}" for emo in emotion_list]

    plot_df = df.copy()
    
    if category_value:
        # 특정 항목에 대해 시간 흐름에 따른 감정군 변화 추이
        plot_df = plot_df[plot_df[category] == category_value]
        time_col = "year"
        grouped = plot_df.groupby(time_col)[target_cols].mean()
        title = f"{category_value.title()} 감정군 {value_type} 변화 추이"

    else:
        # 전체 카테고리별 평균 감정군 변화 추이
        grouped = plot_df.groupby(category)[target_cols].mean()
        title = f"{category.capitalize()}별 감정군 {value_type} 변화 추이"

    grouped.columns = [emo.capitalize() for emo in emotion_list]

    # 시각화
    plt.figure(figsize=(10, 6))
    for emo in grouped.columns:
        plt.plot(grouped.index, grouped[emo], label=emo)
    plt.title(title)
    plt.xlabel(category.title())
    plt.ylabel("비율 (%)" if value_type == "ratio" else "단어 수")
    plt.legend()
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_emotion_score_trend(df, score_column="emotion_score", category="year", category_value=None, 
                             normalized=False, exclude_outliers=False, title=None, save=False, save_path=None):
    '''
    감정 점수 평균 추이를 시각화하는 함수
    
        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - score_column: str, 감정 점수 컬럼 이름
        - category: str, 카테고리 이름
        - category_value: str, 카테고리 값
        - normalized: bool, 정규화 여부
        - exclude_outliers: bool, 이상치 제외 여부
        - title: str, 그래프 제목
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 카테고리 필터링
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 제목 설정
        5. 시각화
        6. 그래프 저장
            
        Returns:
        - None
    '''
    # 데이터프레임 복사
    plot_df = df.copy()

    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    
    # 정규화 사용 시 컬럼 변경
    if normalized:
        score_column = "normalized_emotion_score"

    # 분기: 특정 대상인지 전체 그룹인지
    if category_value:
        plot_df = plot_df[plot_df[category] == category_value]
        time_col = "year" if "year" in df.columns else "decade"
        grouped = plot_df.groupby(time_col)[score_column].mean()
        plot_title = f"{category_value.title()} 감정 점수 추이" if not normalized else f"{category_value.title()} 정규화 감정 점수 추이"
    else:
        grouped = plot_df.groupby(category)[score_column].mean()
        plot_title = f"{category.capitalize()}별 감정 점수 평균 추이" if not normalized else f"{category.capitalize()}별 정규화 감정 점수 평균 추이"

    # 제목 설정
    if not title:
        title = plot_title
        if exclude_outliers:
            title += " (이상치 제외)"

    # 시각화
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=grouped.index, y=grouped.values, marker="o")
    plt.title(title)
    plt.xlabel(category.title())
    plt.ylabel("평균 감정 점수" if not normalized else "정규화 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/{category}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
#####################################################################################################################################
def plot_topN_artists(df, n=10, score_column="emotion_score", title=None, exclude_outliers=False, normalized=False, 
                      save=False, save_path=None):
    '''
    감정 점수 상위 N명의 아티스트를 시각화하는 함수

        Parameters:
        - df: DataFrame, 감정 점수 데이터프레임
        - n: int, 상위 N명 아티스트 수
        - score_column: str, 감정 점수 컬럼 이름
        - title: str, 그래프 제목
        - exclude_outliers: bool, 이상치 제외 여부
        - normalized: bool, 정규화 여부
        - save: bool, 그래프 저장 여부
        - save_path: str, 그래프 저장 경로
        
        처리과정:
        1. 데이터프레임 복사
        2. 이상치 제거
        3. 정규화 점수로 교체
        4. 아티스트별 평균 감정 점수 계산
        5. 제목 설정
        6. 시각화
        7. 그래프 저장
            
        Returns:
        - None
    '''
    # 데이터프레임 복사
    plot_df = df.copy()

    # 이상치 제거
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    
    # 정규화 점수로 교체
    if normalized:
        score_column = "normalized_emotion_score"
    
    # 아티스트별 평균 감정 점수 계산
    top_artists = plot_df.groupby("artist")[score_column].mean().nlargest(n)

    # 제목 설정
    if not title:
        title = f"상위 {n}명 아티스트 평균 감정 점수" if not normalized else f"상위 {n}명 아티스트 정규화 감정 점수"
        if exclude_outliers:
            title += " (이상치 제외)"

    # 시각화
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=top_artists.index, y=top_artists.values, palette="Set2")
    for i, v in enumerate(top_artists.values):  # 수치 표시 추가
        ax.text(i, v + (max(top_artists.values) * 0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.title(title)
    plt.xlabel("아티스트")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 저장
    if save:
        if save_path is None:
            filename = f"{title}.png"
            save_path = f"../results/plots/category/artist/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
