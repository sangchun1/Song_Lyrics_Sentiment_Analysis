import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import defaultdict
import os

# 감정 사전 불러오는 함수
def load_emotion_lexicon(lexicon_path = "../data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", custom_emotion_map = {
                                                                                            "joy": "love", "trust": "love", "positive": "love", 
                                                                                            "sadness": "sadness", "fear": "sadness", "negative": "sadness",
                                                                                            "anger": "anger", "disgust": "anger",
                                                                                            "anticipation": "hope", "surprise": "surprise",}):
    word_to_emotions = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, association = line.strip().split("\t")
            if int(association) == 1:
                mapped_emotion = custom_emotion_map.get(emotion)
                if mapped_emotion:
                    word_to_emotions.setdefault(word, set()).add(mapped_emotion)
    return word_to_emotions

#####################################################################################################################################
# 감정군별 감정 단어 수 바차트
def plot_emotion_count_bar(df, emotion_list, title="감정군별 감정 단어 수 분포", save=False, save_path=None):
    totals = {emo.capitalize(): df[f"count_{emo}"].sum() for emo in emotion_list}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(totals.keys()), y=list(totals.values()), palette="Blues_d")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("총 감정 단어 수")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 감정군별 감정 단어 비율 바차트
def plot_emotion_ratio_bar(df, emotion_list, title="감정군별 감정 단어 비율 (평균)", save=False, save_path=None):
    ratios = {emo.capitalize(): df[f"ratio_{emo}"].mean() for emo in emotion_list}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(ratios.keys()), y=list(ratios.values()), palette="Oranges_d")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("평균 비율 (%)")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 전체 감정 단어 워드클라우드
def generate_overall_wordcloud(df, title="전체 감정 단어 WordCloud", save=False, save_path=None):
    plot_df = df.copy()
    plot_df["emotion_words"] = plot_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    all_words = [word for sublist in plot_df["emotion_words"].dropna() for word in sublist]
    all_emotion_words = " ".join(all_words)
    wc = WordCloud(width=800, height=400, collocations=False, background_color="white", colormap="viridis").generate(all_emotion_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 감정군별 감정 단어 워드클라우드
def generate_grouped_wordclouds(df, selected_emotion, title="감정군 감정 단어 WordCloud", colormap=None, save=False, save_path=None):
    plot_df = df.copy()
    plot_df["emotion_words"] = plot_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    word_to_emotions = load_emotion_lexicon()
    emotion_word_dict = defaultdict(list)
    for word_list in plot_df["emotion_words"]:
        for word in word_list:
            if word in word_to_emotions:
                for emo in word_to_emotions[word]:
                    emotion_word_dict[emo].append(word)
    text = " ".join(emotion_word_dict[selected_emotion])
    wc = WordCloud(width=800, height=400, collocations=False, colormap=colormap, background_color="white").generate(text)
    plt.figure(figsize=(7, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{selected_emotion.capitalize()} {title}")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = f"{selected_emotion.capitalize()} {title}"
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 특정 곡의 감정 단어 수 시각화
def plot_emotion_count_per_song(df, target_title=None, top_n=5, save=False, save_path=None):
    if target_title:
        row = df[df['title'] == target_title].iloc[0]
        word_columns = [col for col in df.columns if col.startswith("count_")]
        word_freq = {col.replace("count_", "").capitalize(): row[col] for col in word_columns if row[col] > 0}
        top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])
        if top_words:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette="Set2")
            plt.title(f"{row['title']} 감정 단어 수")
            plt.ylabel("빈도")
            plt.tight_layout()       
            if save:
                if save_path is None:
                    filename = f"{row['title']} 감정 단어 수"
                    save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300)
            plt.show()
    else:
        for idx, row in df.iterrows():
            word_columns = [col for col in df.columns if col.startswith("count_")]
            word_freq = {col.replace("count_", "").capitalize(): row[col] for col in word_columns if row[col] > 0}
            top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])
            if top_words:
                plt.figure(figsize=(6, 4))
                sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette="Set2")
                plt.title(f"{row['title']} 감정 단어 수")
                plt.ylabel("빈도")
                plt.tight_layout()
                plt.show()

# 감정 단어 수 상관관계 히트맵
def plot_emotion_word_correlation(df, title="감정 단어 수 상관관계 히트맵", save=False, save_path=None):
    word_columns = [col for col in df.columns if col.startswith("count_")]
    corr = df[word_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

#####################################################################################################################################
# 감정 점수 분포 히스토그램
def plot_emotion_score_histogram(df, score_column="emotion_score", bins=30, title="감정 점수 분포 히스토그램", 
                                 exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df[score_column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 수")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 감정군별 평균 감정 점수 바차트
def plot_avg_emotion_score(df, selected_emotion, title="감정군별 평균 감정 점수", save=False, save_path=None):
    avg_scores_by_emotion = {}
    for emo in selected_emotion:
        mask = df[f"count_{emo}"] > 0  # 해당 감정 단어가 존재하는 곡만
        avg_score = df[mask]["emotion_score"].mean()
        avg_scores_by_emotion[emo] = round(avg_score, 2)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(avg_scores_by_emotion.keys()), y=list(avg_scores_by_emotion.values()), palette="viridis")
    plt.title(title)
    plt.ylabel("평균 감정 점수")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 감정군별 비율과 감정 점수 상관관계 시각화
def plot_emotion_ratio_vs_score(df, emotion_name, score_column="emotion_score", exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    title = f"{emotion_name.capitalize()} 비율 vs 감정 점수"
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제거)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    x = plot_df[score_column]
    y = plot_df[f"ratio_{emotion_name}"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.plot([], [], ' ', label=f"$r$ = {r_value:.2f}")
    plt.plot([], [], ' ', label=f"$R^2$ = {r_squared:.2f}")
    plt.title(title, fontsize=13)
    plt.xlabel("감정 점수")
    plt.ylabel(f"{emotion_name.capitalize()} 비율 (%)")
    plt.legend(loc="best")
    plt.tight_layout()
    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/감정 사전 기반 분석/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# 감정점수 상위/하위 N곡 시각화
def show_top_songs_by_emotion_score(df, score_column="emotion_score", top_n=5, is_bottom=False, normalized=False, exclude_outliers=False):
    from IPython.display import display
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    if normalized:
        score_column = "normalized_emotion_score"
    top_songs = plot_df.sort_values(score_column, ascending=is_bottom).head(top_n)
    display(top_songs[['title', 'artist', score_column, "emotion_score_detail"]])

#####################################################################################################################################
# 장르별 감정군 카운트 시각화
def plot_genre_emotion_count(df, emotion_list, title="장르별 감정군 단어 수 합계", save=False, save_path=None):
    genre_emotion_count = df.groupby("genre")[[f"count_{emo}" for emo in emotion_list]].sum()

    genre_emotion_count.plot(kind="bar", figsize=(12, 6), colormap="Set2")
    plt.title(title)
    plt.xlabel("장르")
    plt.ylabel("감정 단어 수 합계")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정군 비율 시각화
def plot_genre_emotion_ratio(df, emotion_list, title="장르별 감정군 비율 (평균)", save=False, save_path=None):
    genre_emotion_ratio = df.groupby("genre")[[f"ratio_{emo}" for emo in emotion_list]].mean()

    genre_emotion_ratio.plot(kind="bar", figsize=(12, 6), colormap="pastel")
    plt.title(title)
    plt.xlabel("장르")
    plt.ylabel("감정군 비율(%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 평균 감정 점수 시각화
def plot_genre_avg_emotion_score(df, score_column="emotion_score", title="장르별 평균 감정 점수", 
                                 exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    genre_score = plot_df.groupby("genre")[score_column].mean()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=genre_score.index, y=genre_score.values, palette="muted")
    plt.title(title)
    plt.xlabel("장르")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 Top-N 감정 단어 워드클라우드
def generate_genre_topN_wordcloud(df, selected_genre, top_n=50, save=False, save_path=None, color_map="tab10"):
    genre_df = df[df["genre"] == selected_genre].copy()
    genre_df["emotion_words"] = genre_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    word_freq = {}

    for words in genre_df["emotion_words"]:
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])

    wc = WordCloud(width=800, height=400, collocations=False, background_color="white", colormap=color_map).generate_from_frequencies(top_words)
    
    title = f"{selected_genre.capitalize()} 장르 Top {top_n} 감정 단어 WordCloud"
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정 점수 분포 시각화
def plot_genre_emotion_score_distribution(df, selected_genre, score_column="emotion_score", title="감정 점수 분포", 
                                          exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    genre_groups = plot_df.groupby("genre")
    genre_groups = genre_groups.get_group(selected_genre)
    title = f"{selected_genre.capitalize()} {title}"
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(genre_groups[score_column], label=selected_genre, fill=True, alpha=0.5)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 수")
    plt.legend(title="장르")
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정 점수 분포 히스토그램
def plot_genre_emotion_score_histogram(df, selected_genre, score_column="emotion_score", bins=30, title="감정 점수 분포 히스토그램",
                                       exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"  

    genre_groups = plot_df.groupby("genre")
    genre_groups = genre_groups.get_group(selected_genre)
    title = f"{selected_genre.capitalize()} {title}"      

    plt.figure(figsize=(8, 5))
    sns.histplot(genre_groups[score_column], bins=bins, kde=True, label=selected_genre, alpha=0.5)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 수")
    plt.legend(title="장르")
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정 점수 분포 상관관계 히트맵
def plot_genre_emotion_score_heatmap(df, score_column="emotion_score", title="장르별 감정 점수 분포 상관관계 히트맵",
                                     exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    genre_groups = plot_df.groupby("genre")
    genre_emotion_scores = pd.DataFrame()

    for genre, group in genre_groups:
        genre_emotion_scores[genre] = group[score_column]

    corr = genre_emotion_scores.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정군 간 상관관계 히트맵
def plot_genre_emotion_correlation_heatmap(df, emotion_list, selected_genre, title="장르별 감정군 간 상관관계 히트맵", save=False, save_path=None):
    
    genre_emotion_count = df.groupby("genre")[[f"count_{emo}" for emo in emotion_list]].sum()
    genre_emotion_count = genre_emotion_count.get_group(selected_genre)
    corr = genre_emotion_count.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 장르별 감정군 비율 스택 바차트
def plot_genre_emotion_ratio_stacked_bar(df, title="장르별 감정군 비율 스택 바차트", save=False, save_path=None):
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    genre_emotion_count = df.groupby('genre')[emotion_count_cols].sum()
    genre_emotion_ratio = genre_emotion_count.div(genre_emotion_count.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    genre_emotion_ratio.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('장르')
    plt.ylabel('감정 비율')
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/장르/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

#####################################################################################################################################
# 시대별 감정군 카운트 바차트
def plot_decade_emotion_count(df, emotion_list, title="시대별 감정군 단어 수 합계", save=False, save_path=None):
    decade_emotion_count = df.groupby("decade")[[f"count_{emo}" for emo in emotion_list]].sum()
    
    plt.figure(figsize=(12, 6))
    decade_emotion_count.plot(kind="bar", colormap="Set2")
    plt.title(title)
    plt.xlabel("시대")
    plt.ylabel("감정 단어 수 합계")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
    
# 시대별 감정군 비율 바차트
def plot_decade_emotion_ratio(df, emotion_list, title="시대별 감정군 비율 (평균)", save=False, save_path=None):
    decade_emotion_ratio = df.groupby("decade")[[f"ratio_{emo}" for emo in emotion_list]].mean()
    
    plt.figure(figsize=(12, 6))
    decade_emotion_ratio.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel("시대")
    plt.ylabel("감정 비율")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 연도별 감정 점수 편화 추이(평균)
def plot_year_emotion_score_trend(df, score_column="emotion_score", title="연도별 감정 점수 편화 추이", 
                                    exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    year_groups = plot_df.groupby("year")
    year_avg_scores = year_groups[score_column].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=year_avg_scores.index, y=year_avg_scores.values, marker="o")
    plt.title(title)
    plt.xlabel("시대")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 시대별 top-N 단어 워드클라우드
def generate_decade_topN_wordcloud(df, selected_decade, top_n=50, save=False, save_path=None, color_map="tab10"):
    decade_df = df[df["decade"] == selected_decade].copy()
    decade_df["emotion_words"] = decade_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    word_freq = {}

    for words in decade_df["emotion_words"]:
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])

    wc = WordCloud(width=800, height=400, collocations=False, background_color="white", colormap=color_map).generate_from_frequencies(top_words)
    
    title = f"{selected_decade} 시대 Top {top_n} 감정 단어 WordCloud"
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 시대별 감정 점수 히스토그램
def plot_decade_emotion_score_histogram(df, selected_decade, score_column="emotion_score", bins=30, title="감정 점수 분포 히스토그램",
                                       exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"  

    decade_groups = plot_df.groupby("decade")
    decade_groups = decade_groups.get_group(selected_decade)
    title = f"{selected_decade} {title}"      

    plt.figure(figsize=(8, 5))
    sns.histplot(decade_groups[score_column], bins=bins, kde=True, label=selected_decade, alpha=0.5)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 수")
    plt.legend(title="시대")
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 연도별 감정군 트렌드 라인
def plot_year_emotion_trend(df, emotion_list, title="연도별 감정군 트렌드", save=False, save_path=None):
    year_emotion_count = df.groupby("year")[[f"count_{emo}" for emo in emotion_list]].sum()
    
    plt.figure(figsize=(12, 6))
    year_emotion_count.plot(kind="line", colormap="Set2")
    plt.title(title)
    plt.xlabel("연도")
    plt.ylabel("감정 단어 수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 연도별 감정군 점유율 변화 그래프
def plot_year_emotion_ratio_change(df, emotion_list, title="연도별 감정군 점유율 변화", save=False, save_path=None):
    year_emotion_ratio = df.groupby("year")[[f"ratio_{emo}" for emo in emotion_list]].mean()
    
    plt.figure(figsize=(12, 6))
    year_emotion_ratio.plot(kind="line", colormap="Set2")
    plt.title(title)
    plt.xlabel("연도")
    plt.ylabel("감정 비율")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 시대별 감정군 비율 스택 바차트
def plot_decade_emotion_ratio_stacked_bar(df, title="시대별 감정군 비율 스택 바차트", save=False, save_path=None):
    emotion_count_cols = [col for col in df.columns if col.startswith('count_')]
    decade_emotion_count = df.groupby('decade')[emotion_count_cols].sum()
    decade_emotion_ratio = decade_emotion_count.div(decade_emotion_count.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    decade_emotion_ratio.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('시대')
    plt.ylabel('감정 비율')
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/시대/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

#####################################################################################################################################
# 특정 아티스트 감정군 카운트
def plot_artist_emotion_count(df, artist_name, emotion_list, title="특정 아티스트 감정군 카운트", save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name]
    artist_emotion_count = artist_df[[f"count_{emo}" for emo in emotion_list]].sum()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=artist_emotion_count.index, y=artist_emotion_count.values, palette="Set2")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("감정 단어 수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 감정군 비율
def plot_artist_emotion_ratio(df, artist_name, emotion_list, title="특정 아티스트 감정군 비율", save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name]
    artist_emotion_ratio = artist_df[[f"ratio_{emo}" for emo in emotion_list]].mean()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=artist_emotion_ratio.index, y=artist_emotion_ratio.values, palette="Set2")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("감정 비율")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 평균 감정 점수
def plot_artist_avg_emotion_score(df, artist_name, score_column="emotion_score", title="특정 아티스트 평균 감정 점수", 
                                  exclude_outliers=False, normalized=False, save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name]
    plot_df = artist_df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    avg_score = plot_df[score_column].mean()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=[artist_name], y=[avg_score], palette="Set2")
    plt.title(title)
    plt.xlabel("아티스트")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 Top-N 감정 단어 워드 클라우드
def generate_artist_topN_wordcloud(df, artist_name, top_n=50, save=False, save_path=None, color_map="tab10"):
    artist_df = df[df["artist"] == artist_name].copy()
    artist_df["emotion_words"] = artist_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    word_freq = {}

    for words in artist_df["emotion_words"]:
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])

    wc = WordCloud(width=800, height=400, collocations=False, background_color="white", colormap=color_map).generate_from_frequencies(top_words)
    
    title = f"{artist_name} 아티스트 Top {top_n} 감정 단어 WordCloud"
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 감정 분포 변화 히스토그램
def plot_artist_emotion_score_histogram(df, artist_name, score_column="emotion_score", bins=30, title="감정 점수 분포 히스토그램",
                                       exclude_outliers=False, normalized=False, save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name].copy()
    plot_df = artist_df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"  

    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df[score_column], bins=bins, kde=True, label=artist_name, alpha=0.5)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("곡 수")
    plt.legend(title="아티스트")
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 감정 변화 추이
def plot_artist_emotion_trend(df, artist_name, emotion_list, title="특정 아티스트 감정 변화 추이", save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name]
    artist_emotion_count = artist_df.groupby("year")[[f"count_{emo}" for emo in emotion_list]].sum()
    
    plt.figure(figsize=(12, 6))
    artist_emotion_count.plot(kind="line", colormap="Set2")
    plt.title(title)
    plt.xlabel("연도")
    plt.ylabel("감정 단어 수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 특정 아티스트 감정 점수 트렌드 라인
def plot_artist_emotion_score_trend(df, artist_name, score_column="emotion_score", title="특정 아티스트 감정 점수 트렌드", 
                                     exclude_outliers=False, normalized=False, save=False, save_path=None):
    artist_df = df[df["artist"] == artist_name]
    plot_df = artist_df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    year_groups = plot_df.groupby("year")
    year_avg_scores = year_groups[score_column].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=year_avg_scores.index, y=year_avg_scores.values, marker="o")
    plt.title(title)
    plt.xlabel("연도")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


# 평균 감정 점수 상위 N명
def plot_topN_artists(df, n=10, score_column="emotion_score", title="상위 N명 아티스트 감정 점수", 
                     exclude_outliers=False, normalized=False, save=False, save_path=None):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
        title += "(이상치 제외)"
    if normalized:
        score_column = "normalized_emotion_score"
        title += "(정규화)"
    
    top_artists = plot_df.groupby("artist")[score_column].mean().nlargest(n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_artists.index, y=top_artists.values, palette="Set2")
    plt.title(title)
    plt.xlabel("아티스트")
    plt.ylabel("평균 감정 점수")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 상위 아티스트 감정 프로필 레이더 차트
def plot_topN_artist_emotion_profile(df, n=10, emotion_list=None, title="상위 N명 아티스트 감정 프로필", save=False, save_path=None):
    if emotion_list is None:
        emotion_list = [col for col in df.columns if col.startswith('count_')]

    top_artists = df.groupby("artist")[emotion_list].mean().nlargest(n)

    categories = list(top_artists.columns)
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i in range(len(top_artists)):
        values = top_artists.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, label=top_artists.index[i])

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 감정군별 대표 아티스트 highlight plot
def plot_emotion_artist_highlight(df, emotion_list, title="감정군별 대표 아티스트 highlight plot", save=False, save_path=None):
    emotion_artist_count = df.groupby("artist")[emotion_list].sum()
    emotion_artist_count = emotion_artist_count.div(emotion_artist_count.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(emotion_artist_count, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("아티스트")
    plt.tight_layout()

    if save:
        if save_path is None:
            filename = title
            save_path = f"../results/plots/카테고리 분석/아티스트/{filename}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

#####################################################################################################################################
# 감정군 중심 네트워크 그래프 - 감정군과 장르/아티스트를 연결하는 감정 네트워크


# Sankey Diagram - 감정군 ↔ 장르 ↔ 시대 흐름을 시각화


# 감정군 기준 클러스터링 + t-SNE/UMAP - 감정 분포 기반 곡/장르 클러스터 시각화


# 감정군 기준 클러스터링 + t-SNE/UMAP - 감정 분포 기반 곡/시대 클러스터 시각화


# 감정군 기준 클러스터링 + t-SNE/UMAP - 감정 분포 기반 곡/아티스트 클러스터 시각화

