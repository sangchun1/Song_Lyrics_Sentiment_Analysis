import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from scipy.stats import linregress
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

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

def plot_emotion_count_bar(df, emotion_list, title="감정군별 감정 단어 수 분포"):
    totals = {emo: df[f"count_{emo}"].sum() for emo in emotion_list}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(totals.keys()), y=list(totals.values()), palette="Blues_d")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("총 감정 단어 수")
    plt.tight_layout()
    plt.show()

def plot_emotion_ratio_bar(df, emotion_list, title="감정군별 감정 단어 비율 (평균)"):
    ratios = {emo: df[f"ratio_{emo}"].mean() for emo in emotion_list}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(ratios.keys()), y=list(ratios.values()), palette="Oranges_d")
    plt.title(title)
    plt.xlabel("감정군")
    plt.ylabel("평균 비율 (%)")
    plt.tight_layout()
    plt.show()

def generate_overall_wordcloud(df, title="전체 감정 단어 WordCloud"):
    plot_df = df.copy()
    plot_df["emotion_words"] = plot_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    all_words = [word for sublist in plot_df["emotion_words"].dropna() for word in sublist]
    all_emotion_words = " ".join(all_words)
    wc = WordCloud(width=800, height=400, collocations=False, background_color="white").generate(all_emotion_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def generate_grouped_wordclouds(df, selected_emotion, title="감정군 감정 단어 WordCloud"):
    plot_df = df.copy()
    plot_df["emotion_words"] = plot_df["emotion_words"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else x)
    word_to_emotions = load_emotion_lexicon()
    emotion_word_dict = defaultdict(list)
    for word_list in plot_df["emotion_words"]:
        for word in word_list:
            if word in word_to_emotions:
                for emo in word_to_emotions[word]:
                    emotion_word_dict[emo].append(word)
    # if selected_emotion not in emotion_word_dict:
    #     print(f"'{selected_emotion}' 그룹이 존재하지 않습니다.")
    #     return
    text = " ".join(emotion_word_dict[selected_emotion])
    wc = WordCloud(width=800, height=400, collocations=False, background_color="white").generate(text)
    plt.figure(figsize=(7, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{selected_emotion} {title}")
    plt.tight_layout()
    plt.show()

def plot_top_emotion_words_per_song(df, target_title=None, top_n=5):
    if target_title:
        row = df[df['title'] == target_title].iloc[0]
        word_columns = [col for col in df.columns if col.startswith("count_")]
        word_freq = {col: row[col] for col in word_columns if row[col] > 0}
        top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])
        if top_words:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette="coolwarm")
            plt.title(f"곡: {row['title']} - Top {top_n} 감정 단어")
            plt.ylabel("빈도")
            plt.tight_layout()
            plt.show()
    else:
        for idx, row in df.iterrows():
            word_columns = [col for col in df.columns if col.startswith("count_")]
            word_freq = {col: row[col] for col in word_columns if row[col] > 0}
            top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])
            if top_words:
                plt.figure(figsize=(6, 4))
                sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette="coolwarm")
                plt.title(f"곡: {row['title']} - Top {top_n} 감정 단어")
                plt.ylabel("빈도")
                plt.tight_layout()
                plt.show()

def plot_emotion_word_correlation(df, title="감정 단어 수 상관관계 히트맵"):
    word_columns = [col for col in df.columns if col.startswith("count_")]
    corr = df[word_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_emotion_score_histogram(df, score_column="emotion_score", bins=30, title="감정 점수 분포 히스토그램", exclude_outliers=False):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df[score_column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel("감정 점수")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

def plot_normalized_emotion_score_histogram(df, score_column="normalized_emotion_score", bins=30, title="감정 점수 분포 히스토그램 (정규화)"):
    plot_df = df.copy()
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df[score_column], bins=bins, kde=True, color="green")
    plt.title(title)
    plt.xlabel("감정 점수 (정규화)")
    plt.ylabel("곡 수")
    plt.tight_layout()
    plt.show()

def plot_avg_emotion_score(df, selected_emotion, title="감정군별 평균 감정 점수"):
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
    plt.show()

def plot_emotion_ratio_vs_score(df, emotion_name, score_column="emotion_score", exclude_outliers=False):
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    x = plot_df[score_column]
    y = plot_df[f"ratio_{emotion_name}"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.plot([], [], ' ', label=f"$r$ = {r_value:.2f}")
    plt.plot([], [], ' ', label=f"$R^2$ = {r_squared:.2f}")
    plt.title(f"{emotion_name.capitalize()} 비율 vs 감정 점수", fontsize=13)
    plt.xlabel("감정 점수")
    plt.ylabel(f"{emotion_name.capitalize()} 비율 (%)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_emotion_ratio_vs_normalized_score(df, emotion_name, score_column="normalized_emotion_score"):
    x = df[score_column]
    y = df[f"ratio_{emotion_name}"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.plot([], [], ' ', label=f"$r$ = {r_value:.2f}")
    plt.plot([], [], ' ', label=f"$R^2$ = {r_squared:.2f}")
    plt.title(f"{emotion_name.capitalize()} 비율 vs 감정 점수", fontsize=13)
    plt.xlabel("감정 점수")
    plt.ylabel(f"{emotion_name.capitalize()} 비율 (%)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def show_top_songs_by_emotion_score(df, score_column="emotion_score", top_n=5, is_bottom=False, exclude_outliers=False):
    from IPython.display import display
    plot_df = df.copy()
    if exclude_outliers:
        Q1 = plot_df[score_column].quantile(0.25)
        Q3 = plot_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        plot_df = plot_df[(plot_df[score_column] >= lower_bound) & (plot_df[score_column] <= upper_bound)]
    top_songs = plot_df.sort_values(score_column, ascending=is_bottom).head(top_n)
    display(top_songs[['title', 'artist', score_column, "emotion_score_detail"]])

def show_top_songs_by_normalized_emotion_score(df, score_column="normalized_emotion_score", top_n=5, is_bottom=False):
    from IPython.display import display
    top_songs = df.sort_values(score_column, ascending=is_bottom).head(top_n)
    display(top_songs[['title', 'artist', score_column, "emotion_score", "emotion_score_detail"]])