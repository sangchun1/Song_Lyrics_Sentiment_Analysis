# 🎵 노래 가사 기반 감정 분석 프로젝트
감정 기반 추천을 위한 노래 가사 분석

## 📌 프로젝트 개요
이 프로젝트는 노래 가사 데이터를 기반으로 감정 분석을 수행하고, 장르·시대·아티스트별 감정 트렌드를 파악하며, 감정적으로 유사한 곡들을 클러스터링하여 감정 기반 추천 시스템의 기반을 마련하는 것을 목표로 합니다.

### 주요 작업
- NRC 감정 사전 기반 감정군 추출 및 점수화
- 장르, 시대, 아티스트별 감정 변화 분석
- 감정 단어 기반 TF-IDF 벡터화
- 다양한 클러스터링 기법을 통한 군집화 및 군집 해석
- 감정 기반 유사도 추천 알고리즘 설계

---

## 🔍 연구 배경 및 목적

### 연구 배경
- 음악 소비는 감정과 밀접하게 연결되어 있으며, 기존 추천 알고리즘은 주로 장르·아티스트·인기 순에 기반
- 감정적으로 유사한 가사를 기반으로 한 노래 추천이 감정적 몰입도를 높일 수 있음

### 연구 목적
- 가사 기반 감정 분석을 통해 감정 흐름과 트렌드 탐색
- 감정 및 유사도 기반 클러스터링을 활용한 개인화 노래 추천 구현

---

## ⏳ 프로젝트 일정

| 주차 | 내용 |
|------|------|
| 1주차 | 데이터 수집 및 전처리 모듈 구현 |
| 2주차 | 감정 단어 기반 분석 및 점수화, 시각화 모듈화 |
| 3주차 | 장르·연도·아티스트별 감정 트렌드 분석 |
| 4주차 | TF-IDF 기반 유사도 분석 및 클러스터링, 추천 시스템 구현 |

---

## 🎯 기대 효과

- 시대별, 장르별, 아티스트별 감정 흐름 분석 역량 강화
- 감정 기반 추천 시스템 설계 능력 확보
- NLP 기반 감정 분석, 벡터화, 클러스터링 기술에 대한 실무 경험 축적

---

## 🗂 프로젝트 폴더 구조

```
📁 dataset/            # 원본 데이터 (Genius, Top100)
📁 data/               # 전처리된 파일 및 감정 사전
📁 preprocess/         # 전처리 함수, 테스트 노트북
📁 analysis/           # 분석용 노트북 (감정, 카테고리, 유사도)
📁 module/
│   ├── category/      # 장르/연도/아티스트 분석 함수
│   ├── similarity/    # TF-IDF, 클러스터링, 추천 함수
│   └── visualization.py
📁 results/            # 분석 결과(pkl), 시각화 이미지 저장
📄 README.md
```

---

## 🗂 데이터셋 정보

1. [**Genius Song Lyrics**](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
* 약 500만 개 이상의 방대한 노래 가사 포함
* 곡 제목, 아티스트, 장르, 연도, 언어, 조회수 등 다양한 메타데이터 수록
* 다국어 가사도 일부 포함되어 있어 FastText 기반 언어 감지 모델을 활용해 영어 가사만 필터링
* 장르 기반 감정 분석, 아티스트 감정 성향 파악, 유사도 분석을 위한 학습 코퍼스로 활용
* 데이터 양이 방대하므로 전처리 속도 최적화를 위해 병렬 처리 기법 적용

2. [**Top 100 Songs & Lyrics by Year**](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)
* 1959년부터 2023년까지 매년 Billboard Top 100 수록곡 포함
* 약 6,500곡 규모의 샘플이지만, 연도별 감정 트렌드 분석에 매우 유용
* 곡 제목, 아티스트, 발표 연도, 순위, 가사 등 주요 정보 포함
* 감정 변화의 시대적 흐름을 탐색하는 데 핵심 역할

> 영어 가사 필터링, 반복 후렴 제거, 줄임말/속어 치환, 특수문자 정리, 불용어 제거, spaCy 기반 표제어화 등 전처리 수행

---

## 🛠 사용 기술

- **언어 및 분석**: Python, Pandas, Numpy, Scikit-learn, NLTK, spaCy
- **감정 분석**: NRC Emotion Lexicon 기반, 감정군 축소 및 점수화
- **벡터화 및 시각화**: TF-IDF, t-SNE, matplotlib, seaborn, WordCloud
- **군집화**: KMeans, GMM, MeanShift, DBSCAN
- **추천 알고리즘**: 감정 벡터 유사도 기반 추천

---

## 📊 분석 및 기능 구성

### 1️⃣ [전처리](https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis/blob/main/preprocess/preprocess.ipynb)
- 영어 가사 필터링: FastText를 통한 언어 감지 및 제거
- 반복 후렴구 제거: 의미 중복 제거를 통한 감정 단어 정제
- 줄임말 및 속어 정규화: "gonna → going to", "cuz → because" 등 사전 기반 치환
- 특수문자 제거 및 정규화: \n, \t, [, ] 등 제거
- 불용어 제거 및 표제어화: 감탄사, 의미없는 소리 제거 + spaCy 기반 lemmatization
- 감정 단어 필터링 및 매핑: NRC 감정 사전을 기반으로 한 감정군 할당

### 2️⃣ [감정 분석](https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis/blob/main/analysis/emotion_lexicon_analysis.ipynb)
- 감정군: love, hope, sadness, fear, disgust, anger (총 6개)
- 감정 점수화: 각 감정에 가중치 부여 (예: love=+1.0, anger=–1.0)
- 감정 점수 분포 시각화, 감정군별 워드클라우드 생성

### 3️⃣ [카테고리 분석](https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis/blob/main/analysis/category_analysis.ipynb)
- **장르별 분석**: 감정군 비율 및 평균 점수 시각화
- **연도별 분석**: 감정군 비율 및 점수의 시대적 변화 시계열 분석
- **아티스트별 분석**: 감정군 비율 및 시계열 추이 (예: Bruno Mars)

### 4️⃣ [유사도 분석 및 클러스터링](https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis/blob/main/analysis/similarity_clustering_analysis.ipynb)
- 감정 단어 기반 TF-IDF 벡터화 (≈ 44,286곡 × 3,422단어)
- 클러스터링 기법:
  - **KMeans**:
    - 군집 0: 부정·범죄·폭력 (랩)
    - 군집 1: 사랑·감성 (팝/R&B)
    - 군집 2: 희망·극복 (팝/록)
  - **GMM**:
    - 군집 0: 범죄·생존 테마 랩
    - 군집 1: 이별·감정적 아픔 표현
    - 군집 2: 희망·성장 중심 팝/록
- 각 군집의 감정 점수, 장르 분포, 워드클라우드 시각화

### 5️⃣ [개인화 추천 시스템](https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis/blob/main/analysis/similarity_clustering_analysis.ipynb)
- 사용자가 선택한 곡의 감정 벡터 기반 유사도 계산
- 가장 유사한 곡들을 클러스터 및 감정 기반으로 추천
- 예시: Viva La Vida → IU, Joji 등 감정적으로 유사한 곡 추천

---

## ⚠ 한계점 및 향후 계획

### 한계점
- 감정 사전 기반 분석은 문맥 이해 및 복합 감정 표현에 한계
- 일부 가사는 감정 단어 수가 적어 군집화 정밀도에 한계

### 향후 계획
- BERT, RoBERTa 등 딥러닝 기반 문맥 이해형 감정 분류 모델 도입
- 사용자 감정 선호도 및 청취 이력을 반영한 개인화 추천 알고리즘 고도화
- 감정 흐름 기반 시계열 모델 및 감정 네트워크 분석 확장
