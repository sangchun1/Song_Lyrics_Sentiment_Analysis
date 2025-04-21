# 🎵 노래 가사를 활용한 감정 분석 프로젝트

## 📌 프로젝트 개요

이 프로젝트는 노래 가사 데이터를 활용하여 감정 분석을 수행하고, 장르 및 시대에 따른 감정 트렌드를 탐색하는 것을 목표로 합니다. 주요 작업은 다음과 같습니다:

- 기쁨, 슬픔, 분노, 사랑 등의 세부 감정 감지
- 장르, 시대 및 아티스트 감정 변화 분석
- 감정적으로 유사한 노래 그룹화 및 클러스터링
- 유사 노래 군집 간 특성 분석 및 시대/장르 감정 트렌드 연관성 탐색


## 🔍 연구 배경 및 필요성

음악은 인간의 감정을 표현하는 중요한 예술 형태입니다. 그러나 수많은 노래 중 특정 감정에 맞는 곡을 찾기는 쉽지 않습니다. 본 프로젝트는 노래 가사의 감정을 자동으로 분석하여 사용자 맞춤형 음악 추천의 기반을 마련하고자 합니다. 또한, 시대적·장르적 흐름 속에서 음악의 감정 변화를 살펴보며 문화적 통찰을 얻는 데에도 기여하고자 합니다.

## 🗂️ 데이터셋 정보

1. [**Genius Song Lyrics**](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)  
   - 약 6백만 개의 다양한 노래 가사 포함
   - 각 곡의 제목, 아티스트, 장르(tag), 연도, 언어, 조회수 등의 메타데이터 포함  
   - 일부 다국어 가사는 `Genius English Translations` 형태로 제공되어 번역 가사도 활용 가능

2. [**Top 100 Songs & Lyrics by Year**](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)  
   - 1959년부터 2023년까지 매년 미국 빌보드 Top 100 차트 수록곡 포함  
   - 제목, 아티스트, 발표 연도, 순위, 가사 등 정보 포함  
   - 최신 트렌드 반영에 유리하며 시대별 감정 흐름 분석에 활용

> 전처리 과정에서는 영어 가사만 필터링되었으며, 반복 후렴 제거, 줄임말 정규화, 특수문자 제거, 불용어 제거 등을 수행하였습니다.

## ⏳ 프로젝트 일정 (4주)

| 주차  | 내용 |
|-------|------|
| 1주차 | 데이터 수집 및 정제, 전처리 모듈화 완료 |
| 2주차 | 감정 단어 기반 분석, 감정 점수 계산 및 정규화 기준 확정, 시각화 모듈화 |
| 3주차 | 전체 감정 점수 계산 적용, 장르/시대/아티스트별 감정 트렌드 분석 및 시각화 |
| 4주차 | TF-IDF 기반 감정 유사도 분석 및 클러스터링, 결과 정리 및 보고서 작성 |

## 🛠️ 사용 기술

- Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, NLTK)
- 텍스트 처리: 정규표현식 기반 전처리, 형태소 분석 (NLTK TreebankTokenizer)
- 감정 분석: NRC Emotion Lexicon 기반 사전 감정 분석
- 병렬 처리: FastText 언어 식별 모델 활용
- 시각화: Matplotlib, Seaborn, Wordcloud
- 분석 기법: 감정 점수 계산, TF-IDF 기반 유사도 분석, 코사인 유사도, KMeans 클러스터링

## 📊 주요 기능  

### 전처리
- 영어 가사 필터링 및 클렌징  
- 후렴 및 반복구 제거  
- 줄임말 및 특수문자 처리  
- 불용어 제거 및 형태소 분석  
- 병렬 처리 최적화  

### 감정 분석
- 감정 단어 기반 특징 추출  
- 감정 점수 계산 및 정규화  
- 세부 감정 감지 및 시각화  
- 장르/연도/아티스트별 감정 트렌드 분석  

### 유사도 분석 및 클러스터링
- TF-IDF 기반 벡터화  
- 코사인 유사도 계산  
- 감정 기반 클러스터링 수행 (KMeans)
- 군집 간 특성 비교 분석 및 시대/장르별 감정 트렌드 연관성 탐색

### 시각화 및 결과 관리
- 감정 점수 분포 및 트렌드 시각화 자동화  
- 시각화 결과 및 분석 테이블 자동 저장  
- plots 폴더에 자동 저장 및 파일명 지정   

## 🗂️ 프로젝트 구조

📁 Song Lyrics Sentiment Analysis  
├── 📂 dataset/   *# 원본 데이터 (Kaggle에서 수집)*  
│   ├── 📄 Genius Song Lyrics.csv  
│   └── 📄 Top 100 Songs & Lyrics By Year 1959 - 2023 (USA).csv  
│  
├── 📂 data/   *# 전처리된 데이터 및 샘플 저장 및 감정 사전 저장*   
│   ├── 📄 NRC-Emotion-Lexicon-Wordlevel-v0.92.txt    *# 감정사전*   
│   ├── 📄 filler_words.txt                           *# 노래 가사용 불용어 사전(추가)*   
│   ├── 📄 lid.176.bin                                *# FastText 언어 식별 모델*   
│   ├── 📄 genius_cleaned.pkl   
│   └── 📄 top100_cleaned.pkl   
│  
├── 📂 preprocess/   *# 전처리 코드와 노트북*  
│   ├── 📄 preprocessing.py   *# 함수 모듈화된 전처리 코드*  
│   └── 📄 preprocess.ipynb   *# 데이터 전처리 실험 및 확인용 노트북*  
│  
├── 📂 analysis/   *# 분석 및 시각화 노트북*   
│   ├── 📄 emotion_lexicon_analysis.ipynb    *# 감정 단어 및 감정 점수 기반 분석*   
│   ├── 📄 category_analysis.ipynb           *# 장르별, 시대별, 아티스트별 분석*   
│   └── 📄 visualization.ipynb               *# 시각화*   
│  
├── 📂 module/   *# 분석 및 시각화 함수 모듈* 
│   ├── 📂 category    *# 카테고리 분석 모듈 패키지*   
│   │   ├── 📄 __init__.py    
│   │   ├── 📄 category_utils.py  *# 카데고리 분석 공통 함수 모듈*   
│   │   ├── 📄 genre_analysis.py  *# 장르별 분석 함수 모듈*   
│   │   ├── 📄 decade_analysis.py *# 시대별 분석 함수 모듈*   
│   │   └── 📄 artist_analysis.py *# 아티스트별 분석 함수 모듈*  
│   ├── 📂 similarity   *# 유사도 및 클러스터링 분석 모듈 패키지*  
│   │   ├── 📄 __init__.py   
│   │   ├── 📄 similarity_analysis.py *# 유사도 분석 함수 모듈*   
│   │   ├── 📄 clustering_analysis.py *# 클러스터링 분석 함수 모듈*   
│   │   └── 📄 recommendation_engine.py *# 추천 알고리즘 함수 모듈*  
│   ├── 📄 __init__.py  
│   └── 📄 visualization.py   *# 시각화 함수 모듈*   
│  
├── 📂 results/   *# 시각화 이미지, 보고서, 발표자료*    
│   ├── 📂 plots  *# 시각화 이미지*  
│   ├── 📄 genius_emotion_analysis.pkl    *# Genius Song Lyrics 데이터 감정 분석 결과*   
│   ├── 📄 top100_emotion_analysis.pkl    *# Top 100 Songs & Lyrics 데이터 감정 분석 결과*   
│   ├── 📄 emotion_analysis.pkl           *# 전체 데이터 감정 분석 결과*   
│   └── 📄 category_analysis.pkl          *# 카데고리 분석 결과*    
│  
└── 📄 README.md   *# 프로젝트 소개 문서*

## 🎯 기대 효과

- 실제 노래 가사 데이터를 활용한 텍스트 분석 실무 경험
- 감정 기반 노래 추천 시스템의 기초 구현
- 시대별 감정 흐름을 통해 음악과 사회 문화적 변화 탐색
- 감성 분석 및 전처리 자동화 능력 향상
- 향후 개인화 추천 알고리즘 설계 기반 마련

## ⚠️ 한계점 및 향후 연구 방향

- 감정 분석은 사전 기반 규칙에 의존하여 복합 감정을 반영하기 어려움
- 추후 확장 방향:
  - 감정 강도 회귀 예측
  - 딥러닝 기반 멀티태스크 감정 분류
  - 감정 흐름 시계열 모델링
  - 감정 그래프 생성 및 음악 네트워크 분석
