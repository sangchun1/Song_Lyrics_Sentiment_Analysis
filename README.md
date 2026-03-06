# Song Lyrics Sentiment Analysis for Emotion-Aware Music Recommendation

A research-oriented music recommendation project that analyzes song lyrics, builds emotion-aware representations, and retrieves emotionally similar songs using both a **baseline lexicon-based pipeline** and a **proposed emotion-context vector pipeline**.

---

## Overview

This repository explores how lyrical emotion can be used as a signal for music recommendation.

The project starts from a **baseline representation** based on lexicon-derived emotion ratios and extends it to a **proposed emotion-context representation** that combines:

- line-level text embeddings
- emotion ratio features
- emotion intensity features
- VAD (Valence, Arousal, Dominance) features
- diversity-aware retrieval with MMR

The overall goal is to build a content-based recommendation system that captures not only lexical overlap, but also the **emotional and semantic structure** of song lyrics.

---

## Key Ideas

### 1. Baseline
The baseline represents each song using lexicon-based emotion features derived from lyrics.

- NRC-based emotion ratio vectors
- cosine similarity retrieval
- optional candidate filtering
- offline evaluation with pseudo ground truth

### 2. Proposed Method
The proposed pipeline builds a song-level **emotion-context vector** by aggregating line embeddings with emotion-aware weights.

Core intuition:

- emotionally important lyric lines should contribute more
- repeated lyrical structure should not dominate embeddings
- semantic context and emotion structure should be modeled together
- retrieval should balance relevance and diversity

---

## What This Repository Contains

- preprocessing pipeline for raw lyrics data
- lexicon-based baseline retrieval pipeline
- emotion-context vector construction pipeline
- Chroma-based vector indexing and querying
- retrieval with cosine similarity + MMR reranking
- evaluation pipeline for ranking quality and diversity

---

## Method Summary

### Baseline Representation
The baseline pipeline extracts lexicon-based song features from cleaned lyrics and uses them for retrieval.

Typical components:
- emotion ratios
- optional intensity / VAD augmentation
- cosine similarity ranking

### Proposed Representation: Emotion-Context Vector
The proposed representation builds a song vector from lyric lines:

1. split lyrics into lines
2. deduplicate repeated lines for embedding
3. embed each line with a sentence-transformer model
4. compute line-level emotion / intensity / VAD signals
5. assign higher weights to emotionally aligned lines
6. aggregate into a song-level vector
7. index song vectors in a vector database
8. retrieve candidates and rerank with MMR

---

## Datasets

This project is designed for English lyrics corpora and has been developed around datasets such as:

- **Genius Song Lyrics**
- **Top 100 Songs & Lyrics by Year**

Typical metadata used in the project:
- `song_id`
- `title`
- `artist`
- `year`
- `genre`
- `lyrics`

---

## Preprocessing

The preprocessing stage is designed for large-scale lyrics data and includes:

- language filtering
- text cleaning and normalization
- whitespace normalization
- line splitting
- optional repeated-line handling
- preparation of text for both:
  - lexicon-based analysis
  - embedding-based emotion-context modeling

The current pipeline also supports FastText-based language filtering.

---

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis.git
cd Song_Lyrics_Sentiment_Analysis
pip install -e .
```

For development dependencies:
```
pip install -e ".[dev]"
```

Because the source code is under `src/`, run commands from the repository root with `PYTHONPATH=src`.

### Bash / zsh
```bash
export PYTHONPATH=src
```

### PowerShell
```powershell
$env:PYTHONPATH = "src"
```

---

## Required Assets

Before running the pipelines, prepare the required resources under `assets/` according to your config files.

Examples:
- FastText language ID model
- NRC emotion lexicon
- NRC emotion intensity lexicon
- VAD lexicon

You may need to adjust paths in:
- `configs/data.yaml`
- `configs/main.yaml`
- `configs/emotion_context.yaml`

---

## Quick Start

### 1. Preprocess Raw Lyrics

```bash
python -m lyrics_reco.pipeline.run_preprocess \
  --input data/raw/lyrics.csv \
  --output data/processed/genius_processed.csv
```

Example options you may additionally use:
- `--start-year`
- `--end-year`
- `--use-fasttext`
- `--fasttext-model`
- `--fasttext-threshold`
- `--expand-multi-artist`

---

### 2. Run Baseline Retrieval + Evaluation

```bash
python -m lyrics_reco.pipeline.run_baseline \
  --data data/processed/genius_processed.csv \
  --eval-config configs/eval.yaml \
  --retrieval-config configs/retrieval.yaml \
  --emotion-config configs/emotion_context.yaml \
  --save-vectors-csv
```

Optional flags:
- `--include-intensity`
- `--include-vad`
- `--n-queries`
- `--top-m`
- `--top-k`
- `--mmr-lambda`
- `--disable-mmr`

---

### 3. Run Proposed Emotion-Context Pipeline

```bash
python -m lyrics_reco.pipeline.run_proposed \
  --data data/processed/genius_processed.csv \
  --eval-config configs/eval.yaml \
  --retrieval-config configs/retrieval.yaml \
  --emotion-config configs/emotion_context.yaml \
  --rebuild-index
```

If vectors are already built, you can reuse them:

```bash
python -m lyrics_reco.pipeline.run_proposed \
  --data data/processed/genius_processed.csv \
  --vectors-csv artifacts/runs/<run_id>/emotion_context_vectors.csv
```

Optional flags:
- `--n-queries`
- `--top-m`
- `--top-k`
- `--disable-mmr`
- `--mmr-lambda`
- `--ild-space {emotion_ratio, embedding, z}`

---

## Configuration

The repository is config-driven.

### `configs/data.yaml`
Controls:
- raw / processed paths
- schema
- language filtering
- text cleaning
- line processing
- deduplication for embedding text

### `configs/emotion_context.yaml`
Controls:
- embedding model
- line splitting rules
- emotion / intensity / VAD usage
- line weighting strategy
- song-level aggregation

### `configs/retrieval.yaml`
Controls:
- VectorDB backend
- Chroma persistence path
- retrieval depth (`top_m`)
- final cutoff (`top_k`)
- MMR reranking
- candidate filtering

### `configs/eval.yaml`
Controls:
- number of sampled queries
- `K` values for evaluation
- pseudo ground truth construction
- enabled metrics

### `configs/baseline.yaml`
Controls:
- baseline method
- TF-IDF settings
- emotion weighting behavior
- output paths

---

## Evaluation

The repository currently supports offline evaluation with the following metrics:

- **Recall@K**
- **NDCG@K**
- **Emotion Consistency@K**
- **ILD@K** (Intra-List Diversity)

The default evaluation setup uses **pseudo ground truth**, typically based on metadata conditions such as:
- same genre
- year proximity
- optional exclusion of the same song / artist

This makes the framework useful for research iteration even when explicit user-feedback labels are unavailable.

---

## Output Files

Each run creates structured artifacts under `artifacts/runs/<run_id>/` and `reports/runs/<run_id>/`.

Typical outputs include:

### Preprocessing
- processed CSV

### Baseline
- `baseline_lexicon_features.csv`
- `baseline_recommendations.csv`
- evaluation tables / metrics
- run config and logs

### Proposed
- `emotion_context_vectors.csv`
- `proposed_recommendations.csv`
- `per_query_metrics.csv`
- `summary_metrics.csv`
- run config and logs

---

## Current Research Focus

This repository is currently organized around the following research direction:

1. maintain a lexicon-based baseline
2. build an emotion-context vector from line embeddings and emotion features
3. compare baseline vs proposed retrieval quality
4. evaluate both relevance and diversity
5. analyze whether emotion-aware representations improve recommendation quality and interpretability

---

## Planned Extensions

- late fusion between baseline and proposed representations
- improved lyric normalization for slang / contractions / repeated structures
- larger embedding models for lyric semantics
- richer pseudo-ground-truth construction
- qualitative cluster interpretation
- user study for perceived emotional naturalness
- hybrid recommendation with user-song interaction data

---

## Reproducibility Notes

To keep experiments reproducible:

- run all commands from the repository root
- keep config files version-controlled
- store generated artifacts by run ID
- keep lexicon and embedding settings fixed for fair comparisons
- compare baseline and proposed pipelines under matched evaluation settings

---

## License

This repository is released under the **GPL-3.0 License**.

---

## Acknowledgements

This project builds on:
- lyrics-based emotion analysis
- sentence-transformer text embeddings
- lexicon-based affect modeling
- diversity-aware retrieval with MMR
- vector search with Chroma

It is developed as a research-oriented codebase for exploring **emotion-aware music recommendation from song lyrics**.