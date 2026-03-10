# Song Lyrics Emotion-Aware Music Recommendation

A lyrics-based music recommendation project that compares:

- **Baseline**: TF-IDF song representation with emotion-aware term weighting
- **Proposed**: emotion-context vector representation

The main entry point for quick use is **`demo.py`**.  
Given one input song, it retrieves top-k similar songs from both models and shows how emotionally similar they are.

---

## What this project does

This project uses song lyrics to recommend emotionally similar songs.

The demo:

- takes a query song by `song_id` or `title` + `artist`
- loads saved **baseline** TF-IDF vectors and **proposed** vectors
- returns top-k recommendations from each model
- shows:
  - model similarity score
  - emotion similarity score
- saves results to `artifacts/demo/`

---

## Install

```bash
git clone https://github.com/sangchun1/Song_Lyrics_Sentiment_Analysis.git
cd Song_Lyrics_Sentiment_Analysis

pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Run the demo

Recommended:

```bash
python -m lyrics_reco.cli.demo --title "Hello" --artist "Adele" --k 10
```

You can also run it directly from the repository root:

```bash
python src/lyrics_reco/cli/demo.py --title "Hello" --artist "Adele" --k 10
```

### 2. Query by `song_id`

```bash
python -m lyrics_reco.cli.demo --song-id SONG_000123 --k 10
```

### 3. Specify vector files manually

```bash
python -m lyrics_reco.cli.demo \
  --title "Hello" \
  --artist "Adele" \
  --k 10 \
  --baseline-vectors artifacts/runs/<baseline_run>/baseline_tfidf_weighted.npz \
  --baseline-song-ids artifacts/runs/<baseline_run>/baseline_tfidf_song_ids.npy \
  --proposed-vectors artifacts/runs/<proposed_run>/emotion_context_vectors.csv
```

---

## Required inputs for the demo

The demo expects:

* processed song metadata CSV

  * default: `data/processed/genius_processed.csv`
* baseline vectors

  * default priority:

    1. `--baseline-vectors`
    2. `artifacts/vectors/baseline_tfidf_weighted.npz`
    3. `artifacts/vectors/baseline_tfidf.npz`
    4. `artifacts/vectors/baseline_vectors.npz`
    5. `artifacts/vectorizers/baseline_tfidf_weighted.npz`
    6. `artifacts/vectorizers/baseline_vectors.npz`
    7. latest `artifacts/runs/*/baseline_tfidf_weighted.npz`
    8. latest `artifacts/runs/*/baseline_tfidf.npz`
    9. latest `artifacts/runs/*/baseline_vectors.npz`
    10. latest `artifacts/runs/*/baseline_lexicon_features.csv`
* baseline song id mapping for `.npz` baseline vectors

  * optional but recommended when loading sparse TF-IDF vectors
  * default priority:

    1. `--baseline-song-ids`
    2. `artifacts/vectors/baseline_song_ids.npy`
    3. `artifacts/vectors/baseline_tfidf_song_ids.npy`
    4. latest `artifacts/runs/*/baseline_song_ids.npy`
    5. latest `artifacts/runs/*/baseline_tfidf_song_ids.npy`
* proposed vectors

  * default priority:

    1. `--proposed-vectors`
    2. latest `artifacts/runs/*/emotion_context_vectors.csv`
* emotion config

  * default: `configs/emotion_context.yaml`

---

## Output

The demo saves files under:

```text
artifacts/demo/
```

Typical outputs:

* `*_baseline_topK.csv`
* `*_proposed_topK.csv`
* `*_summary.json`

Each recommendation table includes fields such as:

* rank
* recommended song id
* title / artist
* year / genre
* model score
* emotion similarity
* emotion similarity (%)

---

## Useful options

```bash
--song-id              Exact query song id
--title                Query title
--artist               Query artist
--k                    Number of recommendations
--top-m                Candidate pool size before reranking
--emotion-space        auto | ratio | ratio_vad
--baseline-song-ids    Optional song_id order for baseline `.npz` vectors
--baseline-use-mmr     Apply MMR to baseline too
--baseline-lambda      Baseline MMR lambda
--proposed-disable-mmr Disable MMR for proposed model
--proposed-lambda      Proposed MMR lambda
--save-dir             Output directory
--output-prefix        Output filename prefix
```

---

## Minimal workflow

If you already finished preprocessing and vector generation, you only need this:

```bash
python -m lyrics_reco.cli.demo --title "Hello" --artist "Adele" --k 10
```

If the vectors are not found automatically, pass them explicitly:

```bash
python -m lyrics_reco.cli.demo \
  --title "Hello" \
  --artist "Adele" \
  --baseline-vectors <path_to_baseline_vectors> \
  --baseline-song-ids <path_to_baseline_song_ids> \
  --proposed-vectors <path_to_proposed_vectors>
```

---

## Project structure

```text
src/lyrics_reco/
├── baseline/
├── common/
├── emotion_context/
├── evaluation/
├── lexicon/
├── pipeline/
├── preprocess/
├── retrieval/
├── vectordb/
└── cli/
    └── demo.py
```

---

## Notes

* Run commands from the repository root.
* The processed CSV should contain at least `song_id`.
* For title-based search, `title` is required and `artist` helps disambiguate duplicates.
* The demo compares **baseline vs proposed** on the same query song, so it is useful for quick qualitative checks and presentations.
* For TF-IDF baseline vectors saved as `.npz`, keeping the matching `baseline_song_ids.npy` or `baseline_tfidf_song_ids.npy` file is recommended.

---

## License

This repository is released under the GPL-3.0 License.

