"""lyrics_reco.pipeline

Executable pipelines (CLI-style modules).

Included:
- run_preprocess.py: build data/processed/genius_processed.csv (CSV-only)
- run_baseline.py: baseline retrieval + evaluation (lexicon-vector default)
- inspect_year_distribution.py: compare year distribution overall vs Top-N by views
- utils.py: shared helpers (config access, query sampling, I/O)
"""
