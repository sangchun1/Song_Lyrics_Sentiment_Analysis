"""
lyrics_reco.baseline

Baseline representations & retrieval utilities.

Baseline options:
- TF-IDF on lyrics_dedup (general lexical baseline)
- Emotion-only TF-IDF (tokens restricted to NRC lexicon words)
- Lexicon feature vectors (emotion counts/ratios, optional intensity/VAD)
- Cosine Top-K retrieval on sparse matrices

Tabular outputs should be saved as CSV via lyrics_reco.common.io.
"""

from .tokenize import simple_tokenize, load_word_set, prepare_text_series
from .tfidf import (
    build_tfidf,
    build_emotion_tfidf,
    compute_term_weights_from_intensity,
    apply_term_weights,
    save_vocab_idf_csv,
    load_vocab_idf_csv,
)
from .emotion_features import build_lexicon_feature_table
from .similarity import topk_cosine_for_index, batch_topk_cosine