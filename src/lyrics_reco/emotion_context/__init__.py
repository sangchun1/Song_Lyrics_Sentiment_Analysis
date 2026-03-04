"""lyrics_reco.emotion_context

Build "Emotion-Context" vectors z(s) for songs.

Concept:
- Split lyrics into lines
- Encode each line with a sentence embedding model (Sentence-Transformers)
- Compute lexicon features per line (NRC emotions + optional intensity + optional VAD)
- Compute a per-line weight w(line) from emotion/intensity/VAD
- Aggregate line embeddings/features into a song vector z(s)

Main entrypoints:
- build_song_vectors_from_df (streaming/batched)
- EmotionContextBuilder (convenience wrapper)
"""

from .builder import EmotionContextBuilder, build_song_vectors_from_df

__all__ = ["EmotionContextBuilder", "build_song_vectors_from_df"]
