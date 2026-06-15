"""Embedding client using BAAI/bge-small-en-v1.5 via sentence-transformers.

Produces 384-dimensional embeddings from review body text.
Uses the full dataset by default; stratified sampling only when
PULSE_MAX_EMBED_REVIEWS is set and exceeded.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from pulse.models.models import Review

logger = logging.getLogger(__name__)

# Default model — BAAI/bge-small-en-v1.5 is a compact, high-quality
# English embedding model that runs locally (no API key required).
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# The model outputs 384-d vectors.
EMBEDDING_DIM = 384

# Global model cache to avoid reloading across calls.
_model: SentenceTransformer | None = None


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    from sentence_transformers import SentenceTransformer

    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        dim = _model.get_sentence_embedding_dimension() if hasattr(_model, 'get_sentence_embedding_dimension') else _model.get_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d).", dim)
    return _model


def build_texts(reviews: List[Review]) -> List[str]:
    """Extract embedding-ready text from reviews (body only)."""
    return [r.text for r in reviews]


def _stratified_sample(
    reviews: List[Review], max_reviews: int
) -> Tuple[List[Review], np.ndarray]:
    """Stratified sample by rating to preserve distribution.

    Returns (sampled_reviews, sample_mask) where mask is a boolean array
    indicating which original indices were selected.
    """
    from sklearn.model_selection import train_test_split

    n = len(reviews)
    if n <= max_reviews:
        return reviews, np.ones(n, dtype=bool)

    logger.warning(
        "Review count %d exceeds max_embed_reviews %d — applying stratified sampling.",
        n,
        max_reviews,
    )

    ratings = np.array([r.rating for r in reviews])
    indices = np.arange(n)
    # train_test_split returns (train, test); we want train
    sampled_indices, _ = train_test_split(
        indices, train_size=max_reviews, stratify=ratings, random_state=42
    )
    sampled_indices = np.sort(sampled_indices)
    mask = np.zeros(n, dtype=bool)
    mask[sampled_indices] = True
    sampled = [reviews[i] for i in sampled_indices]

    logger.info("Sampled %d / %d reviews (stratified by rating).", len(sampled), n)
    return sampled, mask


def embed_reviews(
    reviews: List[Review],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    show_progress: bool = True,
    max_reviews: int | None = None,
) -> Tuple[np.ndarray, List[Review], np.ndarray]:
    """Embed review texts and return (embeddings, used_reviews, sample_mask).

    Args:
        reviews: List of Review objects to embed.
        model_name: HuggingFace model ID (default: BAAI/bge-small-en-v1.5).
        batch_size: Batch size for the SentenceTransformer encode call.
        show_progress: Show tqdm progress bar during encoding.
        max_reviews: If set and exceeded, stratified-sample down to this many.
            Defaults to PULSE_MAX_EMBED_REVIEWS env var or unlimited (None).

    Returns:
        embeddings: (n_used, dim) float32 array.
        used_reviews: Reviews that were actually embedded (sampled subset).
        sample_mask: Boolean mask of length len(reviews); True for reviews
            that were included in the output.
    """
    if not reviews:
        raise ValueError("No reviews provided for embedding.")

    # Resolve max_reviews from env if not explicitly passed
    if max_reviews is None:
        env_val = os.environ.get("PULSE_MAX_EMBED_REVIEWS")
        if env_val:
            try:
                max_reviews = int(env_val)
            except ValueError:
                logger.warning("Invalid PULSE_MAX_EMBED_REVIEWS=%s — ignoring.", env_val)

    # Apply sampling if needed
    if max_reviews and len(reviews) > max_reviews:
        used_reviews, sample_mask = _stratified_sample(reviews, max_reviews)
    else:
        used_reviews = reviews
        sample_mask = np.ones(len(reviews), dtype=bool)

    texts = build_texts(used_reviews)
    logger.info("Embedding %d reviews (batch_size=%d) …", len(texts), batch_size)

    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine similarity via dot product
    )

    logger.info("Embeddings produced: shape=%s", embeddings.shape)
    return embeddings, used_reviews, sample_mask
