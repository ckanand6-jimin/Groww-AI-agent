"""HDBSCAN clustering and cluster ranking for review embeddings.

Produces ranked Cluster objects from reduced embeddings, excluding
the noise cluster (-1).  Ranking uses a composite score that rewards
large, low-rated, and recent clusters.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from pulse.models.models import Cluster, Review

logger = logging.getLogger(__name__)

# Default HDBSCAN parameters.
DEFAULT_MIN_CLUSTER_SIZE = 15
DEFAULT_MIN_SAMPLES = 10
# How many representative snippets to select per cluster.
DEFAULT_TOP_SNIPPETS = 5
# Default top-K clusters to return.
DEFAULT_TOP_K = 5
# Random seed for reproducibility.
DEFAULT_RANDOM_STATE = 42
# Clipping: minimum rating severity to avoid zero-weight clusters.
MIN_RATING_SEVERITY = 0.05


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for %s=%s — using default %d.", key, val, default)
        return default


def cluster_reviews(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Run HDBSCAN on reduced embeddings.

    Args:
        reduced_embeddings: (n_samples, n_components) float32 array.
        min_cluster_size: Minimum points to form a cluster.
            Default: PULSE_HDBSCAN_MIN_CLUSTER_SIZE env or 10.
        min_samples: HDBSCAN min_samples parameter.
            Default: PULSE_HDBSCAN_MIN_SAMPLES env or 5.
        random_state: Seed for reproducibility.

    Returns:
        labels: (n_samples,) int array of cluster labels.
            -1 = noise (excluded from output clusters).
    """
    if min_cluster_size is None:
        min_cluster_size = _env_int("PULSE_HDBSCAN_MIN_CLUSTER_SIZE", DEFAULT_MIN_CLUSTER_SIZE)
    if min_samples is None:
        min_samples = _env_int("PULSE_HDBSCAN_MIN_SAMPLES", DEFAULT_MIN_SAMPLES)

    logger.info(
        "HDBSCAN: n_samples=%d, min_cluster_size=%d, min_samples=%d",
        reduced_embeddings.shape[0],
        min_cluster_size,
        min_samples,
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",  # Leaf selection — produces more, smaller clusters
    )

    labels = clusterer.fit_predict(reduced_embeddings)

    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique[unique >= 0])
    n_noise = int(counts[unique == -1].sum()) if -1 in unique else 0

    logger.info(
        "HDBSCAN complete: clusters=%d, noise=%d, labels_distribution=%s",
        n_clusters,
        n_noise,
        dict(zip(unique.tolist(), counts.tolist())),
    )

    return labels


def _compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embeddings (already L2-normalized)."""
    return embeddings.mean(axis=0)


def _select_representative_snippets(
    cluster_embeddings: np.ndarray,
    cluster_reviews_list: List[Review],
    top_k: int = DEFAULT_TOP_SNIPPETS,
) -> List[str]:
    """Return snippets of reviews closest to the cluster centroid."""
    if len(cluster_reviews_list) == 0:
        return []

    centroid = _compute_centroid(cluster_embeddings).reshape(1, -1)
    similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    snippets = []
    for idx in top_indices:
        text = cluster_reviews_list[idx].text
        # Trim to a reasonable snippet length
        if len(text) > 300:
            text = text[:297] + "..."
        snippets.append(text)

    return snippets


def _compute_recency_score(
    reviews: List[Review], reference_date: datetime | None = None
) -> float:
    """Compute a normalised recency score [0, 1] for a group of reviews.

    1.0 = all reviews from the last week; 0.0 = all reviews very old.

    Uses the median review date compared against the overall date range
    of the reviews in this cluster.  If reference_date is provided,
    that anchors the "most recent" end of the scale.
    """
    if not reviews:
        return 0.5

    dates = [r.published_at for r in reviews]
    earliest = min(dates)
    latest = max(dates)

    if earliest == latest:
        return 0.5

    median_date = sorted(dates)[len(dates) // 2]

    # Normalize: 1.0 = median == latest, 0.0 = median == earliest
    total_span = (latest - earliest).total_seconds()
    if total_span == 0:
        return 0.5

    offset = (median_date - earliest).total_seconds()
    return offset / total_span


def build_clusters(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    reviews: List[Review],
    top_k: int = DEFAULT_TOP_K,
    top_snippets: int = DEFAULT_TOP_SNIPPETS,
) -> List[Cluster]:
    """Build ranked Cluster objects from HDBSCAN labels.

    Steps:
        1. Group reviews by label (skip noise label -1).
        2. Compute cluster stats (size, avg_rating, rating_std, date span).
        3. Compute composite rank_score = size_norm * rating_severity * recency.
        4. Sort by rank_score descending, assign ordinal ranks.
        5. Return top_k clusters.

    Args:
        reduced_embeddings: (n_samples, n_components) reduced vectors.
        labels: HDBSCAN cluster labels for each sample.
        reviews: Full review objects (same order as embeddings).
        top_k: Number of top clusters to return.
        top_snippets: Representative snippets per cluster.

    Returns:
        Sorted list of Cluster objects (highest rank first), up to top_k.
    """
    unique_labels = np.unique(labels)
    cluster_labels = [lab for lab in unique_labels if lab >= 0]

    if not cluster_labels:
        logger.warning("HDBSCAN produced no non-noise clusters.")
        return []

    clusters: List[Cluster] = []

    # Precompute normalisation ranges for ranking
    cluster_sizes = []
    for lab in cluster_labels:
        mask = labels == lab
        cluster_sizes.append(int(mask.sum()))

    max_size = max(cluster_sizes) if cluster_sizes else 1

    for lab in cluster_labels:
        mask = labels == lab
        indices = np.where(mask)[0]

        cluster_reviews = [reviews[i] for i in indices]
        cluster_embeddings = reduced_embeddings[indices]

        n = len(cluster_reviews)
        ratings = np.array([r.rating for r in cluster_reviews])

        avg_rating = float(ratings.mean())
        rating_std = float(ratings.std()) if n > 1 else 0.0

        # Date range
        dates = [r.published_at for r in cluster_reviews]
        earliest_date = min(dates)
        latest_date = max(dates)

        # Recency score
        recency = _compute_recency_score(cluster_reviews)

        # Rating severity: lower ratings → higher severity → higher weight
        # Normalize to [0, 1] where 5-star = 0, 1-star = 1
        rating_severity = max((5.0 - avg_rating) / 4.0, MIN_RATING_SEVERITY)

        # Composite rank score
        size_norm = n / max_size if max_size > 0 else 1.0
        rank_score = size_norm * rating_severity * (0.5 + 0.5 * recency)

        # Representative snippets
        snippets = _select_representative_snippets(
            cluster_embeddings, cluster_reviews, top_snippets
        )

        cluster = Cluster(
            cluster_id=int(lab),
            review_ids=[r.review_id for r in cluster_reviews],
            member_indices=indices.tolist(),
            representative_snippets=snippets,
            cluster_size=n,
            avg_rating=round(avg_rating, 2),
            rating_std=round(rating_std, 2),
            earliest_date=earliest_date,
            latest_date=latest_date,
            recency_score=round(recency, 4),
            rank_score=round(rank_score, 4),
            rank=0,  # assigned below
        )
        clusters.append(cluster)

    # Sort descending by rank_score
    clusters.sort(key=lambda c: c.rank_score, reverse=True)

    # Assign ordinal ranks
    for i, c in enumerate(clusters):
        c.rank = i + 1

    # Return top K
    top = clusters[:top_k]

    logger.info(
        "Clusters built: total=%d, top_k=%d, "
        "rank_scores=%s",
        len(clusters),
        len(top),
        [c.rank_score for c in top],
    )

    return top
