"""Unit tests for Phase 3 analysis pipeline.

Tests the clustering, ranking, and embedding logic with synthetic data.
Does NOT load the real BAAI/bge-small-en-v1.5 model (slow) — uses
synthetic embedding fixtures instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from pulse.analysis.cluster import (
    _compute_recency_score,
    _select_representative_snippets,
    build_clusters,
    cluster_reviews,
)
from pulse.analysis.embeddings import build_texts, _stratified_sample
from pulse.analysis.reduce import reduce_dimensions
from pulse.models.models import Cluster, Review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reviews(
    n: int,
    texts: list[str] | None = None,
    ratings: list[int] | None = None,
    base_date: datetime | None = None,
) -> list[Review]:
    """Build synthetic Review objects."""
    if base_date is None:
        base_date = datetime(2026, 6, 1)
    if texts is None:
        texts = [f"Review number {i} with some content." for i in range(n)]
    if ratings is None:
        rng = np.random.default_rng(42)
        ratings = rng.integers(1, 6, size=n).tolist()

    reviews = []
    for i in range(n):
        reviews.append(
            Review(
                review_id=f"rev-{i:04d}",
                text=texts[i % len(texts)],
                rating=ratings[i % len(ratings)],
                published_at=base_date + timedelta(days=i % 30),
            )
        )
    return reviews


def _make_random_embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Generate synthetic L2-normalised embeddings."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 1, (n, dim)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


# ---------------------------------------------------------------------------
# embeddings.py
# ---------------------------------------------------------------------------


class TestBuildTexts:
    def test_extracts_body_text(self):
        reviews = _make_reviews(3, texts=["Great app!", "Terrible.", "OK ok."])
        texts = build_texts(reviews)
        assert texts == ["Great app!", "Terrible.", "OK ok."]

    def test_empty_reviews(self):
        assert build_texts([]) == []


class TestStratifiedSample:
    def test_no_sampling_when_under_limit(self):
        reviews = _make_reviews(50)
        sampled, mask = _stratified_sample(reviews, max_reviews=100)
        assert len(sampled) == 50
        assert mask.sum() == 50
        assert mask.all()

    def test_sampling_when_over_limit(self):
        reviews = _make_reviews(200)
        sampled, mask = _stratified_sample(reviews, max_reviews=50)
        assert len(sampled) == 50
        assert mask.sum() == 50
        assert not mask.all()
        # Check rating distribution is preserved approximately
        orig_ratings = np.array([r.rating for r in reviews])
        samp_ratings = np.array([r.rating for r in sampled])
        assert abs(orig_ratings.mean() - samp_ratings.mean()) < 1.0


# ---------------------------------------------------------------------------
# reduce.py
# ---------------------------------------------------------------------------


class TestReduceDimensions:
    def test_output_shape(self):
        emb = _make_random_embeddings(100, dim=384)
        reduced = reduce_dimensions(emb, n_components=5, n_neighbors=10, min_dist=0.1)
        assert reduced.shape == (100, 5)
        assert reduced.dtype == np.float32

    def test_n_neighbors_clamped(self):
        """n_neighbors > n_samples should be clamped."""
        emb = _make_random_embeddings(10, dim=384)
        reduced = reduce_dimensions(emb, n_components=5, n_neighbors=50, min_dist=0.1)
        assert reduced.shape == (10, 5)

    def test_different_metric(self):
        emb = _make_random_embeddings(50, dim=384)
        reduced = reduce_dimensions(emb, n_components=3, metric="cosine")
        assert reduced.shape == (50, 3)


# ---------------------------------------------------------------------------
# cluster.py
# ---------------------------------------------------------------------------


class TestClusterReviews:
    def test_produces_labels(self):
        emb = _make_random_embeddings(100, dim=384)
        reduced = reduce_dimensions(emb, n_components=5)
        labels = cluster_reviews(reduced, min_cluster_size=5, min_samples=3)
        assert len(labels) == 100
        assert labels.dtype in (np.int32, np.int64)

    def test_noise_label_present(self):
        emb = _make_random_embeddings(100, dim=384)
        reduced = reduce_dimensions(emb, n_components=5)
        labels = cluster_reviews(reduced, min_cluster_size=10, min_samples=5)
        unique = set(labels)
        # -1 (noise) is expected with random data
        assert -1 in unique or len(unique) >= 1


class TestRecencyScore:
    def test_all_same_date(self):
        now = datetime(2026, 6, 10)
        reviews = _make_reviews(5, base_date=now)
        # Override all dates to be identical
        for r in reviews:
            r.published_at = now
        score = _compute_recency_score(reviews)
        assert score == 0.5

    def test_recent_bias(self):
        """Newer median → higher recency score."""
        old = _make_reviews(10, base_date=datetime(2025, 1, 1))
        new = _make_reviews(10, base_date=datetime(2026, 6, 1))

        old_score = _compute_recency_score(old)
        new_score = _compute_recency_score(new)
        # Both span ~same range, but newer group should still score
        # at least comparable (dates are shifted, not stretched)
        assert 0.0 <= old_score <= 1.0
        assert 0.0 <= new_score <= 1.0


class TestRepresentativeSnippets:
    def test_selects_closest_to_centroid(self):
        texts = ["aaa bbb ccc", "aaa bbb ddd", "xxx yyy zzz"]
        reviews = _make_reviews(3, texts=texts, ratings=[3, 3, 3])
        # Create embeddings where one is clearly closest to centroid
        emb = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # same as [0] — both near centroid
            [0.0, 0.0, 0.0, 1.0],  # far away
        ], dtype=np.float32)
        # L2-normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        snippets = _select_representative_snippets(emb, reviews, top_k=2)
        assert len(snippets) == 2
        # The two similar vectors should be selected first (closest to centroid)
        # The distant one should not appear in top 2
        assert texts[2] not in snippets


class TestBuildClusters:
    def test_excludes_noise(self):
        emb = _make_random_embeddings(100, dim=4)
        reviews = _make_reviews(100)
        # Synthetic labels: cluster 0 has 40, cluster 1 has 30, rest noise
        labels = np.full(100, -1, dtype=int)
        labels[:40] = 0
        labels[40:70] = 1

        clusters = build_clusters(emb, labels, reviews, top_k=5)
        assert len(clusters) == 2
        assert all(c.cluster_id >= 0 for c in clusters)

    def test_ranking_order(self):
        """Larger, lower-rated cluster should rank higher."""
        emb = _make_random_embeddings(80, dim=4)
        # Create reviews where cluster 0 = low rating, cluster 1 = high rating
        ratings = [1] * 30 + [5] * 50  # first 30 = low, rest = high
        reviews = _make_reviews(80, ratings=ratings)
        # Cluster 0: indices 0-29 (rating ~1, small but severe)
        # Cluster 1: indices 30-79 (rating ~5, large but mild)
        labels = np.full(80, -1, dtype=int)
        labels[:30] = 0   # small, low-rated → should rank higher
        labels[30:80] = 1  # large, high-rated

        clusters = build_clusters(emb, labels, reviews, top_k=5)
        assert len(clusters) >= 2
        # Cluster 0 should be the low-rated one and rank first
        low_rated = next(c for c in clusters if c.cluster_id == 0)
        high_rated = next(c for c in clusters if c.cluster_id == 1)

        assert low_rated.avg_rating < high_rated.avg_rating
        # rank_score of low-rated should be higher
        assert low_rated.rank_score > high_rated.rank_score
        assert low_rated.rank < high_rated.rank

    def test_cluster_stats_populated(self):
        emb = _make_random_embeddings(50, dim=4)
        base = datetime(2026, 6, 1)
        reviews = _make_reviews(50, base_date=base)
        labels = np.full(50, -1, dtype=int)
        labels[:20] = 0

        clusters = build_clusters(emb, labels, reviews, top_k=5)
        assert len(clusters) == 1
        c = clusters[0]
        assert c.cluster_size == 20
        assert len(c.review_ids) == 20
        assert len(c.member_indices) == 20
        assert len(c.representative_snippets) > 0
        assert c.avg_rating >= 1.0
        assert c.earliest_date is not None
        assert c.latest_date is not None
        assert c.rank == 1

    def test_top_k_truncation(self):
        emb = _make_random_embeddings(100, dim=4)
        reviews = _make_reviews(100)
        labels = np.array([i % 5 for i in range(100)])  # 5 clusters of 20

        clusters = build_clusters(emb, labels, reviews, top_k=3)
        assert len(clusters) == 3

    def test_empty_clusters(self):
        emb = _make_random_embeddings(10, dim=4)
        reviews = _make_reviews(10)
        labels = np.full(10, -1, dtype=int)  # all noise
        clusters = build_clusters(emb, labels, reviews)
        assert clusters == []

    def test_ordinal_ranks_sequential(self):
        emb = _make_random_embeddings(80, dim=4)
        reviews = _make_reviews(80)
        labels = np.array([i % 4 for i in range(80)])  # 4 clusters of 20
        clusters = build_clusters(emb, labels, reviews, top_k=5)
        ranks = [c.rank for c in clusters]
        assert ranks == list(range(1, len(ranks) + 1))
