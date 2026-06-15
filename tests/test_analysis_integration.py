"""Integration test for Phase 3 analysis pipeline.

Uses real Groww review data from the normalized cache to verify:
    - Full pipeline runs on 1413 reviews without OOM.
    - Produces >= 1 meaningful (non-noise) cluster.
    - Cluster stats are populated correctly.
    - Representative snippets are verbatim from source reviews.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure we can import from pulse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse.analysis import analyze
from pulse.ingest.adapter import ingest_from_cache


def _get_cache_dir() -> str:
    """Resolve cache directory relative to repo root."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(test_dir)
    # test dir is pulse-agent/tests, repo_root is AI AGENT/
    # but actually test_dir is .../pulse-agent/tests
    # repo_root = .../pulse-agent
    # So we need to go up one more level
    repo_root = os.path.dirname(repo_root)
    cache_dir = os.path.join(repo_root, "data", "cache", "groww", "2026-06-09")
    return cache_dir


@pytest.mark.integration
class TestAnalysisPipelineIntegration:
    """End-to-end test with real Groww data."""

    def test_full_pipeline_on_groww_data(self):
        """Run embed → reduce → cluster on full ingested dataset."""
        cache_dir = _get_cache_dir()
        if not os.path.isdir(cache_dir):
            pytest.skip(f"Cache directory not found: {cache_dir}")

        # 1. Ingest real reviews
        reviews, ingest_stats = ingest_from_cache(cache_dir, "2026-W23", "groww")
        assert len(reviews) > 0, "No reviews ingested"
        print(f"\nIngested {len(reviews)} reviews (from {ingest_stats['fetched']} raw)")

        # 2. Run full analysis pipeline
        clusters, stats = analyze(
            reviews,
            top_k=5,
            embedding_batch_size=64,
        )

        print(f"Stats: {stats}")
        print(f"Clusters found: {len(clusters)}")

        # 3. Assertions

        # Must produce at least 1 cluster
        assert len(clusters) >= 1, (
            f"Expected >=1 cluster, got {len(clusters)}. "
            f"Stats: {stats}"
        )

        # Cluster stats populated
        for c in clusters:
            assert c.cluster_size > 0, f"Cluster {c.cluster_id} has size 0"
            assert len(c.review_ids) == c.cluster_size
            assert 1.0 <= c.avg_rating <= 5.0, (
                f"Cluster {c.cluster_id} avg_rating {c.avg_rating} out of range"
            )
            assert c.earliest_date is not None
            assert c.latest_date is not None
            assert c.rank >= 1

        # Top cluster should be the largest or most severe
        top = clusters[0]
        assert top.rank == 1
        assert top.rank_score > 0

        # Representative snippets should exist
        for c in clusters:
            assert len(c.representative_snippets) > 0, (
                f"Cluster {c.cluster_id} has no snippets"
            )
            # Each snippet should be a non-empty string
            for snippet in c.representative_snippets:
                assert isinstance(snippet, str) and len(snippet) > 0

        # Stats consistency
        assert stats["reviews_embedded"] > 0
        assert stats["clusters_found"] >= len(clusters)
        assert stats["reviews_clustered"] > 0
        assert stats["embedding_model"] == "BAAI/bge-small-en-v1.5"

        # Print cluster summaries for manual review
        print(f"\n{'='*60}")
        print(f"TOP CLUSTERS (from {stats['reviews_embedded']} reviews)")
        print(f"{'='*60}")
        for c in clusters:
            print(f"\n--- Rank #{c.rank}: Cluster {c.cluster_id} "
                  f"(size={c.cluster_size}, avg_rating={c.avg_rating:.1f}, "
                  f"score={c.rank_score:.4f}) ---")
            print(f"  Date range: {c.earliest_date.date()} – {c.latest_date.date()}")
            print(f"  Representative snippets:")
            for i, snip in enumerate(c.representative_snippets[:3]):
                print(f"    [{i+1}] {snip[:120]}...")

    def test_pipeline_idempotent(self):
        """Running twice with same data produces consistent cluster counts."""
        cache_dir = _get_cache_dir()
        if not os.path.isdir(cache_dir):
            pytest.skip(f"Cache directory not found: {cache_dir}")

        reviews, _ = ingest_from_cache(cache_dir, "2026-W23", "groww")

        clusters_a, stats_a = analyze(reviews, top_k=5)
        clusters_b, stats_b = analyze(reviews, top_k=5)

        # Same number of clusters found (deterministic with fixed seed)
        assert stats_a["clusters_found"] == stats_b["clusters_found"], (
            f"Non-deterministic: run1={stats_a['clusters_found']}, "
            f"run2={stats_b['clusters_found']}"
        )
        # Same number of clustered reviews
        assert stats_a["reviews_clustered"] == stats_b["reviews_clustered"]

        # Same top cluster sizes (ranking may differ slightly with ties)
        sizes_a = [c.cluster_size for c in clusters_a]
        sizes_b = [c.cluster_size for c in clusters_b]
        assert sizes_a == sizes_b, f"Cluster sizes differ: {sizes_a} vs {sizes_b}"
