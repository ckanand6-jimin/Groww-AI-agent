"""Clustering and embedding pipeline — Phase 3.

Provides ``analyze(reviews, ...)`` as the single entry point that:
    1. Embeds review texts with BAAI/bge-small-en-v1.5.
    2. Reduces dimensions with UMAP.
    3. Clusters with HDBSCAN.
    4. Ranks and returns top-K Cluster objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from pulse.analysis.embeddings import embed_reviews
from pulse.analysis.reduce import reduce_dimensions
from pulse.analysis.cluster import build_clusters, cluster_reviews
from pulse.models.models import Cluster, Review

logger = logging.getLogger(__name__)


def analyze(
    reviews: List[Review],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 5,
    embedding_batch_size: int = 64,
    max_embed_reviews: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
    umap_n_components: int | None = None,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> Tuple[List[Cluster], Dict]:
    """Run the full Phase 3 analysis pipeline.

    Args:
        reviews: PII-scrubbed, deduplicated Review objects.
        model_name: SentenceTransformer model ID.
        top_k: How many top clusters to return.
        embedding_batch_size: Batch size for SentenceTransformer.
        max_embed_reviews: If set and exceeded, stratified-sample.
        umap_n_neighbors, umap_min_dist, umap_n_components: UMAP params.
        hdbscan_min_cluster_size, hdbscan_min_samples: HDBSCAN params.

    Returns:
        clusters: Ranked list of top Cluster objects.
        stats: Dict with keys: reviews_embedded, clusters_found,
            reviews_clustered, embedding_model.
    """
    if not reviews:
        raise ValueError("No reviews provided for analysis.")

    # ---- 1. Embed ----
    embeddings, used_reviews, sample_mask = embed_reviews(
        reviews,
        model_name=model_name,
        batch_size=embedding_batch_size,
        max_reviews=max_embed_reviews,
    )

    # ---- 2. Reduce ----
    reduced = reduce_dimensions(
        embeddings,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=umap_n_components,
        metric="cosine",
    )

    # ---- 3. Cluster ----
    labels = cluster_reviews(
        reduced,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )

    # ---- 4. Build & rank ----
    clusters = build_clusters(
        reduced_embeddings=reduced,
        labels=labels,
        reviews=used_reviews,
        top_k=top_k,
    )

    # Count clustered reviews (excludes noise)
    reviews_clustered = int((labels >= 0).sum())
    clusters_found = int((np.unique(labels) >= 0).sum())

    stats = {
        "reviews_embedded": len(used_reviews),
        "reviews_original": len(reviews),
        "reviews_sampled": len(used_reviews) < len(reviews),
        "clusters_found": clusters_found,
        "reviews_clustered": reviews_clustered,
        "embedding_model": model_name,
    }

    logger.info("Analysis complete: %s", stats)
    return clusters, stats
