"""UMAP dimensionality reduction for review embeddings.

Reduces high-dimensional embeddings to a lower-dimensional space
suitable for density-based clustering (HDBSCAN).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sensible defaults for review clustering.
DEFAULT_N_NEIGHBORS = 10
DEFAULT_MIN_DIST = 0.05
DEFAULT_N_COMPONENTS = 5
DEFAULT_RANDOM_STATE = 42


def reduce_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    n_components: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embedding dimensions with UMAP.

    Args:
        embeddings: (n_samples, original_dim) float32 array.
        n_neighbors: UMAP n_neighbors parameter. Controls local vs global
            structure. Default: PULSE_UMAP_N_NEIGHBORS env or 15.
        min_dist: UMAP min_dist parameter. Controls how tightly points
            are packed. Default: PULSE_UMAP_MIN_DIST env or 0.1.
        n_components: Target dimension. Default: PULSE_UMAP_N_COMPONENTS
            env or 5.
        random_state: Seed for reproducibility.
        metric: Distance metric for UMAP (default: "cosine").

    Returns:
        reduced: (n_samples, n_components) float32 array.
    """
    import umap

    n_samples = embeddings.shape[0]

    # Resolve parameters with env-var overrides
    if n_neighbors is None:
        n_neighbors = _env_int("PULSE_UMAP_N_NEIGHBORS", DEFAULT_N_NEIGHBORS)
    if min_dist is None:
        min_dist = _env_float("PULSE_UMAP_MIN_DIST", DEFAULT_MIN_DIST)
    if n_components is None:
        n_components = _env_int("PULSE_UMAP_N_COMPONENTS", DEFAULT_N_COMPONENTS)

    # Clamp n_neighbors to number of samples
    n_neighbors = min(n_neighbors, n_samples - 1)
    n_neighbors = max(n_neighbors, 2)

    logger.info(
        "UMAP: n_samples=%d, n_components=%d, n_neighbors=%d, min_dist=%.3f, metric=%s",
        n_samples,
        n_components,
        n_neighbors,
        min_dist,
        metric,
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        low_memory=False,
    )

    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete: shape=%s", reduced.shape)

    return reduced


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for %s=%s — using default %d.", key, val, default)
        return default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float for %s=%s — using default %.3f.", key, val, default)
        return default
