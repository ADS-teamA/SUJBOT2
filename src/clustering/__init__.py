"""
Semantic clustering module for vector database chunks.

This module provides semantic clustering capabilities to group chunks
based on their embedding similarity, enabling:
- Topic-based retrieval
- Diversity-aware search
- Cluster-based analytics

Supported algorithms:
- HDBSCAN: Automatic cluster count, noise handling, density-based
- Agglomerative: Hierarchical clustering with cosine distance
- Guided cosine clustering: Spherical k-means, fuzzy c-means, nearest-centroid

All algorithms use cosine distance metrics for consistency with
the embedding space (normalized vectors).
"""

from .semantic_clusterer import (
    SemanticClusterer,
    ClusteringResult,
    ClusterInfo,
)

# Import ClusteringConfig from centralized config
from src.config import ClusteringConfig

__all__ = [
    "SemanticClusterer",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusterInfo",
]
