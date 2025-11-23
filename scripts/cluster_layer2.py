"""
Utility: Cluster Layer 2 vectors using Agglomerative clustering and visualize with UMAP.

Loads the existing FAISS vector store from `vector_db/`, reconstructs Layer 2
embeddings, runs semantic clustering (agglomerative), and saves a UMAP plot
to `output/clusters/`.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from src.faiss_vector_store import FAISSVectorStore
from src.clustering import SemanticClusterer, ClusteringConfig
from src.utils.faiss_utils import reconstruct_all_vectors


def main(vector_db_path: str = "vector_db", algorithm: str = "agglomerative", min_size: int = 5, n_clusters: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("cluster_layer2")

    path = Path(vector_db_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector DB not found at: {path}")

    store = FAISSVectorStore.load(path)
    dim = store.dimensions

    n_layer2 = store.index_layer2.ntotal
    if n_layer2 == 0:
        logger.warning("Layer 2 index is empty; nothing to cluster.")
        return

    logger.info(f"Reconstructing {n_layer2} vectors from Layer 2 (dim={dim})...")
    embeddings = reconstruct_all_vectors(store.index_layer2, dim)

    # Build chunk_id list from layer2 metadata
    chunk_ids = [meta.get("chunk_id", f"layer2_idx_{i}") for i, meta in enumerate(store.metadata_layer2)]
    if len(chunk_ids) != len(embeddings):
        logger.warning(
            f"Metadata count ({len(chunk_ids)}) != embeddings ({len(embeddings)}); trimming to match."
        )
        n = min(len(chunk_ids), len(embeddings))
        chunk_ids = chunk_ids[:n]
        embeddings = embeddings[:n]

    logger.info(f"Running {algorithm.upper()} clustering with UMAP visualization enabled...")
    clustering_config = ClusteringConfig(
        algorithm=algorithm,
        n_clusters=n_clusters,  # auto-detect via silhouette if None
        min_cluster_size=min_size,
        enable_cluster_labels=False,
        enable_visualization=True,
        visualization_output_dir="output/clusters",
    )
    clusterer = SemanticClusterer(clustering_config)

    result = clusterer.cluster_embeddings(embeddings, chunk_ids)

    logger.info(
        f"Done. Clusters: {result.n_clusters}, Noise: {result.noise_count}. "
        f"Plots saved under: {clustering_config.visualization_output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Layer 2 vectors and visualize with UMAP")
    parser.add_argument("--vector-db", default="vector_db", help="Path to vector DB directory")
    parser.add_argument("--algorithm", choices=["agglomerative", "hdbscan"], default="agglomerative")
    parser.add_argument("--min-size", type=int, default=5, help="Min cluster size for HDBSCAN")
    parser.add_argument("--n-clusters", type=int, default=None, help="Agglomerative: number of clusters (auto if omitted)")
    args = parser.parse_args()

    main(args.vector_db, args.algorithm, args.min_size, args.n_clusters)
