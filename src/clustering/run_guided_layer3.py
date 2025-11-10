"""
Run guided clustering on Layer 3 (chunk-level) embeddings using seed queries.

Usage example:
  python -m src.clustering.run_guided_layer3 \
      --vector-store vector_db \
      --algorithm spherical_kmeans \
      --queries "fire safety" "evacuation plan" \
      --output output/clusters/guided_l3.json

This clusters Layer 3 chunk embeddings into K clusters. Provide queries to
seed centroids manually or supply --clusters for automatic initialization.
Outputs JSON with per-chunk summaries (falling back to truncated content).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.clustering import ClusteringConfig, SemanticClusterer
from src.embedding_generator import EmbeddingConfig, EmbeddingGenerator
from src.faiss_vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided clustering on Layer 3 (chunks)")
    parser.add_argument(
        "--vector-store",
        type=str,
        default=str(Path("vector_db")),
        help="Path to vector store directory (default: vector_db)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["spherical_kmeans", "agglomerative", "fuzzy_cmeans"],
        default="spherical_kmeans",
        help="Guided clustering algorithm",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Seed queries (one or more). Each query becomes a centroid.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to a text file with one query per line",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters when no seed queries are provided",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Max iterations for iterative algorithms",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance (centroid L2 delta)",
    )
    parser.add_argument(
        "--fuzziness",
        type=float,
        default=2.0,
        help="Fuzziness coefficient m (>1) for fuzzy c-means",
    )
    parser.add_argument(
        "--init-method",
        type=str,
        choices=["kmeans++", "random"],
        default="kmeans++",
        help="Centroid initialization method when auto-initializing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for assignments/summary",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable UMAP visualization (saved under output/clusters)",
    )
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=220,
        help="Maximum characters for generated chunk summaries (default: 220)",
    )
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []
    if args.queries:
        queries.extend(args.queries)
    if args.queries_file:
        path = Path(args.queries_file)
        if not path.exists():
            raise FileNotFoundError(f"Queries file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    queries.append(q)
    return queries


def summarize_chunk(meta: dict, max_chars: int) -> Optional[str]:
    """
    Derive a concise summary for a chunk.

    Preference order:
      1. Explicit chunk_summary metadata
      2. Pre-existing summary field
      3. Normalized/truncated primary content
    """
    for key in ("chunk_summary", "summary"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    content = meta.get("content")
    if not isinstance(content, str):
        return None

    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    queries = load_queries(args)
    seed_labels: Optional[List[str]] = None

    if queries:
        k = len(queries)
        print(f"Seeds (k={k}):")
        for idx, query in enumerate(queries, 1):
            print(f"  {idx}. {query}")
        seed_labels = queries
    else:
        if args.clusters is None or args.clusters <= 0:
            raise ValueError("Provide --clusters when no seed queries are supplied.")
        k = args.clusters
        print(f"Auto-initialized clustering (k={k}) without user-provided seeds.")

    # Load FAISS vector store
    store_path = Path(args.vector_store)
    store = FAISSVectorStore.load(store_path)

    # Load Layer 3 embeddings and metadata
    embeddings, metadata = store.get_layer_embeddings_and_metadata(layer=3)
    if embeddings.size == 0:
        raise RuntimeError("Layer 3 is empty in the provided vector store.")

    chunk_ids = [m.get("chunk_id", str(idx)) for idx, m in enumerate(metadata)]
    chunk_id_to_meta = {cid: meta for cid, meta in zip(chunk_ids, metadata)}
    chunk_id_to_index = {cid: idx for idx, cid in enumerate(chunk_ids)}

    seed_centroids = None
    if queries:
        embedder = EmbeddingGenerator(EmbeddingConfig())
        if embedder.dimensions != store.dimensions:
            raise RuntimeError(
                f"Embedding dim mismatch: embedder={embedder.dimensions}, "
                f"store={store.dimensions}. Ensure identical embedding models."
            )
        seed_centroids = embedder.embed_texts(queries)

    clustering_config = ClusteringConfig(
        algorithm="agglomerative",  # placeholder; guided mode overrides algorithm
        enable_visualization=args.viz,
        visualization_output_dir="output/clusters",
    )
    clusterer = SemanticClusterer(clustering_config)
    result = clusterer.guided_cluster(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        seed_centroids=seed_centroids,
        seed_labels=seed_labels,
        algorithm=args.algorithm,
        max_iter=args.max_iter,
        tol=args.tol,
        n_clusters=k,
        fuzziness=args.fuzziness,
        init_method=args.init_method,
    )

    memberships = result.memberships

    print("\nCluster summary:")
    for cid in sorted(result.cluster_info):
        info = result.cluster_info[cid]
        label = info.label or f"Cluster {cid}"
        if memberships is not None and cid < memberships.shape[1]:
            member_indices = [chunk_id_to_index[ch] for ch in result.get_cluster_chunks(cid)]
            avg_conf = float(memberships[member_indices, cid].mean()) if member_indices else 0.0
            print(f"- [{cid}] {label}: {info.size} chunks (avg={avg_conf:.2f})")
        else:
            print(f"- [{cid}] {label}: {info.size} chunks")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "algorithm": args.algorithm,
            "k": k,
            "queries": queries if queries else None,
            "n_chunks": int(embeddings.shape[0]),
            "quality_metrics": result.quality_metrics,
            "clusters": {},
        }

        for cid in sorted(result.cluster_info):
            cluster_chunks = result.get_cluster_chunks(cid)
            avg_conf = None
            if memberships is not None and cid < memberships.shape[1] and cluster_chunks:
                indices = [chunk_id_to_index[ch] for ch in cluster_chunks]
                avg_conf = float(memberships[indices, cid].mean())

            payload["clusters"][int(cid)] = {
                "label": result.cluster_info[cid].label,
                "size": result.cluster_info[cid].size,
                "avg_confidence": avg_conf,
                "chunks": [
                    {
                        "chunk_id": chunk_id,
                        "document_id": chunk_id_to_meta.get(chunk_id, {}).get("document_id"),
                        "section_id": chunk_id_to_meta.get(chunk_id, {}).get("section_id"),
                        "section_title": chunk_id_to_meta.get(chunk_id, {}).get("section_title"),
                        "section_path": chunk_id_to_meta.get(chunk_id, {}).get("section_path"),
                        "page_number": chunk_id_to_meta.get(chunk_id, {}).get("page_number"),
                        "summary": summarize_chunk(chunk_id_to_meta.get(chunk_id, {}), args.summary_max_chars),
                    }
                    for chunk_id in cluster_chunks
                ],
            }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        print(f"\nSaved assignments: {output_path}")


if __name__ == "__main__":
    main()
