"""
Run guided clustering on Layer 2 (sections) using seed queries.

Usage examples:
  python -m src.clustering.run_guided_layer2 \
      --vector-store vector_db \
      --algorithm spherical_kmeans \
      --queries "požární bezpečnost" "revize hasicích přístrojů" "evakuační plán"

  python -m src.clustering.run_guided_layer2 \
      --vector-store vector_db \
      --algorithm agglomerative \
      --queries-file data/seed_queries.txt

This clusters Layer 2 section embeddings into K clusters. Provide queries to
seed centroids manually or supply --clusters for automatic initialization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.clustering.semantic_clusterer import SemanticClusterer
from src.clustering import ClusteringConfig
from src.faiss_vector_store import FAISSVectorStore
from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided clustering on Layer 2 (sections)")
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
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters when no seed queries are provided",
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


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    queries = load_queries(args)
    seed_labels: Optional[List[str]] = None

    if queries:
        k = len(queries)
        print(f"Seeds (k={k}):")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        seed_labels = queries
    else:
        if args.clusters is None or args.clusters <= 0:
            raise ValueError("Provide --clusters when no seed queries are supplied.")
        k = args.clusters
        print(f"Auto-initialized clustering (k={k}) without user-provided seeds.")

    # Load FAISS vector store
    vs_path = Path(args.vector_store)
    store = FAISSVectorStore.load(vs_path)

    # Load Layer 2 embeddings + metadata
    l2_embeddings, l2_metadata = store.get_layer_embeddings_and_metadata(layer=2)
    if l2_embeddings.size == 0:
        raise RuntimeError("Layer 2 is empty in the provided vector store")

    chunk_ids = [m.get("chunk_id", str(i)) for i, m in enumerate(l2_metadata)]
    chunk_id_to_meta = {m.get("chunk_id", str(i)): m for i, m in enumerate(l2_metadata)}
    chunk_id_to_index = {cid: idx for idx, cid in enumerate(chunk_ids)}

    seed_centroids = None
    if queries:
        embedder = EmbeddingGenerator(EmbeddingConfig())
        if embedder.dimensions != store.dimensions:
            raise RuntimeError(
                f"Embedding dim mismatch: embedder={embedder.dimensions}, store={store.dimensions}. "
                "Use the same embedding model as used for indexing."
            )
        seed_centroids = embedder.embed_texts(queries)

    # Build clusterer and run guided clustering
    cconfig = ClusteringConfig(
        algorithm="agglomerative",  # placeholder; guided method overrides selection
        enable_visualization=args.viz,
    )
    clusterer = SemanticClusterer(cconfig)
    result = clusterer.guided_cluster(
        embeddings=l2_embeddings,
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

    # Print summary
    print("\nCluster summary:")
    for cid in sorted(result.cluster_info.keys()):
        info = result.cluster_info[cid]
        label = info.label or f"Cluster {cid}"
        if memberships is not None and cid < memberships.shape[1]:
            member_indices = [chunk_id_to_index[c] for c in result.get_cluster_chunks(cid)]
            avg_conf = float(memberships[member_indices, cid].mean()) if member_indices else 0.0
            print(f"- [{cid}] {label}: {info.size} sections (avg={avg_conf:.2f})")
        else:
            print(f"- [{cid}] {label}: {info.size} sections")
        # Show up to 3 representative section titles
        reps = result.get_cluster_chunks(cid)[:3]
        titles = []
        for rid in reps:
            meta = chunk_id_to_meta.get(rid, {})
            t = meta.get("section_title") or meta.get("section_path") or rid
            titles.append(t)
        if titles:
            print(f"    e.g., {titles}")

    # Save JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": args.algorithm,
            "k": k,
            "queries": queries if queries else None,
            "n_sections": int(l2_embeddings.shape[0]),
            "quality_metrics": result.quality_metrics,
            "clusters": {
                int(cid): {
                    "label": result.cluster_info[cid].label,
                    "size": result.cluster_info[cid].size,
                    "sections": [
                        {
                            "chunk_id": chunk_id,
                            "section_title": chunk_id_to_meta.get(chunk_id, {}).get("section_title"),
                            "section_path": chunk_id_to_meta.get(chunk_id, {}).get("section_path"),
                            "page_number": chunk_id_to_meta.get(chunk_id, {}).get("page_number"),
                            "summary": chunk_id_to_meta.get(chunk_id, {}).get("content"),
                        }
                        for chunk_id in result.get_cluster_chunks(cid)
                    ],
                    "avg_confidence": (
                        float(
                            memberships[
                                [chunk_id_to_index[c] for c in result.get_cluster_chunks(cid)],
                                cid,
                            ].mean()
                        )
                        if (
                            memberships is not None
                            and cid < memberships.shape[1]
                            and result.get_cluster_chunks(cid)
                        )
                        else None
                    ),
                }
                for cid in sorted(result.cluster_info.keys())
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved assignments: {out_path}")


if __name__ == "__main__":
    main()
