"""
Nearest-neighbour visualization for FAISS vector store.

Embeds a topic string (e.g., "požární bezpečnost"), retrieves k nearest
neighbours from a chosen layer, and visualizes the embedding space with UMAP,
highlighting the query and neighbours.
"""

import logging
from pathlib import Path
from typing import List, Optional
import time

import numpy as np

from src.faiss_vector_store import FAISSVectorStore
from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.summary_generator import SummaryGenerator
from src.config import SummarizationConfig


def _reconstruct_all_vectors(index, dim: int) -> np.ndarray:
    """Reconstruct all vectors from a FAISS IndexFlatIP (float32)."""
    import faiss

    n = index.ntotal
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)

    if hasattr(index, "reconstruct_n"):
        try:
            return index.reconstruct_n(0, n)
        except Exception:
            pass

    vecs: List[np.ndarray] = []
    for i in range(n):
        v = index.reconstruct(i)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        vecs.append(v.astype(np.float32, copy=False))
    return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)


def run_nn_viz(
    query: str,
    vector_db: Path,
    layer: int = 2,
    k: int = 10,
    document_filter: Optional[str] = None,
    out_dir: Path = Path("output/nn_viz"),
    include_summary: bool = False,
    summary_chars: int = 150,
) -> Path:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("nn_viz")

    store = FAISSVectorStore.load(vector_db)
    dims = store.dimensions
    logger.info(
        f"Vector store loaded: dim={dims}, L1={store.index_layer1.ntotal}, "
        f"L2={store.index_layer2.ntotal}, L3={store.index_layer3.ntotal}"
    )

    # Reconstruct embeddings and pick metadata per layer
    if layer == 1:
        index = store.index_layer1
        metadata = store.metadata_layer1
    elif layer == 2:
        index = store.index_layer2
        metadata = store.metadata_layer2
    elif layer == 3:
        index = store.index_layer3
        metadata = store.metadata_layer3
    else:
        raise ValueError("layer must be 1, 2, or 3")

    if index.ntotal == 0:
        raise RuntimeError(f"Layer {layer} index is empty")

    logger.info(f"Reconstructing {index.ntotal} vectors from layer {layer}...")
    emb = _reconstruct_all_vectors(index, dims)

    # Embed the query
    embedder = EmbeddingGenerator(EmbeddingConfig.from_env())
    if embedder.dimensions != dims:
        raise ValueError(
            f"Embedder dims ({embedder.dimensions}) != store dims ({dims}). "
            "Ensure .env embedding model matches the store."
        )
    q = embedder.embed_texts([query])  # (1, D), normalized

    # Retrieve nearest neighbours via the vector store API (adds indices in results)
    if layer == 3:
        results = store.search_layer3(q, k=k, document_filter=document_filter)
    elif layer == 2:
        results = store.search_layer2(q, k=k, document_filter=document_filter)
    else:
        results = store.search_layer1(q, k=k)

    if not results:
        raise RuntimeError("No neighbours returned (check filters or store contents)")

    nn_indices = [r["index"] for r in results if "index" in r]
    if not nn_indices:
        raise RuntimeError("Search results missing indices; cannot highlight neighbours")

    # Build UMAP projection including the query as an extra point
    import umap
    import matplotlib.pyplot as plt

    data = np.vstack([emb, q.astype(np.float32)])  # Query last row
    logger.info("Running UMAP dimensionality reduction (cosine metric)...")
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(data)

    pts = proj[:-1]
    qpt = proj[-1]

    # Prepare colors/masks
    mask_nn = np.zeros(len(emb), dtype=bool)
    mask_nn[nn_indices] = True

    # Plot
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"nn_layer{layer}_{ts}.png"

    fig, ax = plt.subplots(figsize=(12, 9))
    # Background points
    ax.scatter(pts[~mask_nn, 0], pts[~mask_nn, 1], c="lightgray", s=20, alpha=0.4, label="Other points", edgecolors="none")
    # Nearest neighbours
    ax.scatter(pts[mask_nn, 0], pts[mask_nn, 1], c="tab:red", s=50, alpha=0.8, label=f"Top-{k} neighbours", edgecolors="black", linewidths=0.3)
    # Query point
    ax.scatter([qpt[0]], [qpt[1]], c="tab:blue", s=120, marker="*", label="Query", edgecolors="black", linewidths=0.6)

    ax.set_title(f"Nearest Neighbours (Layer {layer})\nQuery: {query}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Optional summaries
    summaries = [None] * len(results)
    if include_summary and results:
        sconf = SummarizationConfig.from_env(max_chars=summary_chars)
        summarizer = SummaryGenerator(config=sconf)
        for i, r in enumerate(results):
            text = r.get("content") or ""
            title = r.get("section_title") or ""
            if text:
                try:
                    summaries[i] = summarizer.generate_section_summary(text, section_title=title)
                except Exception:
                    summaries[i] = (text[:summary_chars].strip() + "...") if text else None

    # Print textual results
    print(f"\nTop {k} neighbours (layer {layer}):\n")
    for i, r in enumerate(results, 1):
        title = r.get("section_title") or r.get("document_id") or r.get("chunk_id")
        section_id = r.get("section_id")
        page = r.get("page_number")
        print(
            f"{i:2d}. score={r['score']:.4f}  doc={r.get('document_id')}  section={title} "
            f"section_id={section_id} page={page} id={r.get('chunk_id')} index={r.get('index')}"
        )
        if include_summary and summaries[i - 1]:
            print(f"    summary: {summaries[i - 1]}")

    logging.info(f"Visualization saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nearest neighbour visualization over FAISS store")
    parser.add_argument("--query", required=True, help="Topic text to embed, e.g., 'požární bezpečnost'")
    parser.add_argument("--vector-db", default="vector_db", help="Path to vector DB directory")
    parser.add_argument("--layer", type=int, default=2, choices=[1, 2, 3], help="Which layer to search/visualize")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbours to return & highlight")
    parser.add_argument("--document-filter", default=None, help="Optional document_id to filter results")
    parser.add_argument("--out-dir", default="output/nn_viz", help="Output directory for the plot")
    parser.add_argument("--include-summary", action="store_true", help="Generate and print summaries of neighbours")
    parser.add_argument("--summary-chars", type=int, default=150, help="Max characters for summaries")

    args = parser.parse_args()
    run_nn_viz(
        query=args.query,
        vector_db=Path(args.vector_db),
        layer=args.layer,
        k=args.k,
        document_filter=args.document_filter,
        out_dir=Path(args.out_dir),
        include_summary=args.include_summary,
        summary_chars=args.summary_chars,
    )
