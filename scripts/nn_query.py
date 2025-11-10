"""
Nearest-neighbour query against the FAISS vector store.

Embeds an input topic string (e.g., "požární bezpečnost") using the
configured embedding model (.env) and returns the k nearest neighbours
from the selected layer (1=document, 2=section, 3=chunk).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.faiss_vector_store import FAISSVectorStore
from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.summary_generator import SummaryGenerator
from src.config import SummarizationConfig


def search_nn(
    query: str,
    vector_db: Path,
    layer: int = 3,
    k: int = 10,
    document_filter: Optional[str] = None,
    show_section_meta: bool = True,
    include_summary: bool = False,
    summary_chars: int = 150,
):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("nn_query")

    # Load vector store
    store = FAISSVectorStore.load(vector_db)
    dims = store.dimensions
    logger.info(f"Vector store: dim={dims}, L1={store.index_layer1.ntotal}, L2={store.index_layer2.ntotal}, L3={store.index_layer3.ntotal}")

    # Build embedder from env (must match store dims/model)
    embedder = EmbeddingGenerator(EmbeddingConfig.from_env())
    if embedder.dimensions != dims:
        raise ValueError(
            f"Embedder dims ({embedder.dimensions}) != store dims ({dims}). "
            f"Set EMBEDDING_PROVIDER/EMBEDDING_MODEL in .env to match the store."
        )

    # Embed query text
    q = embedder.embed_texts([query])

    # Route to the desired layer
    if layer == 3:
        results = store.search_layer3(q, k=k, document_filter=document_filter)
    elif layer == 2:
        results = store.search_layer2(q, k=k, document_filter=document_filter)
    elif layer == 1:
        results = store.search_layer1(q, k=k)
    else:
        raise ValueError("layer must be 1, 2, or 3")

    # Optional summaries
    summaries = [None] * len(results)
    if include_summary and results:
        # Initialize summarizer with override for max_chars
        sconf = SummarizationConfig.from_env(max_chars=summary_chars)
        summarizer = SummaryGenerator(config=sconf)
        for idx, r in enumerate(results):
            text = r.get("content") or ""
            title = r.get("section_title") or ""
            if text:
                try:
                    summaries[idx] = summarizer.generate_section_summary(text, section_title=title)
                except Exception:
                    summaries[idx] = (text[:summary_chars].strip() + "...") if text else None

    # Pretty print
    print(f"\nTop {k} nearest neighbours (layer {layer}):\n")
    for i, r in enumerate(results, 1):
        title = r.get("section_title") or r.get("document_id") or r.get("chunk_id")
        section_id = r.get("section_id")
        page = r.get("page_number")
        chunk_id = r.get("chunk_id")
        line = (
            f"{i:2d}. score={r['score']:.4f}  doc={r.get('document_id')}  "
            f"section={title}"
        )
        if show_section_meta:
            line += f"  section_id={section_id} page={page}"
        line += f"  id={chunk_id}"
        print(line)
        if include_summary and summaries[i - 1]:
            print(f"    summary: {summaries[i - 1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nearest neighbour query over FAISS store")
    parser.add_argument("--query", required=True, help="Topic text to embed, e.g., 'požární bezpečnost'")
    parser.add_argument("--vector-db", default="vector_db", help="Path to vector DB directory")
    parser.add_argument("--layer", type=int, default=3, choices=[1, 2, 3], help="Which layer to search")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbours to return")
    parser.add_argument("--document-filter", default=None, help="Optional document_id to filter results")
    parser.add_argument("--no-section-meta", action="store_true", help="Do not print section_id/page")
    parser.add_argument("--include-summary", action="store_true", help="Generate and print summaries of neighbours")
    parser.add_argument("--summary-chars", type=int, default=150, help="Max characters for summaries")

    args = parser.parse_args()

    search_nn(
        query=args.query,
        vector_db=Path(args.vector_db),
        layer=args.layer,
        k=args.k,
        document_filter=args.document_filter,
        show_section_meta=not args.no_section_meta,
        include_summary=args.include_summary,
        summary_chars=args.summary_chars,
    )
