"""
Index benchmark documents for evaluation.

Indexes privacy policy documents from benchmark_dataset/privacy_qa/
into benchmark_db/ directory using production configuration (KG disabled).

Usage:
    uv run python scripts/index_benchmark_docs.py
"""

import json
import logging
from pathlib import Path

from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from src.cost_tracker import get_global_tracker, reset_global_tracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Index benchmark documents."""
    # Paths
    source_dir = Path("benchmark_dataset/privacy_qa")
    output_dir = Path("benchmark_db")
    results_dir = Path("benchmark_results")

    logger.info("=" * 80)
    logger.info("INDEXING BENCHMARK DOCUMENTS")
    logger.info("=" * 80)

    # Create directories
    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Check source documents exist
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}\n"
            f"Expected privacy policy documents at {source_dir.absolute()}"
        )

    # Find all text files (privacy policies)
    doc_files = sorted(source_dir.glob("*.txt"))

    if not doc_files:
        raise FileNotFoundError(
            f"No .txt files found in {source_dir}\n" f"Expected privacy policy text files"
        )

    logger.info(f"Found {len(doc_files)} documents to index:")
    for doc_file in doc_files:
        logger.info(f"  - {doc_file.name}")

    # Load production config with KG disabled
    logger.info("\nConfiguration:")
    logger.info("  - Knowledge Graph: DISABLED (per benchmark requirements)")
    logger.info("  - Hybrid Search: ENABLED (production setting)")
    logger.info("  - Reranking: Default from .env")
    logger.info(f"  - Output: {output_dir}")

    # Load config from environment (no overrides to avoid conflicts)
    config = IndexingConfig.from_env()

    # Manually configure for benchmark
    config.enable_knowledge_graph = False  # DISABLE for benchmark (per user request)
    config.enable_hybrid_search = True  # ENABLE for proper evaluation
    config.vector_store_path = output_dir  # Use benchmark_db instead of vector_db

    # Save indexing config for reproducibility
    config_dict = {
        "extraction": {
            "ocr_engine": config.extraction_config.ocr_engine,
            "extract_hierarchy": config.extraction_config.extract_hierarchy,
            "enable_smart_hierarchy": config.extraction_config.enable_smart_hierarchy,
            "hierarchy_tolerance": config.extraction_config.hierarchy_tolerance,
        },
        "summarization": {
            "max_chars": config.summarization_config.max_chars,
            "style": config.summarization_config.style,
            "temperature": config.summarization_config.temperature,
            "max_tokens": config.summarization_config.max_tokens,
        },
        "chunking": {
            "chunk_size": config.chunking_config.chunk_size,
            "chunk_overlap": config.chunking_config.chunk_overlap,
            "enable_contextual": config.chunking_config.enable_contextual,
            "enable_multi_layer": config.chunking_config.enable_multi_layer,
        },
        "embedding": {
            "provider": config.embedding_config.provider,
            "model": config.embedding_config.model,
            "batch_size": config.embedding_config.batch_size,
            "normalize": config.embedding_config.normalize,
        },
        "enable_knowledge_graph": False,  # Disabled for benchmark
        "enable_hybrid_search": True,  # Enabled for benchmark
        "enable_reranking": config.enable_reranking,
        "speed_mode": config.speed_mode,
        "vector_store_path": str(config.vector_store_path),
    }

    config_path = results_dir / "indexing_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"\nIndexing config saved: {config_path}")

    # Reset cost tracker
    reset_global_tracker()

    # Initialize pipeline
    logger.info("\nInitializing indexing pipeline...")
    pipeline = IndexingPipeline(config)

    # Index all documents as a batch (with fixed merging)
    logger.info("\n" + "=" * 80)
    logger.info("INDEXING DOCUMENTS")
    logger.info("=" * 80 + "\n")

    try:
        pipeline.index_batch(
            document_paths=[str(f) for f in doc_files],
            output_dir=output_dir,
            save_per_document=False,  # Save only final combined store
        )
        logger.info(f"✓ Successfully indexed {len(doc_files)} documents")
    except Exception as e:
        logger.error(f"✗ Indexing failed: {e}\n")
        raise

    # Save indexing costs
    tracker = get_global_tracker()
    total_cost = tracker.get_total_cost()

    cost_summary = {
        "total_cost_usd": total_cost,
        "cost_breakdown": tracker.cost_by_provider,
        "total_tokens": tracker.get_total_tokens(),
        "document_count": len(doc_files),
        "cost_per_document_usd": total_cost / len(doc_files) if doc_files else 0,
    }

    cost_path = results_dir / "indexing_cost.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Documents indexed: {len(doc_files)}")
    logger.info(f"Vector store: {output_dir}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost per document: ${cost_summary['cost_per_document_usd']:.4f}")
    logger.info(f"\nConfiguration saved: {config_path}")
    logger.info(f"Cost breakdown saved: {cost_path}")
    logger.info("\nReady for benchmark evaluation!")
    logger.info("  Run: uv run python scripts/run_benchmark.py")


if __name__ == "__main__":
    main()
