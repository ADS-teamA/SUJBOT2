#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner - FULL SOTA 2025 SYSTEM

Runs complete indexing pipeline with ALL advanced features:
- PHASE 1: Smart hierarchy extraction (Docling)
- PHASE 2: Generic summaries
- PHASE 3: Multi-layer chunking + SAC (contextual retrieval)
- PHASE 4: Embeddings + FAISS indexing
- PHASE 5A: Knowledge Graph extraction ✅
- PHASE 5B: Hybrid Search (BM25 + Dense + RRF) ✅

Supported formats: PDF, DOCX, PPTX, XLSX, HTML
Input: Single document OR directory (batch processing)

Usage:
    # Single document
    python run_pipeline.py <document_path>
    python run_pipeline.py "data/document.pdf"

    # Batch processing (directory)
    python run_pipeline.py <directory_path>
    python run_pipeline.py "data/documents/"

Outputs saved to: output/<document_name>/<timestamp>/
- phase1_extraction.json - Document structure & hierarchy
- phase2_summaries.json - Generic summaries
- phase3_chunks.json - Multi-layer chunks with SAC
- phase4_vector_store/ - FAISS + BM25 indexes + metadata
- <document_id>_kg.json - Knowledge graph (if enabled)

Configuration: All settings controlled via config.json file
- Copy config.json.example to config.json and fill in your values
- See config.json.example for available options
- Hybrid Search: "hybrid_search": {"enable": true}
- Knowledge Graph: "knowledge_graph": {"enable": true}
"""

import sys
import logging
import argparse
import shutil
from pathlib import Path

# CRITICAL: Validate config.json before doing anything else
try:
    from src.config import get_config
    _config = get_config()  # This will fail if config.json is invalid or missing
except FileNotFoundError as e:
    print(f"\n❌ ERROR: config.json not found!")
    print(f"\nPlease create config.json from config.json.example:")
    print(f"  cp config.json.example config.json")
    print(f"  # Edit config.json with your settings")
    sys.exit(1)
except ValueError as e:
    print(f"\n❌ ERROR: Invalid configuration in config.json!")
    print(f"\n{e}")
    print(f"\nPlease fix the errors in config.json")
    print(f"See config.json.example for reference")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: Failed to load configuration!")
    print(f"\n{e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from src.cost_tracker import get_global_tracker, reset_global_tracker


def print_header(text: str):
    """Print formatted header."""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
    print()


def print_success(text: str):
    """Print success message."""
    print(f"[OK]{text}")


def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


def get_supported_documents(directory: Path) -> list:
    """
    Get list of all supported documents in directory.

    Args:
        directory: Path to directory

    Returns:
        List of document paths
    """
    supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt", ".tex", ".latex"]
    documents = []

    for ext in supported_formats:
        documents.extend(directory.glob(f"*{ext}"))
        documents.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    documents = sorted(set(documents))

    return documents


def run_complete_pipeline(input_path: Path, output_base: Path = None, merge_target: Path = None, storage_backend: str = None):
    """
    Run complete SOTA 2025 RAG pipeline.

    Features:
    - Smart hierarchy extraction
    - Generic summaries
    - Contextual retrieval (SAC)
    - Hybrid search (BM25 + Dense + RRF)
    - Knowledge graph extraction
    - Automatic merge with existing vector store

    Args:
        input_path: Path to document file or directory
        output_base: Base output directory (default: output/)
        merge_target: Path to existing vector store to merge into (e.g., vector_db/)
        storage_backend: Storage backend override ('faiss' or 'postgresql', default: from config.json)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"✗ Error: Path not found: {input_path}")
        sys.exit(1)

    # Check if it's a directory
    if input_path.is_dir():
        documents = get_supported_documents(input_path)
        if not documents:
            print(f"✗ Error: No supported documents found in directory: {input_path}")
            print(f"  Supported formats: PDF, DOCX, PPTX, XLSX, HTML")
            sys.exit(1)

        print_header(f"BATCH PROCESSING - {len(documents)} documents from {input_path.name}")
        print_info(f"Found documents:")
        for doc in documents:
            print_info(f"  - {doc.name}")
        print()

        # Process each document
        for i, document_path in enumerate(documents, 1):
            print()
            print("=" * 80)
            print(f"PROCESSING [{i}/{len(documents)}]: {document_path.name}")
            print("=" * 80)
            print()
            run_single_document(document_path, output_base, merge_target, storage_backend)

        print_header("BATCH PROCESSING COMPLETE")
        print_success(f"Processed {len(documents)} documents")
        print_info(f"Outputs saved to: {output_base or Path('output')}")
        return

    # Single document processing
    run_single_document(input_path, output_base, merge_target, storage_backend)


def run_single_document(document_path: Path, output_base: Path = None, merge_target: Path = None, storage_backend: str = None):
    """
    Process single document through complete SOTA 2025 pipeline.

    Args:
        document_path: Path to document file
        output_base: Base output directory (default: output/)
        merge_target: Path to existing vector store to merge into (e.g., vector_db/)
        storage_backend: Storage backend override ('faiss' or 'postgresql', default: from config.json)
    """
    document_path = Path(document_path)

    # Validate format
    supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt", ".tex", ".latex"]
    if document_path.suffix.lower() not in supported_formats:
        print(f"✗ Error: Unsupported format: {document_path.suffix}")
        print(f"  Supported formats: {', '.join(supported_formats)}")
        return

    # Setup output directory
    if output_base is None:
        output_base = Path(__file__).parent / "output"

    doc_name = document_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = output_base / doc_name  # No timestamp - enables resume functionality
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"SOTA 2025 RAG PIPELINE - {document_path.name}")

    # Reset cost tracker for this document
    reset_global_tracker()

    # Create pipeline with FULL SOTA configuration from .env
    print_info("Initializing pipeline with SOTA 2025 configuration...")

    # Create config with optional storage backend override
    config_overrides = {}
    if storage_backend is not None:
        config_overrides["storage_backend"] = storage_backend

    config = IndexingConfig.from_env(**config_overrides)  # Loads all settings from config.json

    # Print active configuration
    print_info(f"LLM Model: {config.summarization_config.model}")
    print_info(f"Embedding Model: {config.embedding_config.model}")
    print_info(f"Max Tokens: {config.chunking_config.max_tokens} tokens (HybridChunker)")
    print_info(f"SAC (Contextual Retrieval): {'ON' if config.chunking_config.enable_contextual else 'OFF'}")
    print_info(f"Hybrid Search (BM25+Dense): {'ON ✅' if config.enable_hybrid_search else 'OFF'}")
    print_info(f"Knowledge Graph: {'ON ✅' if config.enable_knowledge_graph else 'OFF'}")
    print_info(f"Storage Backend: {config.storage_backend.upper()}")
    print_info(f"Output: {output_dir}")
    print()

    pipeline = IndexingPipeline(config)

    # Run pipeline with intermediate saves
    try:
        print_header("RUNNING PIPELINE (ALL PHASES)")

        result = pipeline.index_document(
            document_path=document_path,
            save_intermediate=True,  # Save phase1-3 JSON outputs
            output_dir=output_dir
        )

        # Check if indexing was skipped due to duplicate detection
        if result is None:
            print_header("DOCUMENT SKIPPED")
            print_info("Document was identified as duplicate and skipped")
            print_info("No indexing or merging performed")
            return

        vector_store = result["vector_store"]
        knowledge_graph = result["knowledge_graph"]
        stats = result["stats"]

        # Save PHASE 4: Vector store (FAISS + BM25 if hybrid)
        print_info("Saving vector store...")
        vs_path = output_dir / "phase4_vector_store"
        vector_store.save(vs_path)
        print_success(f"Vector store saved: {vs_path}")

        # Save PHASE 5A: Knowledge Graph (if enabled)
        kg_path = None
        if knowledge_graph:
            kg_path = output_dir / f"{doc_name}_kg.json"
            knowledge_graph.save_json(str(kg_path))
            print_success(f"Knowledge graph saved: {kg_path}")

        # MERGE with existing vector store (if --merge flag provided)
        if merge_target:
            merge_target = Path(merge_target)

            # Try to load existing vector store, or initialize new one if empty/corrupt
            from src.hybrid_search import HybridVectorStore
            from src.faiss_vector_store import FAISSVectorStore

            # Check if we can load existing store
            can_load_existing = False
            if merge_target.exists():
                # Check if directory has ALL required vector store files
                required_files = [
                    "faiss_metadata.json",
                    "faiss_arrays.pkl",
                    "faiss_layer1.index",
                    "faiss_layer2.index",
                    "faiss_layer3.index",
                    "bm25_layer1_arrays.pkl",
                    "bm25_layer1_config.json",
                    "bm25_layer2_arrays.pkl",
                    "bm25_layer2_config.json",
                    "bm25_layer3_arrays.pkl",
                    "bm25_layer3_config.json",
                    "hybrid_config.json",
                ]
                can_load_existing = all((merge_target / f).exists() for f in required_files)

            if not can_load_existing:
                # Initialize new vector store (directory doesn't exist or is empty/incomplete)
                print()
                print_info(f"Initializing new vector store at: {merge_target}")
                merge_target.mkdir(parents=True, exist_ok=True)
                # Copy new store to merge target
                for item in vs_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, merge_target / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, merge_target / item.name, dirs_exist_ok=True)
                print_success(f"Initialized vector store at: {merge_target}")
            else:
                # Load and merge with existing store
                print()
                print_header("MERGING WITH EXISTING VECTOR STORE")
                print_info(f"Target: {merge_target}")

                try:
                    print_info("Loading existing vector store...")
                    existing_store = HybridVectorStore.load(merge_target)
                    existing_stats_before = existing_store.get_stats()

                    print_info(f"Existing store: {existing_stats_before['total_vectors']} vectors, "
                              f"{existing_stats_before['documents']} documents")
                    print_info(f"New store: {stats['vector_store']['total_vectors']} vectors, "
                              f"{stats['vector_store']['documents']} documents")

                    # Merge vector stores with deduplication
                    print_info("Merging FAISS indexes with deduplication...")
                    faiss_merge_stats = existing_store.faiss_store.merge(
                        vector_store if isinstance(vector_store, FAISSVectorStore) else vector_store.faiss_store
                    )

                    print_info("Merging BM25 indexes...")
                    if hasattr(existing_store, 'bm25_store') and hasattr(vector_store, 'bm25_store'):
                        existing_store.bm25_store.merge(vector_store.bm25_store)

                    # Save merged store
                    print_info("Saving merged vector store...")
                    existing_store.save(merge_target)

                    merged_stats = existing_store.get_stats()
                    print_success(f"Merge complete!")
                    print_info(f"Merged store: {merged_stats['total_vectors']} vectors, "
                              f"{merged_stats['documents']} documents")
                    print_info(f"Added: {faiss_merge_stats['added']} vectors, "
                              f"Skipped: {faiss_merge_stats['skipped']} duplicates")

                except FileNotFoundError as e:
                    print()
                    print_info(f"[ERROR] Vector store files missing: {e}")
                    logger.error(f"Vector store merge failed - files missing: {e}", exc_info=True)
                    print_info(f"Vector store merge skipped - new store will be created instead")
                except (PermissionError, OSError) as e:
                    print()
                    print_info(f"[ERROR] Cannot write to vector store: {e}")
                    logger.error(f"Vector store merge failed - IO error: {e}", exc_info=True)
                    print_info(f"Vector store merge skipped - keeping separate stores")
                except (ValueError, RuntimeError) as e:
                    # Expected errors from merge validation/corruption detection
                    print()
                    print_info(f"[ERROR] Vector store validation failed: {e}")
                    logger.error(f"Vector store merge failed - validation error: {e}", exc_info=True)
                    print_info(f"Vector store merge skipped - store may be corrupted")
                # Removed broad Exception catch - let programming errors propagate!

                # Knowledge graph merging
                try:

                    # Merge Knowledge Graphs with cross-document relationships
                    if knowledge_graph and config.enable_knowledge_graph:
                        print()
                        print_info("Merging knowledge graphs with cross-document deduplication...")

                        from src.graph import (
                            KnowledgeGraph,
                            UnifiedKnowledgeGraphManager,
                            CrossDocumentRelationshipDetector
                        )

                        # Initialize unified KG manager (uses merge_target directory for storage)
                        # KG files should be stored in the same directory as the vector store
                        manager = UnifiedKnowledgeGraphManager(storage_dir=str(merge_target))

                        # Initialize cross-document detector
                        detector = CrossDocumentRelationshipDetector(
                            use_llm_validation=False,  # Fast pattern-based detection
                            confidence_threshold=0.7
                        )

                        # Load or create unified KG
                        unified_kg = manager.load_or_create()

                        print_info(f"Current unified KG: {len(unified_kg.entities)} entities, "
                                  f"{len(unified_kg.relationships)} relationships")
                        print_info(f"New document KG: {len(knowledge_graph.entities)} entities, "
                                  f"{len(knowledge_graph.relationships)} relationships")

                        # Merge with deduplication and cross-doc detection
                        unified_kg = manager.merge_document_graph(
                            unified_kg=unified_kg,
                            document_kg=knowledge_graph,
                            document_id=doc_name,
                            cross_doc_detector=detector
                        )

                        # Save unified KG + per-document backup
                        manager.save(unified_kg, document_id=doc_name)

                        # Get statistics
                        doc_stats = manager.get_document_statistics(unified_kg)

                        print_success(f"KG merge complete with cross-document relationships!")
                        print_info(f"Unified KG: {len(unified_kg.entities)} entities, "
                                  f"{len(unified_kg.relationships)} relationships")
                        print_info(f"Documents in unified KG: {doc_stats['total_documents']}")
                        print_info(f"Cross-document entities: {doc_stats['cross_document_entities']} "
                                  f"({doc_stats['cross_document_entity_percentage']:.1f}%)")
                        print_info(f"Saved: {merge_target / 'unified_kg.json'}")

                except FileNotFoundError as e:
                    print()
                    print_info(f"[ERROR] KG file not found: {e}")
                    logger.error(f"KG merge failed - file missing: {e}", exc_info=True)
                    logger.error(f"Document: {doc_name}, Merge target: {merge_target}")
                    print_info(f"Document '{doc_name}' KG will not be merged into unified graph")
                except PermissionError as e:
                    print()
                    print_info(f"[ERROR] Cannot write to {merge_target}: Permission denied")
                    logger.error(f"KG merge failed - permission error: {e}", exc_info=True)
                    print_info(f"Document '{doc_name}' KG will not be merged")
                except (KeyError, AttributeError) as e:
                    # Data structure errors - likely data corruption
                    print()
                    print_info(f"[ERROR] KG data structure error: {e}")
                    logger.error(f"KG merge failed - data integrity issue: {e}", exc_info=True)
                    logger.error(f"Document: {doc_name}")
                    if 'unified_kg' in locals():
                        logger.error(f"Unified KG state: {len(unified_kg.entities)} entities, {len(unified_kg.relationships)} relationships")
                    logger.error(f"New KG state: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relationships)} relationships")
                    print_info(f"Document '{doc_name}' KG appears corrupted - skipping merge")
                except (ValueError, RuntimeError) as e:
                    # Expected validation/merge errors
                    print()
                    print_info(f"[ERROR] KG merge validation failed: {e}")
                    logger.error(f"KG merge failed - validation error: {e}", exc_info=True)
                    print_info(f"Document '{doc_name}' KG validation failed - skipping merge")
                # Removed broad Exception catch - let unexpected errors propagate!

        # Print comprehensive statistics
        print_header("INDEXING COMPLETE")
        print_success(f"Document processed: {document_path.name}")
        print()

        print_info("📊 PIPELINE STATISTICS")
        print_info("-" * 60)

        # Chunking stats
        chunking = stats["chunking"]
        print_info(f"Layer 1 (documents):  {chunking['layer1_count']:4d} chunks")
        print_info(f"Layer 2 (sections):   {chunking['layer2_count']:4d} chunks")
        print_info(f"Layer 3 (paragraphs): {chunking['layer3_count']:4d} chunks (PRIMARY)")
        print_info(f"Total chunks:         {chunking['total_chunks']:4d}")

        if 'layer3_avg_size' in chunking:
            print()
            print_info(f"Layer 3 avg size:     {chunking['layer3_avg_size']:.0f} chars")
            if 'context_avg_overhead' in chunking:
                print_info(f"SAC context overhead: {chunking['context_avg_overhead']:.0f} chars/chunk")

        # Vector store stats
        vs_stats = stats["vector_store"]
        print()
        print_info(f"Vector dimensions:    {vs_stats.get('dense_dimensions', vs_stats.get('dimensions', 'N/A'))}D")
        print_info(f"Total vectors:        {vs_stats['total_vectors']}")
        print_info(f"Documents indexed:    {vs_stats['documents']}")

        # Hybrid search stats
        if stats.get("hybrid_enabled"):
            print()
            print_info("✅ Hybrid Search: ACTIVE")
            print_info(f"   BM25 Layer 1:      {vs_stats.get('bm25_layer1_docs', 0)} docs")
            print_info(f"   BM25 Layer 2:      {vs_stats.get('bm25_layer2_docs', 0)} docs")
            print_info(f"   BM25 Layer 3:      {vs_stats.get('bm25_layer3_docs', 0)} docs")
            print_info(f"   RRF Fusion k:      {config.hybrid_fusion_k}")
        else:
            print()
            print_info("[INFO]  Hybrid Search: DISABLED (dense-only retrieval)")

        # Knowledge graph stats
        if stats.get("kg_construction_failed"):
            print()
            print_info("[FAILED]Knowledge Graph: FAILED")
            print_info(f"   Error: {stats.get('kg_error', 'Unknown error')}")
            print_info("   Continuing with vector search only")
        elif stats.get("kg_enabled") and knowledge_graph:
            print()
            print_info("✅ Knowledge Graph: ACTIVE")
            print_info(f"   Entities:          {stats['kg_entities']}")
            print_info(f"   Relationships:     {stats['kg_relationships']}")
        elif stats.get("kg_enabled"):
            print()
            print_info("[WARNING]  Knowledge Graph: ENABLED but no graph generated")
        else:
            print()
            print_info("[INFO]  Knowledge Graph: DISABLED")

        print()

    except FileNotFoundError as e:
        logger.error(f"Document not found: {e}", exc_info=True)
        print(f"\n✗ ERROR: Document not found")
        print(f"   {e}")
        print("\nPlease check:")
        print(f"  - Path is correct: {document_path}")
        print("  - File exists and is readable")
        sys.exit(1)

    except PermissionError as e:
        logger.error(f"Permission denied: {e}", exc_info=True)
        print(f"\n✗ ERROR: Permission denied")
        print(f"   {e}")
        print("\nPlease check:")
        print("  - File/directory permissions")
        print("  - Current user has read/write access")
        sys.exit(1)

    except MemoryError as e:
        logger.error(f"Out of memory: {e}", exc_info=True)
        print(f"\n✗ ERROR: Out of memory")
        print(f"   Document may be too large: {document_path}")
        print("\nSolutions:")
        print("  - Process smaller documents")
        print("  - Increase system memory")
        print("  - Reduce batch_size in config")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n\n[WARNING]  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)

    except Exception as e:
        # Check if it's an API error (works for both anthropic and openai)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['api', 'authentication', 'auth', 'key', 'quota', 'rate limit']):
            logger.error(f"API error: {e}", exc_info=True)
            print(f"\n✗ ERROR: API call failed")
            print(f"   {e}")
            print("\nPlease check:")
            print("  - API keys are valid (check .env file)")
            print("  - API quota/rate limits")
            print("  - Internet connectivity")
            print("  - API service status")
        else:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n✗ ERROR: {e}")
            print()
            print("Common issues:")
            print("  - Missing API keys (check .env file)")
            print("  - Invalid document format")
            print("  - Insufficient disk space")
            print("\nFor more details, check the log file above.")
        sys.exit(1)

    # Print cost summary
    tracker = get_global_tracker()

    if tracker.total_cost > 0:
        print_header("💰 API COST SUMMARY")
        print_info(f"Total cost:        ${tracker.total_cost:.4f} USD")
        print_info(f"Total calls:       {len(tracker.entries)}")
        print_info(f"Input tokens:      {tracker.total_input_tokens:,}")
        print_info(f"Output tokens:     {tracker.total_output_tokens:,}")
        print()

        # Cost by provider
        if tracker.cost_by_provider:
            print_info("Cost by provider:")
            for provider, cost in tracker.cost_by_provider.items():
                print_info(f"  {provider:12s}  ${cost:.4f}")
            print()

        # Cost by operation
        if tracker.cost_by_operation:
            print_info("Cost by operation:")
            for operation, cost in tracker.cost_by_operation.items():
                print_info(f"  {operation:12s}  ${cost:.4f}")
        print()

    # Final success message
    print("=" * 80)
    print(f"✅ ALL DONE! Outputs saved in: {output_dir}")
    print("=" * 80)
    print()
    print("📁 Output files:")
    print(f"  • phase1_extraction.json    - Document structure")
    print(f"  • phase2_summaries.json      - Section summaries")
    print(f"  • phase3_chunks.json         - Multi-layer chunks")
    print(f"  • phase4_vector_store/       - FAISS + BM25 indexes")
    if knowledge_graph:
        print(f"  • {doc_name}_kg.json         - Knowledge graph")
    print()
    print("🚀 To use with agent:")
    print(f"  uv run python -m src.agent.cli --vector-store {vs_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete SOTA 2025 RAG Pipeline - Index documents with automatic merge support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index single document (auto-merges to vector_db/)
  python run_pipeline.py data/document.pdf

  # Index directory batch (auto-merges all to vector_db/)
  python run_pipeline.py data/documents/

  # Index to custom location instead of vector_db
  python run_pipeline.py data/document.pdf --merge custom_db

  # Index without merging (keep separate)
  python run_pipeline.py data/document.pdf --no-merge
        """
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to document file or directory to index"
    )

    parser.add_argument(
        "--merge",
        type=str,
        metavar="TARGET",
        default="vector_db",  # Default to vector_db/ for automatic merging
        help="Merge indexed documents into existing vector store at TARGET path (default: vector_db)"
    )

    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable automatic merging to vector_db (keep documents separate)"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["faiss", "postgresql"],
        default=None,
        help="Storage backend for vectors (default: from config.json). Options: faiss, postgresql"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    # Use merge target unless --no-merge is specified
    merge_target = None if args.no_merge else Path(args.merge)

    run_complete_pipeline(input_path, merge_target=merge_target, storage_backend=args.backend)
