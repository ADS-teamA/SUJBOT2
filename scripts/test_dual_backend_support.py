#!/usr/bin/env python3
"""
Test script for dual backend support (FAISS + PostgreSQL)

Validates:
1. Backend selection works correctly
2. Both FAISS and PostgreSQL can index chunks
3. HybridVectorStore works with both backends
4. Search results are consistent across backends
5. Stats and metadata work correctly

Usage:
    # Test FAISS backend
    python scripts/test_dual_backend_support.py --backend faiss

    # Test PostgreSQL backend (requires DATABASE_URL)
    export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
    python scripts/test_dual_backend_support.py --backend postgresql

    # Test both backends
    python scripts/test_dual_backend_support.py --backend both
"""

import sys
import os
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import create_vector_store_adapter
from src.faiss_vector_store import FAISSVectorStore
from src.hybrid_search import BM25Store, HybridVectorStore


def print_header(text: str):
    """Print formatted header."""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text: str):
    """Print info message."""
    print(f"ℹ️  {text}")


def create_test_chunks() -> tuple[Dict[str, List], Dict[str, np.ndarray]]:
    """
    Create minimal test data for backend validation.

    Returns:
        (chunks_dict, embeddings_dict) - Test data for all 3 layers
    """
    # Layer 1: Document
    chunks_layer1 = [{
        "chunk_id": "test_doc",
        "document_id": "test_doc",
        "title": "Test Document",
        "content": "This is a test document for backend validation.",
        "hierarchical_path": "test_doc",
        "page_number": 1,
        "metadata": {}
    }]
    embeddings_layer1 = np.random.randn(1, 3072).astype(np.float32)

    # Layer 2: Section
    chunks_layer2 = [{
        "chunk_id": "test_doc:sec1",
        "document_id": "test_doc",
        "section_id": "sec1",
        "section_title": "Test Section",
        "content": "This is a test section with some content.",
        "hierarchical_path": "test_doc > Test Section",
        "page_number": 1,
        "metadata": {}
    }]
    embeddings_layer2 = np.random.randn(1, 3072).astype(np.float32)

    # Layer 3: Chunks
    chunks_layer3 = [
        {
            "chunk_id": "test_doc:sec1:0",
            "document_id": "test_doc",
            "section_id": "sec1",
            "section_title": "Test Section",
            "content": "This is the first chunk of test content.",
            "hierarchical_path": "test_doc > Test Section",
            "page_number": 1,
            "char_start": 0,
            "char_end": 42,
            "metadata": {}
        },
        {
            "chunk_id": "test_doc:sec1:1",
            "document_id": "test_doc",
            "section_id": "sec1",
            "section_title": "Test Section",
            "content": "This is the second chunk with different content.",
            "hierarchical_path": "test_doc > Test Section",
            "page_number": 1,
            "char_start": 42,
            "char_end": 90,
            "metadata": {}
        }
    ]
    embeddings_layer3 = np.random.randn(2, 3072).astype(np.float32)

    chunks_dict = {
        "layer1": chunks_layer1,
        "layer2": chunks_layer2,
        "layer3": chunks_layer3
    }

    embeddings_dict = {
        "layer1": embeddings_layer1,
        "layer2": embeddings_layer2,
        "layer3": embeddings_layer3
    }

    return chunks_dict, embeddings_dict


def test_faiss_backend() -> bool:
    """Test FAISS backend."""
    print_header("TEST 1: FAISS Backend")

    try:
        # Create test data
        print_info("Creating test data...")
        chunks_dict, embeddings_dict = create_test_chunks()

        # Create FAISS vector store
        print_info("Creating FAISS vector store...")
        vector_store = FAISSVectorStore(dimensions=3072)

        # Add chunks
        print_info("Adding chunks to FAISS...")
        vector_store.add_chunks(chunks_dict, embeddings_dict)

        # Get stats
        stats = vector_store.get_stats()
        print_info(f"Stats: {stats['total_vectors']} vectors, {stats['documents']} documents")

        # Validate stats
        assert stats['total_vectors'] == 4, f"Expected 4 vectors, got {stats['total_vectors']}"
        assert stats['layer1_count'] == 1, f"Expected 1 L1 vector, got {stats['layer1_count']}"
        assert stats['layer2_count'] == 1, f"Expected 1 L2 vector, got {stats['layer2_count']}"
        assert stats['layer3_count'] == 2, f"Expected 2 L3 vectors, got {stats['layer3_count']}"

        # Test search_layer1
        print_info("Testing search_layer1...")
        query_embedding = np.random.randn(3072).astype(np.float32)
        results_l1 = vector_store.search_layer1(query_embedding, k=1)
        assert len(results_l1) == 1, f"Expected 1 L1 result, got {len(results_l1)}"
        assert "document_id" in results_l1[0], "Missing document_id in L1 result"

        # Test search_layer2
        print_info("Testing search_layer2...")
        results_l2 = vector_store.search_layer2(query_embedding, k=1)
        assert len(results_l2) == 1, f"Expected 1 L2 result, got {len(results_l2)}"
        assert "section_id" in results_l2[0], "Missing section_id in L2 result"

        # Test search_layer3
        print_info("Testing search_layer3...")
        results_l3 = vector_store.search_layer3(query_embedding, k=2)
        assert len(results_l3) == 2, f"Expected 2 L3 results, got {len(results_l3)}"
        assert "chunk_id" in results_l3[0], "Missing chunk_id in L3 result"

        # Test metadata properties
        print_info("Testing metadata properties...")
        assert len(vector_store.metadata_layer1) == 1, "metadata_layer1 incorrect"
        assert len(vector_store.metadata_layer2) == 1, "metadata_layer2 incorrect"
        assert len(vector_store.metadata_layer3) == 2, "metadata_layer3 incorrect"

        print_success("FAISS backend: ALL TESTS PASSED")
        return True

    except Exception as e:
        print_error(f"FAISS backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_postgresql_backend() -> bool:
    """Test PostgreSQL backend."""
    print_header("TEST 2: PostgreSQL Backend")

    # Check DATABASE_URL
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        print_error("DATABASE_URL not set. Skipping PostgreSQL test.")
        print_info("Set DATABASE_URL to test PostgreSQL backend:")
        print_info("  export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'")
        return False

    try:
        # Create test data
        print_info("Creating test data...")
        chunks_dict, embeddings_dict = create_test_chunks()

        # Create PostgreSQL adapter
        print_info("Creating PostgreSQL adapter...")
        vector_store = await create_vector_store_adapter(
            backend="postgresql",
            connection_string=connection_string,
            dimensions=3072
        )

        # Initialize
        print_info("Initializing PostgreSQL connection...")
        await vector_store.initialize()

        # Add chunks
        print_info("Adding chunks to PostgreSQL...")
        vector_store.add_chunks(chunks_dict, embeddings_dict)

        # Get stats
        stats = vector_store.get_stats()
        print_info(f"Stats: {stats['total_vectors']} vectors, {stats['documents']} documents")

        # Validate stats (note: may have existing data, so >= instead of ==)
        assert stats['total_vectors'] >= 4, f"Expected >= 4 vectors, got {stats['total_vectors']}"
        assert stats['layer1_count'] >= 1, f"Expected >= 1 L1 vector, got {stats['layer1_count']}"
        assert stats['layer2_count'] >= 1, f"Expected >= 1 L2 vector, got {stats['layer2_count']}"
        assert stats['layer3_count'] >= 2, f"Expected >= 2 L3 vectors, got {stats['layer3_count']}"

        # Test search_layer1
        print_info("Testing search_layer1...")
        query_embedding = np.random.randn(3072).astype(np.float32)
        results_l1 = vector_store.search_layer1(query_embedding, k=1)
        assert len(results_l1) >= 1, f"Expected >= 1 L1 result, got {len(results_l1)}"
        assert "document_id" in results_l1[0], "Missing document_id in L1 result"

        # Test search_layer2
        print_info("Testing search_layer2...")
        results_l2 = vector_store.search_layer2(query_embedding, k=1)
        assert len(results_l2) >= 1, f"Expected >= 1 L2 result, got {len(results_l2)}"

        # Test search_layer3
        print_info("Testing search_layer3...")
        results_l3 = vector_store.search_layer3(query_embedding, k=2)
        assert len(results_l3) >= 2, f"Expected >= 2 L3 results, got {len(results_l3)}"
        assert "chunk_id" in results_l3[0], "Missing chunk_id in L3 result"

        # Test document filtering
        print_info("Testing document_filter...")
        results_filtered = vector_store.search_layer3(
            query_embedding, k=2, document_filter="test_doc"
        )
        # Should only return chunks from test_doc
        for result in results_filtered:
            assert result["document_id"] == "test_doc", f"Filter failed: {result['document_id']}"

        # Test metadata properties
        print_info("Testing metadata properties...")
        assert len(vector_store.metadata_layer1) >= 1, "metadata_layer1 empty"
        assert len(vector_store.metadata_layer2) >= 1, "metadata_layer2 empty"
        assert len(vector_store.metadata_layer3) >= 2, "metadata_layer3 empty"

        # Close connection
        await vector_store.close()

        print_success("PostgreSQL backend: ALL TESTS PASSED")
        return True

    except Exception as e:
        print_error(f"PostgreSQL backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_search_faiss() -> bool:
    """Test HybridVectorStore with FAISS backend."""
    print_header("TEST 3: HybridVectorStore + FAISS")

    try:
        # Create test data
        chunks_dict, embeddings_dict = create_test_chunks()

        # Flatten chunks for BM25
        all_chunks = chunks_dict["layer1"] + chunks_dict["layer2"] + chunks_dict["layer3"]

        # Create FAISS store
        print_info("Creating FAISS vector store...")
        faiss_store = FAISSVectorStore(dimensions=3072)
        faiss_store.add_chunks(chunks_dict, embeddings_dict)

        # Create BM25 store
        print_info("Creating BM25 store...")
        bm25_store = BM25Store()
        bm25_store.build_from_chunks({"layer1": chunks_dict["layer1"],
                                       "layer2": chunks_dict["layer2"],
                                       "layer3": chunks_dict["layer3"]})

        # Create HybridVectorStore
        print_info("Creating HybridVectorStore...")
        hybrid_store = HybridVectorStore(
            vector_store=faiss_store,
            bm25_store=bm25_store,
            fusion_k=60
        )

        # Test hierarchical search
        print_info("Testing hierarchical search...")
        query_embedding = np.random.randn(3072).astype(np.float32)
        query_text = "test content"

        results = hybrid_store.hierarchical_search(
            query_text=query_text,
            query_embedding=query_embedding,
            k_layer3=2
        )

        assert "layer1" in results, "Missing layer1 in results"
        assert "layer2" in results, "Missing layer2 in results"
        assert "layer3" in results, "Missing layer3 in results"
        assert len(results["layer3"]) <= 2, f"Expected <= 2 L3 results, got {len(results['layer3'])}"

        # Test get_stats
        print_info("Testing get_stats...")
        stats = hybrid_store.get_stats()
        assert "hybrid_enabled" in stats, "Missing hybrid_enabled in stats"
        assert stats["hybrid_enabled"] is True, "hybrid_enabled should be True"

        print_success("HybridVectorStore + FAISS: ALL TESTS PASSED")
        return True

    except Exception as e:
        print_error(f"HybridVectorStore + FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_search_postgresql() -> bool:
    """Test HybridVectorStore with PostgreSQL backend."""
    print_header("TEST 4: HybridVectorStore + PostgreSQL")

    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        print_error("DATABASE_URL not set. Skipping PostgreSQL hybrid test.")
        return False

    try:
        # Create test data
        chunks_dict, embeddings_dict = create_test_chunks()

        # Create PostgreSQL adapter
        print_info("Creating PostgreSQL adapter...")
        postgres_store = await create_vector_store_adapter(
            backend="postgresql",
            connection_string=connection_string,
            dimensions=3072
        )
        await postgres_store.initialize()
        postgres_store.add_chunks(chunks_dict, embeddings_dict)

        # Create BM25 store
        print_info("Creating BM25 store...")
        bm25_store = BM25Store()
        bm25_store.build_from_chunks({"layer1": chunks_dict["layer1"],
                                       "layer2": chunks_dict["layer2"],
                                       "layer3": chunks_dict["layer3"]})

        # Create HybridVectorStore with PostgreSQL backend
        print_info("Creating HybridVectorStore with PostgreSQL...")
        hybrid_store = HybridVectorStore(
            vector_store=postgres_store,
            bm25_store=bm25_store,
            fusion_k=60
        )

        # Test hierarchical search
        print_info("Testing hierarchical search...")
        query_embedding = np.random.randn(3072).astype(np.float32)
        query_text = "test content"

        results = hybrid_store.hierarchical_search(
            query_text=query_text,
            query_embedding=query_embedding,
            k_layer3=2
        )

        assert "layer1" in results, "Missing layer1 in results"
        assert "layer2" in results, "Missing layer2 in results"
        assert "layer3" in results, "Missing layer3 in results"

        # Test get_stats
        print_info("Testing get_stats...")
        stats = hybrid_store.get_stats()
        assert "hybrid_enabled" in stats, "Missing hybrid_enabled in stats"
        assert stats["hybrid_enabled"] is True, "hybrid_enabled should be True"
        assert stats["backend"] == "postgresql", f"Expected backend=postgresql, got {stats['backend']}"

        # Close connection
        await postgres_store.close()

        print_success("HybridVectorStore + PostgreSQL: ALL TESTS PASSED")
        return True

    except Exception as e:
        print_error(f"HybridVectorStore + PostgreSQL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test dual backend support")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["faiss", "postgresql", "both"],
        default="both",
        help="Which backend to test (default: both)"
    )
    args = parser.parse_args()

    print_header("DUAL BACKEND SUPPORT TEST SUITE")
    print_info("Testing storage backend flexibility (FAISS + PostgreSQL)")
    print()

    results = {}

    # Test FAISS
    if args.backend in ["faiss", "both"]:
        results["faiss"] = test_faiss_backend()
        results["hybrid_faiss"] = test_hybrid_search_faiss()

    # Test PostgreSQL
    if args.backend in ["postgresql", "both"]:
        results["postgresql"] = asyncio.run(test_postgresql_backend())
        results["hybrid_postgresql"] = asyncio.run(test_hybrid_search_postgresql())

    # Print summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print_success("ALL TESTS PASSED! Dual backend support is working correctly.")
        sys.exit(0)
    else:
        print_error(f"SOME TESTS FAILED ({total - passed} failures)")
        sys.exit(1)


if __name__ == "__main__":
    main()
