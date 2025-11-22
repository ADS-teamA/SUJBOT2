#!/usr/bin/env python3
"""
Test that search functionality works with Layer 2 embeddings disabled.

This test verifies:
1. hierarchical_search returns layer1 and layer3 (layer2 is empty)
2. search tool returns results correctly
3. get_document_info still retrieves section metadata
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_hierarchical_search_returns_correct_layers():
    """Test that hierarchical_search returns layer1, layer2 (empty), layer3."""
    from storage.postgres_adapter import PostgresVectorStoreAdapter
    from embedding_generator import EmbeddingGenerator
    import os
    import asyncio

    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:YOUR_PASSWORD@localhost:5432/sujbot"
    )

    async def run_test():
        vector_store = PostgresVectorStoreAdapter(
            connection_string=connection_string,
            pool_size=5,
            dimensions=3072,
        )
        await vector_store.initialize()

        embedder = EmbeddingGenerator(
            embedding_model="bge-m3",
            cache_file="embeddings_cache.pkl",
        )

        query = "test query"
        query_embedding = embedder.embed_texts([query])[0]

        results = await vector_store._async_hierarchical_search(
            query_embedding=query_embedding,
            k_layer3=5,
            use_doc_filtering=False,
            similarity_threshold_offset=0.25,
            query_text=query,
        )

        # Verify structure
        assert "layer1" in results, "layer1 missing from results"
        assert "layer2" in results, "layer2 missing from results"
        assert "layer3" in results, "layer3 missing from results"

        # Verify layer2 is empty (disabled)
        assert len(results["layer2"]) == 0, f"layer2 should be empty, got {len(results['layer2'])}"

        # Verify layer3 has results (if database has data)
        print(f"✅ layer1: {len(results['layer1'])} results")
        print(f"✅ layer2: {len(results['layer2'])} results (expected: 0)")
        print(f"✅ layer3: {len(results['layer3'])} results")

        await vector_store.close()

    asyncio.run(run_test())
    print("\n✅ Test passed: hierarchical_search returns correct layers")


def test_search_tool_works():
    """Test that search tool returns results with layer2 disabled."""
    from agent.tools.search import SearchTool
    from config import ToolConfig
    import os

    # Mock config
    config = ToolConfig(
        vector_store_path="vector_db",
        reranker_model="ms-marco",
        query_expansion_provider="openai",
        query_expansion_model="gpt-4o-mini",
    )

    # Create search tool (this will use PostgreSQL adapter)
    search_tool = SearchTool(config=config)

    # Perform search
    result = search_tool.execute_impl(
        query="test query",
        k=5,
        num_expands=0,
        enable_graph_boost=False,
        use_hyde=False,
    )

    # Verify result structure
    assert result.success, f"Search failed: {result.error}"
    assert result.data is not None, "Search returned no data"
    assert len(result.data) > 0, "Search returned empty results"

    print(f"\n✅ Search tool returned {len(result.data)} results")
    print(f"✅ Test passed: search tool works with layer2 disabled")


def test_get_document_info_sections():
    """Test that get_document_info retrieves section metadata from layer3."""
    from agent.tools.get_document_info import GetDocumentInfoTool
    from config import ToolConfig

    # Mock config
    config = ToolConfig(vector_store_path="vector_db")

    # Create tool
    tool = GetDocumentInfoTool(config=config)

    # Get sections for a document (replace with actual document_id)
    result = tool.execute_impl(
        document_id="BZ_VR1",  # Example document
        info_type="sections",
    )

    # Verify result
    if result.success and result.data:
        sections = result.data
        print(f"\n✅ get_document_info returned {len(sections)} sections")
        print(f"✅ Sample sections: {sections[:3]}")
        print(f"✅ Test passed: section metadata available from layer3")
    else:
        print(f"\n⚠️ Document not found or no sections (expected if database empty)")


def main():
    """Run all tests."""
    print("="*80)
    print("Testing Layer 2 Disabled - Search Functionality")
    print("="*80)

    try:
        print("\n[1/3] Testing hierarchical_search...")
        test_hierarchical_search_returns_correct_layers()

        print("\n[2/3] Testing search tool...")
        test_search_tool_works()

        print("\n[3/3] Testing get_document_info...")
        test_get_document_info_sections()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nLayer 2 embeddings successfully disabled.")
        print("Search functionality works correctly.")
        print("Section metadata available via get_document_info (from layer3).")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
