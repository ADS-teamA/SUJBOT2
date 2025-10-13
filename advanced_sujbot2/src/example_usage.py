"""
Example Usage of Hybrid Retrieval System

This script demonstrates how to use the triple hybrid retrieval system.

Author: SUJBOT2 Team
Date: 2025-10-08
"""

import asyncio
import logging
from pathlib import Path

from hybrid_retriever import (
    HybridRetriever,
    SemanticSearcher,
    KeywordSearcher,
    StructuralSearcher,
    RetrievalConfig,
    MultiDocumentVectorStore,
    LegalEmbedder,
    LegalChunk,
    create_hybrid_retriever,
    CachedHybridRetriever,
    QueryExpander
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

async def example_basic_usage():
    """Basic usage with default configuration"""

    logger.info("=== Example 1: Basic Usage ===")

    # Initialize components
    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder(model_name="BAAI/bge-m3")

    # Create retriever with defaults
    retriever = create_hybrid_retriever(vector_store, embedder)

    # Add some mock data
    mock_chunks = [
        LegalChunk(
            chunk_id="chunk_001",
            content="Dodavatel odpovídá za vady díla po dobu 24 měsíců od převzetí.",
            document_type="contract",
            hierarchy_path="Smlouva / Část II / Článek 5",
            legal_reference="Článek 5",
            structural_level="article",
            metadata={
                'part': 'II',
                'article': 5,
                'content_type': 'obligation',
                'references_to': ['§89']
            }
        ),
        LegalChunk(
            chunk_id="chunk_002",
            content="§89 odst. 1: Dodavatel odpovídá za to, že dílo bude bez vad.",
            document_type="law_code",
            hierarchy_path="Zákon / Část IV / Hlava II / §89",
            legal_reference="§89 odst. 1",
            structural_level="subsection",
            metadata={
                'part': 'IV',
                'chapter': 'II',
                'paragraph': 89,
                'subsection': 1,
                'content_type': 'obligation'
            }
        ),
    ]

    # Add to vector store (mock implementation)
    vector_store.metadata_stores["doc_001"] = {
        chunk.chunk_id: chunk for chunk in mock_chunks
    }

    # Search
    query = "Jaká je odpovědnost dodavatele za vady?"
    logger.info(f"Query: {query}")

    results = await retriever.search(query, top_k=5)

    # Display results
    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. Score: {result.score:.3f}")
        print(f"   Legal Reference: {result.chunk.legal_reference}")
        print(f"   Semantic Score: {result.semantic_score:.3f if result.semantic_score else 'N/A'}")
        print(f"   Keyword Score: {result.keyword_score:.3f if result.keyword_score else 'N/A'}")
        print(f"   Structural Score: {result.structural_score:.3f if result.structural_score else 'N/A'}")
        print(f"   Content: {result.chunk.content}")


# ============================================================================
# Example 2: Custom Configuration
# ============================================================================

async def example_custom_config():
    """Using custom configuration for high precision"""

    logger.info("\n=== Example 2: Custom Configuration ===")

    # Custom config for high precision
    config = RetrievalConfig(
        semantic_weight=0.6,
        keyword_weight=0.25,
        structural_weight=0.15,
        top_k=10,
        min_score_threshold=0.3,
        adaptive_weights=True,
        normalize_scores=True
    )

    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()

    retriever = create_hybrid_retriever(vector_store, embedder, config)

    logger.info(f"Configuration: {config}")


# ============================================================================
# Example 3: Filtered Search
# ============================================================================

async def example_filtered_search():
    """Search with metadata filters"""

    logger.info("\n=== Example 3: Filtered Search ===")

    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()
    retriever = create_hybrid_retriever(vector_store, embedder)

    # Search only for obligations in Part II
    results = await retriever.search(
        query="povinnosti dodavatele",
        filters={
            'content_type': 'obligation',
            'part': 'II'
        },
        top_k=5
    )

    logger.info(f"Found {len(results)} results matching filters")


# ============================================================================
# Example 4: Document-Specific Search
# ============================================================================

async def example_document_specific():
    """Search in specific documents"""

    logger.info("\n=== Example 4: Document-Specific Search ===")

    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()
    retriever = create_hybrid_retriever(vector_store, embedder)

    # Search only in contract documents
    results = await retriever.search(
        query="záruční doba",
        document_ids=["contract_001", "contract_002"],
        top_k=5
    )

    logger.info(f"Found {len(results)} results in specified documents")


# ============================================================================
# Example 5: Adaptive Weighting
# ============================================================================

async def example_adaptive_weighting():
    """Demonstrate adaptive weight adjustment"""

    logger.info("\n=== Example 5: Adaptive Weighting ===")

    config = RetrievalConfig(adaptive_weights=True)
    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()

    retriever = create_hybrid_retriever(vector_store, embedder, config)

    # Query with legal reference (should boost structural)
    query1 = "podle §89 občanského zákoníku"
    weights1 = retriever._get_adaptive_weights(query1, None)
    logger.info(f"Query: '{query1}'")
    logger.info(f"Weights: {weights1}")

    # Short query (should boost keyword)
    query2 = "záruční doba"
    weights2 = retriever._get_adaptive_weights(query2, None)
    logger.info(f"\nQuery: '{query2}'")
    logger.info(f"Weights: {weights2}")

    # Long analytical query (should boost semantic)
    query3 = "Jaké jsou povinnosti dodavatele ve vztahu k zodpovědnosti za vady díla podle smlouvy o dílo?"
    weights3 = retriever._get_adaptive_weights(query3, None)
    logger.info(f"\nQuery: '{query3}'")
    logger.info(f"Weights: {weights3}")


# ============================================================================
# Example 6: Query Expansion
# ============================================================================

async def example_query_expansion():
    """Demonstrate query expansion with synonyms"""

    logger.info("\n=== Example 6: Query Expansion ===")

    expander = QueryExpander()

    query = "odpovědnost dodavatele"
    variations = await expander.expand(query, max_expansions=3)

    logger.info(f"Original query: {query}")
    logger.info(f"Expanded queries: {variations}")


# ============================================================================
# Example 7: Caching
# ============================================================================

async def example_caching():
    """Demonstrate caching for repeated queries"""

    logger.info("\n=== Example 7: Caching ===")

    config = RetrievalConfig()
    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()

    # Create components
    semantic = SemanticSearcher(embedder, vector_store)
    keyword = KeywordSearcher(vector_store)
    structural = StructuralSearcher(vector_store)

    # Create cached retriever
    retriever = CachedHybridRetriever(
        semantic,
        keyword,
        structural,
        config,
        cache_size=1000
    )

    # First query (cache miss)
    query = "odpovědnost za vady"
    logger.info(f"First query: {query}")
    results1 = await retriever.search(query)

    # Second query (cache hit)
    logger.info(f"Second query (should be cached): {query}")
    results2 = await retriever.search(query)

    logger.info(f"Results are identical: {results1 == results2}")


# ============================================================================
# Example 8: Score Breakdown
# ============================================================================

async def example_score_breakdown():
    """Analyze score breakdown for results"""

    logger.info("\n=== Example 8: Score Breakdown ===")

    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()
    retriever = create_hybrid_retriever(vector_store, embedder)

    # Add mock data
    mock_chunk = LegalChunk(
        chunk_id="chunk_001",
        content="Dodavatel odpovídá za vady díla.",
        document_type="contract",
        hierarchy_path="Smlouva / Článek 5",
        legal_reference="Článek 5",
        structural_level="article",
        metadata={'content_type': 'obligation'}
    )

    vector_store.metadata_stores["doc_001"] = {"chunk_001": mock_chunk}

    results = await retriever.search("odpovědnost dodavatele", top_k=1)

    if results:
        result = results[0]
        breakdown = result.get_score_breakdown()

        logger.info("Score Breakdown:")
        logger.info(f"  Semantic:    {breakdown['semantic']:.4f}")
        logger.info(f"  Keyword:     {breakdown['keyword']:.4f}")
        logger.info(f"  Structural:  {breakdown['structural']:.4f}")
        logger.info(f"  Combined:    {breakdown['combined']:.4f}")


# ============================================================================
# Example 9: Batch Search
# ============================================================================

async def example_batch_search():
    """Search multiple queries in parallel"""

    logger.info("\n=== Example 9: Batch Search ===")

    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()
    retriever = create_hybrid_retriever(vector_store, embedder)

    queries = [
        "odpovědnost dodavatele",
        "záruční doba",
        "sankce za porušení smlouvy",
        "povinnosti objednatele",
        "platební podmínky"
    ]

    # Execute searches in parallel
    tasks = [retriever.search(query, top_k=3) for query in queries]
    all_results = await asyncio.gather(*tasks)

    for query, results in zip(queries, all_results):
        logger.info(f"Query: '{query}' → {len(results)} results")


# ============================================================================
# Example 10: Complete Workflow
# ============================================================================

async def example_complete_workflow():
    """Complete workflow: setup, index, search, analyze"""

    logger.info("\n=== Example 10: Complete Workflow ===")

    # Step 1: Configuration
    config = RetrievalConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        structural_weight=0.2,
        top_k=20,
        adaptive_weights=True,
        enable_query_expansion=False
    )

    # Step 2: Initialize
    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()
    retriever = create_hybrid_retriever(vector_store, embedder, config)

    logger.info("✓ Retriever initialized")

    # Step 3: Index documents (mock)
    # In real usage, this would load and index actual documents
    logger.info("✓ Documents indexed")

    # Step 4: Search
    query = "Jaké jsou povinnosti dodavatele podle §89?"
    logger.info(f"✓ Searching for: {query}")

    results = await retriever.search(query, top_k=5)

    # Step 5: Analyze results
    logger.info(f"✓ Found {len(results)} results")

    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. {result.chunk.legal_reference} (Score: {result.score:.3f})")
        print(f"   Method: {result.retrieval_method}")
        print(f"   Document: {result.document_id}")
        print(f"   Hierarchy: {result.chunk.hierarchy_path}")
        print(f"   Content: {result.chunk.content[:100]}...")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples"""

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_config),
        ("Filtered Search", example_filtered_search),
        ("Document-Specific Search", example_document_specific),
        ("Adaptive Weighting", example_adaptive_weighting),
        ("Query Expansion", example_query_expansion),
        ("Caching", example_caching),
        ("Score Breakdown", example_score_breakdown),
        ("Batch Search", example_batch_search),
        ("Complete Workflow", example_complete_workflow),
    ]

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL SYSTEM - EXAMPLES")
    print("=" * 80)

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Error in {name}: {e}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
