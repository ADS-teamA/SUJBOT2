#!/usr/bin/env python3
"""
Analyze what search tool returns and whether layer 2 (sections) are useful.

This script:
1. Performs a search using the search tool
2. Shows what hierarchical_search returns (layer1, layer2, layer3)
3. Compares layer2 (section summaries) vs layer3 (chunks) results
4. Evaluates whether layer2 embeddings add value
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from storage.postgres_adapter import PostgresVectorStoreAdapter
from embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def analyze_search_results(query: str = "Jak funguje nakládání s odpady?"):
    """Analyze search results across all layers."""

    # Initialize components
    logger.info("Initializing vector store and embedder...")

    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql://sujbot:sujbot123@localhost:5432/sujbot"
    )

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

    # Perform search
    logger.info(f"\n{'='*80}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*80}\n")

    query_embedding = embedder.embed_texts([query])[0]

    # Get hierarchical search results
    results = await vector_store._async_hierarchical_search(
        query_embedding=query_embedding,
        k_layer3=5,
        use_doc_filtering=False,
        similarity_threshold_offset=0.25,
        query_text=query,
        document_filter=None,
    )

    # Analyze each layer
    print("\n" + "="*80)
    print("LAYER 1 (Document-level)")
    print("="*80)
    for i, doc in enumerate(results["layer1"], 1):
        print(f"\n{i}. Document: {doc['document_id']}")
        print(f"   Score: {doc['score']:.4f}")
        print(f"   Content preview: {doc['content'][:200]}...")

    print("\n" + "="*80)
    print("LAYER 2 (Section-level)")
    print("="*80)
    print(f"Retrieved: {len(results['layer2'])} sections")
    for i, section in enumerate(results["layer2"], 1):
        print(f"\n{i}. Section: {section.get('section_title', 'N/A')}")
        print(f"   Document: {section['document_id']}")
        print(f"   Score: {section['score']:.4f}")
        print(f"   Section ID: {section.get('section_id', 'N/A')}")
        print(f"   Content preview: {section['content'][:200]}...")

    print("\n" + "="*80)
    print("LAYER 3 (Chunk-level) - WHAT SEARCH TOOL ACTUALLY RETURNS")
    print("="*80)
    print(f"Retrieved: {len(results['layer3'])} chunks")
    for i, chunk in enumerate(results["layer3"], 1):
        print(f"\n{i}. Chunk: {chunk['chunk_id']}")
        print(f"   Document: {chunk['document_id']}")
        print(f"   Section: {chunk.get('section_title', 'N/A')}")
        print(f"   Score: {chunk['score']:.4f}")
        print(f"   Content: {chunk['content'][:300]}...")

    # Comparison analysis
    print("\n" + "="*80)
    print("ANALYSIS: Are Layer 2 (section summaries) useful?")
    print("="*80)

    layer2_docs = {r['document_id'] for r in results['layer2']}
    layer3_docs = {r['document_id'] for r in results['layer3']}
    layer2_sections = {(r['document_id'], r.get('section_id')) for r in results['layer2']}
    layer3_sections = {(r['document_id'], r.get('section_id')) for r in results['layer3']}

    print(f"\nDocument coverage:")
    print(f"  - Layer 2 documents: {layer2_docs}")
    print(f"  - Layer 3 documents: {layer3_docs}")
    print(f"  - Overlap: {layer2_docs & layer3_docs}")
    print(f"  - Layer 2 unique: {layer2_docs - layer3_docs}")
    print(f"  - Layer 3 unique: {layer3_docs - layer2_docs}")

    print(f"\nSection coverage:")
    print(f"  - Layer 2 sections: {len(layer2_sections)}")
    print(f"  - Layer 3 sections: {len(layer3_sections)}")
    print(f"  - Overlap: {len(layer2_sections & layer3_sections)}")

    print(f"\nScore comparison:")
    if results['layer2']:
        layer2_avg = sum(r['score'] for r in results['layer2']) / len(results['layer2'])
        print(f"  - Layer 2 average score: {layer2_avg:.4f}")
    if results['layer3']:
        layer3_avg = sum(r['score'] for r in results['layer3']) / len(results['layer3'])
        print(f"  - Layer 3 average score: {layer3_avg:.4f}")

    print("\nKey findings:")
    print("  1. Search tool ONLY returns layer3 (chunks)")
    print("  2. Layer 2 (section summaries) are retrieved but IGNORED")
    print("  3. Layer 2 might provide better document/section context")
    print("  4. But section summaries may be too generic for specific queries")

    # Save detailed results to file
    output_file = Path(__file__).parent.parent / "test_search_layers.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "query": query,
                "layer1_count": len(results["layer1"]),
                "layer2_count": len(results["layer2"]),
                "layer3_count": len(results["layer3"]),
                "layer1": results["layer1"],
                "layer2": results["layer2"],
                "layer3": results["layer3"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nDetailed results saved to: {output_file}")

    await vector_store.close()


async def main():
    """Run analysis with multiple queries."""
    queries = [
        "Jak funguje nakládání s odpady?",
        "Kdo vydal zákon o životním prostředí?",
        "Jaké jsou povinnosti provozovatele?",
    ]

    for query in queries:
        try:
            await analyze_search_results(query)
            print("\n\n")
        except Exception as e:
            logger.error(f"Error analyzing query '{query}': {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
