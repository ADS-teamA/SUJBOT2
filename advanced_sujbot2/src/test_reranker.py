"""
Test/Demo script for the reranking module.

This script demonstrates how to use the reranking pipeline
with mock data.

Usage:
    python test_reranker.py
"""

import asyncio
import logging
from typing import List

from reranker import (
    RerankingConfig,
    RerankingPipeline,
    SearchResult,
    CrossEncoderReranker,
    LegalPrecedenceReranker,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_mock_results() -> List[SearchResult]:
    """Create mock search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk_001",
            content="§89 odst. 1: Dodavatel je povinen plnit smluvní podmínky...",
            legal_reference="§89 odst. 1",
            document_id="zakon_89_2012",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89",
            rank=1,
            hybrid_score=0.85,
            metadata={"effective_date": "2012-04-01"}
        ),
        SearchResult(
            chunk_id="chunk_002",
            content="Smlouva stanovuje, že dodavatel musí dodat zboží...",
            legal_reference="Článek 5.1",
            document_id="contract_2024_001",
            document_type="contract",
            hierarchy_path="Článek 5",
            rank=2,
            hybrid_score=0.78,
            metadata={"effective_date": "2024-01-01"}
        ),
        SearchResult(
            chunk_id="chunk_003",
            content="§90: Objednatel má právo kontrolovat plnění...",
            legal_reference="§90",
            document_id="zakon_89_2012",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §90",
            rank=3,
            hybrid_score=0.72,
            metadata={"effective_date": "2012-04-01"}
        ),
        SearchResult(
            chunk_id="chunk_004",
            content="Vyhláška č. 123/2013 Sb. specifikuje postupy dodání...",
            legal_reference="§5",
            document_id="vyhlaska_123_2013",
            document_type="regulation",
            hierarchy_path="§5",
            rank=4,
            hybrid_score=0.65,
            metadata={"effective_date": "2013-06-01"}
        ),
        SearchResult(
            chunk_id="chunk_005",
            content="Metodika doporučuje, aby dodavatelé zohlednili...",
            legal_reference="Bod 3.2",
            document_id="metodika_001",
            document_type="guidance",
            hierarchy_path="Kapitola 3 > Bod 3.2",
            rank=5,
            hybrid_score=0.58,
            metadata={"effective_date": "2020-01-01"}
        ),
    ]


async def test_cross_encoder():
    """Test cross-encoder reranking only."""
    print("\n" + "="*80)
    print("TEST 1: Cross-Encoder Reranking")
    print("="*80 + "\n")

    config = RerankingConfig(
        cross_encoder_device="cpu",
        cross_encoder_batch_size=8
    )

    reranker = CrossEncoderReranker(config)
    results = create_mock_results()

    query = "Jaké jsou povinnosti dodavatele podle zákona?"

    print(f"Query: {query}\n")
    print(f"Reranking {len(results)} results...\n")

    reranked = await reranker.rerank(query, results)

    print("Results after cross-encoder reranking:\n")
    for i, (result, score) in enumerate(reranked, 1):
        print(f"{i}. [{result.legal_reference}] Score: {score:.3f}")
        print(f"   Original rank: {result.rank} → New rank: {i}")
        print(f"   Content: {result.content[:80]}...")
        print()


async def test_legal_precedence():
    """Test legal precedence reranking."""
    print("\n" + "="*80)
    print("TEST 2: Legal Precedence Reranking")
    print("="*80 + "\n")

    config = RerankingConfig(
        enable_precedence_weighting=True,
        precedence_weights={
            "constitutional": 1.0,
            "statutory": 0.9,
            "regulatory": 0.7,
            "contractual": 0.5,
            "guidance": 0.3
        }
    )

    reranker = LegalPrecedenceReranker(config)
    results = create_mock_results()

    print(f"Reranking {len(results)} results by legal precedence...\n")

    reranked = await reranker.rerank(results)

    print("Results after precedence reranking:\n")
    for i, (result, score, factors) in enumerate(reranked, 1):
        print(f"{i}. [{result.legal_reference}] ({result.document_type})")
        print(f"   Precedence score: {score:.3f}")
        print(f"   Factors: hierarchy={factors['hierarchy']:.2f}, "
              f"temporal={factors['temporal']:.2f}, "
              f"specificity={factors['specificity']:.2f}")
        print(f"   Original rank: {result.rank} → New rank: {i}")
        print()


async def test_full_pipeline():
    """Test full reranking pipeline (without graph)."""
    print("\n" + "="*80)
    print("TEST 3: Full Reranking Pipeline")
    print("="*80 + "\n")

    config = RerankingConfig(
        cross_encoder_device="cpu",
        cross_encoder_batch_size=8,
        enable_graph_reranking=False,  # No graph for this test
        enable_precedence_weighting=True,
        ensemble_method="weighted_average",
        ensemble_weights={
            "cross_encoder": 0.6,
            "graph": 0.0,
            "precedence": 0.4
        },
        final_top_k=3,
        min_confidence_threshold=0.0,
        explain_reranking=True
    )

    pipeline = RerankingPipeline(config, knowledge_graph=None)
    results = create_mock_results()

    query = "Jaké jsou povinnosti dodavatele podle zákona?"

    print(f"Query: {query}\n")
    print(f"Initial results: {len(results)}")
    print(f"Configuration:")
    print(f"  - Cross-encoder weight: {config.ensemble_weights['cross_encoder']}")
    print(f"  - Precedence weight: {config.ensemble_weights['precedence']}")
    print(f"  - Top-K: {config.final_top_k}\n")

    ranked_results = await pipeline.rerank(query, results)

    print(f"\nFinal Results (top-{len(ranked_results)}):\n")
    for result in ranked_results:
        print(f"Rank {result.final_rank}: [{result.legal_reference}] ({result.document_type})")
        print(f"  Ensemble Score: {result.scores.ensemble_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Component Scores:")
        print(f"    - Cross-encoder: {result.scores.cross_encoder_score:.3f}")
        print(f"    - Graph: {result.scores.graph_score:.3f}")
        print(f"    - Precedence: {result.scores.precedence_score:.3f}")
        print(f"  Rank Change: {result.original_rank} → {result.final_rank} "
              f"({'+' if result.rank_improvement > 0 else ''}{result.rank_improvement})")
        print(f"  Explanation: {result.reranking_explanation}")
        print(f"  Content: {result.content[:100]}...")
        print()


async def test_fusion_methods():
    """Compare different fusion methods."""
    print("\n" + "="*80)
    print("TEST 4: Fusion Method Comparison")
    print("="*80 + "\n")

    results = create_mock_results()
    query = "Jaké jsou povinnosti dodavatele?"

    methods = ["weighted_average", "borda_count", "rrf"]

    for method in methods:
        print(f"\nFusion Method: {method}")
        print("-" * 40)

        config = RerankingConfig(
            cross_encoder_device="cpu",
            enable_graph_reranking=False,
            enable_precedence_weighting=True,
            ensemble_method=method,
            final_top_k=3,
            explain_reranking=False
        )

        pipeline = RerankingPipeline(config, knowledge_graph=None)
        ranked = await pipeline.rerank(query, results)

        for i, result in enumerate(ranked, 1):
            print(f"{i}. {result.legal_reference}: score={result.scores.ensemble_score:.3f}, "
                  f"confidence={result.confidence:.3f}")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RERANKING MODULE TESTS")
    print("="*80)

    try:
        await test_cross_encoder()
        await test_legal_precedence()
        await test_full_pipeline()
        await test_fusion_methods()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
