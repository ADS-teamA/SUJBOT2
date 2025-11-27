"""
Tests for PHASE 5D: Graph-Vector Integration

Tests cover:
1. Entity extraction from queries
2. Graph boosting by entity mentions
3. Graph boosting by centrality
4. Triple-modal fusion (Dense + Sparse + Graph)
5. Multi-hop query expansion
"""

import pytest
import numpy as np
from unittest.mock import Mock
from pathlib import Path


# Test fixtures
@pytest.fixture
def sample_knowledge_graph():
    """Create sample knowledge graph for testing."""
    from src.graph.models import Entity, Relationship, KnowledgeGraph, EntityType, RelationshipType

    # Create entities
    entities = [
        Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.95,
            source_chunk_ids=["chunk_1", "chunk_3"],
            first_mention_chunk_id="chunk_1",
        ),
        Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="GSSB",
            confidence=0.90,
            source_chunk_ids=["chunk_2"],
            first_mention_chunk_id="chunk_2",
        ),
        Entity(
            id="e3",
            type=EntityType.TOPIC,
            value="waste management",
            normalized_value="waste management",
            confidence=0.85,
            source_chunk_ids=["chunk_1", "chunk_4"],
            first_mention_chunk_id="chunk_1",
        ),
    ]

    # Create relationships
    relationships = [
        Relationship(
            id="r1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="e1",  # GRI 306
            target_entity_id="e2",  # GSSB
            confidence=0.90,
            source_chunk_id="chunk_2",
            evidence_text="GRI 306 issued by GSSB",
        ),
        Relationship(
            id="r2",
            type=RelationshipType.COVERS_TOPIC,
            source_entity_id="e1",  # GRI 306
            target_entity_id="e3",  # waste management
            confidence=0.85,
            source_chunk_id="chunk_1",
            evidence_text="GRI 306 covers waste management",
        ),
    ]

    kg = KnowledgeGraph(entities=entities, relationships=relationships)
    kg.compute_stats()

    return kg


@pytest.fixture
def sample_hybrid_results():
    """Sample results from hybrid search."""
    return {
        "layer3": [
            {
                "chunk_id": "chunk_1",
                "content": "GRI 306 standard covers waste management practices.",
                "rrf_score": 0.032,
                "document_id": "doc_1",
            },
            {
                "chunk_id": "chunk_2",
                "content": "The standard was issued by GSSB in 2020.",
                "rrf_score": 0.029,
                "document_id": "doc_1",
            },
            {
                "chunk_id": "chunk_3",
                "content": "GRI 306 replaces the previous version from 2016.",
                "rrf_score": 0.027,
                "document_id": "doc_1",
            },
            {
                "chunk_id": "chunk_4",
                "content": "Waste management includes disposal and recycling.",
                "rrf_score": 0.025,
                "document_id": "doc_1",
            },
            {
                "chunk_id": "chunk_5",
                "content": "Safety equipment requirements are specified.",
                "rrf_score": 0.020,
                "document_id": "doc_1",
            },
        ]
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = Mock()
    return mock


# Test: EntityAwareSearch Initialization
def test_entity_aware_search_initialization(sample_knowledge_graph):
    """Test EntityAwareSearch initializes correctly."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    assert searcher.kg == sample_knowledge_graph
    assert searcher.llm_model == "gpt-4o-mini"


# Test: Entity Extraction from Query
def test_extract_query_entities_exact_match(sample_knowledge_graph):
    """Test entity extraction with exact matches."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    # Query mentions "GRI 306"
    query = "What are the requirements of GRI 306 standard?"
    matched_entities = searcher.extract_query_entities(query)

    # Should find GRI 306 entity
    assert len(matched_entities) > 0
    assert matched_entities[0]["entity"].normalized_value == "GRI 306"
    assert matched_entities[0]["match_type"] == "exact"
    assert matched_entities[0]["confidence"] > 0


def test_extract_query_entities_multiple_matches(sample_knowledge_graph):
    """Test extraction with multiple entity mentions."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    # Query mentions both GRI 306 and waste management
    query = "GRI 306 covers waste management topics"
    matched_entities = searcher.extract_query_entities(query)

    # Should find both entities
    assert len(matched_entities) >= 2

    entity_values = [e["entity"].normalized_value for e in matched_entities]
    assert "GRI 306" in entity_values
    assert "waste management" in entity_values


def test_extract_query_entities_no_matches(sample_knowledge_graph):
    """Test extraction with no matching entities."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    query = "completely unrelated query about weather"
    matched_entities = searcher.extract_query_entities(query)

    assert len(matched_entities) == 0


# Test: Get Entity Chunks
def test_get_entity_chunks(sample_knowledge_graph):
    """Test retrieving chunks mentioning an entity."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    # Get chunks for GRI 306 (e1)
    chunk_ids = searcher.get_entity_chunks("e1")

    assert len(chunk_ids) == 2
    assert "chunk_1" in chunk_ids
    assert "chunk_3" in chunk_ids


def test_get_entity_chunks_nonexistent_entity(sample_knowledge_graph):
    """Test retrieving chunks for non-existent entity."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    chunk_ids = searcher.get_entity_chunks("nonexistent")

    assert len(chunk_ids) == 0


# Test: Multi-Hop Relationships
def test_get_related_entity_chunks_1hop(sample_knowledge_graph):
    """Test 1-hop relationship traversal."""
    from src.graph_retrieval import EntityAwareSearch

    searcher = EntityAwareSearch(sample_knowledge_graph)

    # Get chunks related to GRI 306 (1-hop)
    chunk_ids = searcher.get_related_entity_chunks("e1", max_depth=1)

    # Should include chunks from GSSB and waste management
    # GSSB (e2) → chunk_2
    # waste management (e3) → chunk_1, chunk_4
    assert "chunk_2" in chunk_ids  # From GSSB
    assert "chunk_1" in chunk_ids or "chunk_4" in chunk_ids  # From waste management


# Test: GraphBooster Initialization
def test_graph_booster_initialization(sample_knowledge_graph):
    """Test GraphBooster initializes and computes centrality."""
    from src.graph_retrieval import GraphBooster

    booster = GraphBooster(sample_knowledge_graph)

    assert booster.kg == sample_knowledge_graph
    assert len(booster.entity_centrality) > 0

    # GRI 306 (e1) has 2 relationships (highest centrality)
    assert booster.entity_centrality["e1"] == 1.0  # Normalized max


def test_entity_centrality_calculation(sample_knowledge_graph):
    """Test centrality scores are calculated correctly."""
    from src.graph_retrieval import GraphBooster

    booster = GraphBooster(sample_knowledge_graph)

    # e1 (GRI 306): 2 relationships → centrality 1.0
    # e2 (GSSB): 1 relationship → centrality 0.5
    # e3 (waste mgmt): 1 relationship → centrality 0.5

    assert booster.entity_centrality["e1"] == 1.0
    assert booster.entity_centrality["e2"] == 0.5
    assert booster.entity_centrality["e3"] == 0.5


# Test: Boost by Entity Mentions
def test_boost_by_entity_mentions(sample_knowledge_graph, sample_hybrid_results):
    """Test boosting chunks that mention query entities."""
    from src.graph_retrieval import GraphBooster

    booster = GraphBooster(sample_knowledge_graph)

    # Extract query entities
    query_entities = [{"entity": sample_knowledge_graph.entities[0], "confidence": 0.9}]  # GRI 306

    # Boost results
    boosted = booster.boost_by_entity_mentions(
        chunk_results=sample_hybrid_results["layer3"],
        query_entities=query_entities,
        boost_weight=0.3,
    )

    # chunk_1 and chunk_3 mention GRI 306, should be boosted
    chunk_1 = next(c for c in boosted if c["chunk_id"] == "chunk_1")
    chunk_5 = next(c for c in boosted if c["chunk_id"] == "chunk_5")

    assert chunk_1["graph_boost"] == 0.3
    assert chunk_1["boosted_score"] > chunk_1["rrf_score"]

    assert chunk_5["graph_boost"] == 0.0
    assert chunk_5["boosted_score"] == chunk_5["rrf_score"]


def test_boost_by_entity_mentions_reorders_results(sample_knowledge_graph, sample_hybrid_results):
    """Test that boosting reorders results correctly."""
    from src.graph_retrieval import GraphBooster

    booster = GraphBooster(sample_knowledge_graph)

    # Extract GRI 306 entity
    query_entities = [{"entity": sample_knowledge_graph.entities[0], "confidence": 0.9}]

    original_order = [c["chunk_id"] for c in sample_hybrid_results["layer3"]]

    # Boost (chunk_1 and chunk_3 mention GRI 306)
    boosted = booster.boost_by_entity_mentions(
        chunk_results=sample_hybrid_results["layer3"],
        query_entities=query_entities,
        boost_weight=0.3,
    )

    new_order = [c["chunk_id"] for c in boosted]

    # Order should change
    assert new_order != original_order

    # Boosted chunks should move to top
    assert "chunk_1" in new_order[:2] or "chunk_3" in new_order[:2]


# Test: Boost by Centrality
def test_boost_by_centrality(sample_knowledge_graph, sample_hybrid_results):
    """Test boosting by entity centrality."""
    from src.graph_retrieval import GraphBooster

    booster = GraphBooster(sample_knowledge_graph)

    boosted = booster.boost_by_centrality(
        chunk_results=sample_hybrid_results["layer3"], boost_weight=0.2
    )

    # chunk_1 mentions GRI 306 (centrality 1.0) → boost = 0.2
    chunk_1 = next(c for c in boosted if c["chunk_id"] == "chunk_1")

    assert chunk_1["centrality_boost"] == 0.2
    assert "boosted_score" in chunk_1


# Test: GraphEnhancedRetriever Initialization
def test_graph_enhanced_retriever_initialization(mock_vector_store, sample_knowledge_graph):
    """Test GraphEnhancedRetriever initializes correctly."""
    from src.graph_retrieval import GraphEnhancedRetriever

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph
    )

    assert retriever.vector_store == mock_vector_store
    assert retriever.kg == sample_knowledge_graph
    assert retriever.entity_search is not None
    assert retriever.graph_booster is not None


def test_graph_enhanced_retriever_with_custom_config(mock_vector_store, sample_knowledge_graph):
    """Test retriever with custom configuration."""
    from src.graph_retrieval import GraphEnhancedRetriever, GraphRetrievalConfig

    config = GraphRetrievalConfig(
        enable_graph_boost=True, graph_boost_weight=0.5, enable_multi_hop=True, max_hop_depth=2
    )

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph, config=config
    )

    assert retriever.config.graph_boost_weight == 0.5
    assert retriever.config.enable_multi_hop == True
    assert retriever.config.max_hop_depth == 2


# Test: Graph-Enhanced Search
def test_graph_enhanced_search_without_boost(
    mock_vector_store, sample_knowledge_graph, sample_hybrid_results
):
    """Test search without graph boosting (baseline)."""
    from src.graph_retrieval import GraphEnhancedRetriever

    # Mock vector store to return sample results
    mock_vector_store.hierarchical_search.return_value = sample_hybrid_results

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph
    )

    query = "GRI 306 requirements"
    query_embedding = np.array([0.1] * 3072)

    results = retriever.search(
        query=query, query_embedding=query_embedding, k=3, enable_graph_boost=False
    )

    # Should just return vector results truncated to k
    assert len(results["layer3"]) == 3

    # Mock should be called with k*2 candidates
    mock_vector_store.hierarchical_search.assert_called_once()


def test_graph_enhanced_search_with_boost(
    mock_vector_store, sample_knowledge_graph, sample_hybrid_results
):
    """Test search with graph boosting enabled."""
    from src.graph_retrieval import GraphEnhancedRetriever, GraphRetrievalConfig

    # Mock vector store
    mock_vector_store.hierarchical_search.return_value = sample_hybrid_results

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store,
        knowledge_graph=sample_knowledge_graph,
        config=GraphRetrievalConfig(graph_boost_weight=0.3),
    )

    query = "GRI 306 requirements"
    query_embedding = np.array([0.1] * 3072)

    results = retriever.search(
        query=query, query_embedding=query_embedding, k=3, enable_graph_boost=True
    )

    # Should apply graph boosting
    assert len(results["layer3"]) == 3

    # Check that chunks have boost scores
    for chunk in results["layer3"]:
        assert "graph_boost" in chunk or "boosted_score" in chunk


def test_graph_boost_weight_respected(
    mock_vector_store, sample_knowledge_graph, sample_hybrid_results
):
    """Graph boost weight from config should control entity boost magnitude."""
    from src.graph_retrieval import GraphEnhancedRetriever, GraphRetrievalConfig

    mock_vector_store.hierarchical_search.return_value = sample_hybrid_results

    config = GraphRetrievalConfig(graph_boost_weight=0.5)
    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph, config=config
    )

    query = "Explain GRI 306 waste rules"
    query_embedding = np.array([0.2] * 3072)

    results = retriever.search(
        query=query, query_embedding=query_embedding, k=3, enable_graph_boost=True
    )

    boosted_chunk = next(c for c in results["layer3"] if c.get("graph_boost", 0) > 0)
    assert boosted_chunk["graph_boost"] == pytest.approx(0.5)


# Test: Statistics
def test_get_stats(mock_vector_store, sample_knowledge_graph):
    """Test statistics retrieval."""
    from src.graph_retrieval import GraphEnhancedRetriever

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph
    )

    stats = retriever.get_stats()

    assert "vector_store_type" in stats
    assert stats["kg_entities"] == 3
    assert stats["kg_relationships"] == 2
    assert "config" in stats


# Test: Multi-Hop Expansion
def test_multi_hop_expansion(mock_vector_store, sample_knowledge_graph, sample_hybrid_results):
    """Test multi-hop graph traversal (placeholder test)."""
    from src.graph_retrieval import GraphEnhancedRetriever, GraphRetrievalConfig

    config = GraphRetrievalConfig(enable_multi_hop=True, max_hop_depth=2)

    mock_vector_store.hierarchical_search.return_value = sample_hybrid_results

    retriever = GraphEnhancedRetriever(
        vector_store=mock_vector_store, knowledge_graph=sample_knowledge_graph, config=config
    )

    query = "What topics are covered by standards issued by GSSB?"
    query_embedding = np.array([0.1] * 3072)

    results = retriever.search(
        query=query, query_embedding=query_embedding, k=3, enable_graph_boost=True
    )

    # Multi-hop expansion is implemented (logged but not yet affecting results)
    # This test verifies it doesn't crash
    assert results is not None


# Integration Test
@pytest.mark.integration
def test_full_graph_retrieval_pipeline():
    """
    Integration test: Full pipeline with graph-enhanced retrieval

    Requires: Actual vector store, KG, embeddings
    Skipped in unit tests.
    """
    pytest.skip("Integration test - requires actual models and data")

    # Example integration flow:
    # 1. hybrid_store.hierarchical_search(k=12)
    # 2. graph_retriever.search(with boosting)
    # 3. Return boosted results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
