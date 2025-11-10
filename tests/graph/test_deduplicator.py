"""
Unit tests for EntityDeduplicator.

Tests entity deduplication logic for graph merging.
"""

import pytest

from src.graph.deduplicator import EntityDeduplicator
from src.graph.models import Entity, EntityType
from src.graph.graph_builder import SimpleGraphBuilder
from src.graph.config import GraphStorageConfig, GraphBackend


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            id="e1",
            type=EntityType.PERSON,
            value="John Smith",
            normalized_value="john smith",
            confidence=0.95,
            source_chunk_ids=["chunk1"],
        ),
        Entity(
            id="e2",
            type=EntityType.PERSON,
            value="John Smith",  # Same person, different ID
            normalized_value="john smith",
            confidence=0.90,
            source_chunk_ids=["chunk2"],
        ),
        Entity(
            id="e3",
            type=EntityType.ORGANIZATION,
            value="Acme Corp",
            normalized_value="acme corp",
            confidence=0.92,
            source_chunk_ids=["chunk1"],
        ),
        Entity(
            id="e4",
            type=EntityType.PERSON,
            value="Jane Doe",
            normalized_value="jane doe",
            confidence=0.88,
            source_chunk_ids=["chunk3"],
        ),
    ]


@pytest.fixture
def graph_with_entities(sample_entities):
    """Create graph with sample entities."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)

    # Add first 2 entities
    graph.add_entities([sample_entities[0], sample_entities[2]])

    return graph


def test_deduplicator_initialization():
    """Test EntityDeduplicator initialization."""
    dedup = EntityDeduplicator()

    assert dedup.use_exact_match is True
    assert dedup.similarity_threshold == 0.98


def test_deduplicator_custom_threshold():
    """Test EntityDeduplicator with custom threshold."""
    dedup = EntityDeduplicator(similarity_threshold=0.95)

    assert dedup.similarity_threshold == 0.95


def test_find_duplicate_exact_match(sample_entities, graph_with_entities):
    """Test finding exact duplicate entity."""
    dedup = EntityDeduplicator()

    # e2 is duplicate of e1 (same normalized_value)
    duplicate_id = dedup.find_duplicate(sample_entities[1], graph_with_entities)

    assert duplicate_id is not None
    assert duplicate_id == "e1"


def test_find_duplicate_no_match(sample_entities, graph_with_entities):
    """Test finding no duplicate for unique entity."""
    dedup = EntityDeduplicator()

    # e4 is unique (Jane Doe)
    duplicate_id = dedup.find_duplicate(sample_entities[3], graph_with_entities)

    assert duplicate_id is None


def test_find_duplicate_different_type(sample_entities, graph_with_entities):
    """Test that different entity types are not considered duplicates."""
    dedup = EntityDeduplicator()

    # Create ORGANIZATION with same name as PERSON
    org_entity = Entity(
        id="e5",
        type=EntityType.ORGANIZATION,
        value="John Smith",  # Same name as person
        normalized_value="john smith",
        confidence=0.90,
        source_chunk_ids=["chunk4"],
    )

    # Should not find duplicate (different type)
    duplicate_id = dedup.find_duplicate(org_entity, graph_with_entities)

    assert duplicate_id is None


def test_find_duplicate_case_insensitive():
    """Test that duplicate detection is case-insensitive."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)
    dedup = EntityDeduplicator()

    # Add entity with lowercase
    entity1 = Entity(
        id="e1",
        type=EntityType.PERSON,
        value="John Smith",
        normalized_value="john smith",
        confidence=0.95,
        source_chunk_ids=["chunk1"],
    )
    graph.add_entities([entity1])

    # Check uppercase variant
    entity2 = Entity(
        id="e2",
        type=EntityType.PERSON,
        value="JOHN SMITH",
        normalized_value="john smith",  # normalized to lowercase
        confidence=0.90,
        source_chunk_ids=["chunk2"],
    )

    duplicate_id = dedup.find_duplicate(entity2, graph)

    assert duplicate_id == "e1"


def test_find_duplicate_whitespace_normalized():
    """Test that duplicate detection handles whitespace normalization."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)
    dedup = EntityDeduplicator()

    # Add entity with normalized whitespace
    entity1 = Entity(
        id="e1",
        type=EntityType.PERSON,
        value="John  Smith",  # Double space
        normalized_value="john smith",  # Single space after normalization
        confidence=0.95,
        source_chunk_ids=["chunk1"],
    )
    graph.add_entities([entity1])

    # Check variant with different whitespace
    entity2 = Entity(
        id="e2",
        type=EntityType.PERSON,
        value="John Smith",  # Single space
        normalized_value="john smith",
        confidence=0.90,
        source_chunk_ids=["chunk2"],
    )

    duplicate_id = dedup.find_duplicate(entity2, graph)

    assert duplicate_id == "e1"


def test_get_stats():
    """Test getting deduplication statistics."""
    dedup = EntityDeduplicator(use_exact_match=True, similarity_threshold=0.95)

    stats = dedup.get_stats()

    assert stats["strategy"] == "exact_match"
    assert stats["threshold"] == 0.95


def test_similarity_matching_not_implemented():
    """Test that similarity matching is not yet implemented."""
    dedup = EntityDeduplicator(use_exact_match=False)

    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)

    entity = Entity(
        id="e1",
        type=EntityType.PERSON,
        value="John Smith",
        normalized_value="john smith",
        confidence=0.95,
        source_chunk_ids=["chunk1"],
    )

    with pytest.raises(NotImplementedError, match="Non-exact matching not yet implemented"):
        dedup.find_duplicate(entity, graph)
