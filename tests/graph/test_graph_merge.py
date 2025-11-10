"""
Integration tests for knowledge graph merging.

Tests SimpleGraphBuilder.merge() and KnowledgeGraphPipeline.merge_graphs().
"""

import pytest

from src.graph.models import Entity, Relationship, EntityType, RelationshipType
from src.graph.graph_builder import SimpleGraphBuilder
from src.graph.config import GraphStorageConfig, GraphBackend


@pytest.fixture
def target_graph():
    """Create target graph with initial entities and relationships."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)

    # Add entities
    entities = [
        Entity(
            id="e1",
            type=EntityType.PERSON,
            value="John Smith",
            normalized_value="john smith",
            confidence=0.95,
            source_chunk_ids=["chunk1", "chunk2"],
        ),
        Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="Acme Corp",
            normalized_value="acme corp",
            confidence=0.92,
            source_chunk_ids=["chunk1"],
        ),
    ]
    graph.add_entities(entities)

    # Add relationship
    relationships = [
        Relationship(
            id="r1",
            type=RelationshipType.ISSUED_BY,  # Person -> Organization
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.90,
            source_chunk_id="chunk1",
        ),
    ]
    graph.add_relationships(relationships)

    return graph


@pytest.fixture
def source_graph_with_duplicates():
    """Create source graph with some duplicate entities."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)

    # Add entities (e3 is duplicate of e1)
    entities = [
        Entity(
            id="e3",
            type=EntityType.PERSON,
            value="John Smith",  # Duplicate of e1
            normalized_value="john smith",
            confidence=0.90,
            source_chunk_ids=["chunk3"],
        ),
        Entity(
            id="e4",
            type=EntityType.LOCATION,
            value="Prague",
            normalized_value="prague",
            confidence=0.88,
            source_chunk_ids=["chunk3"],
        ),
    ]
    graph.add_entities(entities)

    # Add relationships
    relationships = [
        Relationship(
            id="r2",
            type=RelationshipType.APPLIES_TO,  # Person -> Location
            source_entity_id="e3",  # Points to duplicate entity
            target_entity_id="e4",
            confidence=0.85,
            source_chunk_id="chunk3",
        ),
    ]
    graph.add_relationships(relationships)

    return graph


@pytest.fixture
def source_graph_no_duplicates():
    """Create source graph with all unique entities."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(config)

    # Add unique entities
    entities = [
        Entity(
            id="e5",
            type=EntityType.PERSON,
            value="Jane Doe",
            normalized_value="jane doe",
            confidence=0.93,
            source_chunk_ids=["chunk4"],
        ),
        Entity(
            id="e6",
            type=EntityType.REGULATION,
            value="GDPR Article 5",
            normalized_value="gdpr article 5",
            confidence=0.95,
            source_chunk_ids=["chunk4"],
        ),
    ]
    graph.add_entities(entities)

    # Add relationship
    relationships = [
        Relationship(
            id="r3",
            type=RelationshipType.REFERENCES,
            source_entity_id="e5",
            target_entity_id="e6",
            confidence=0.92,
            source_chunk_id="chunk4",
        ),
    ]
    graph.add_relationships(relationships)

    return graph


def test_merge_no_duplicates(target_graph, source_graph_no_duplicates):
    """Test merging graphs with no duplicate entities."""
    initial_entity_count = len(target_graph.entities)
    initial_rel_count = len(target_graph.relationships)

    # Merge
    stats = target_graph.merge(source_graph_no_duplicates)

    # Check stats
    assert stats["entities_added"] == 2
    assert stats["entities_deduplicated"] == 0
    assert stats["relationships_added"] == 1

    # Check final counts
    assert len(target_graph.entities) == initial_entity_count + 2
    assert len(target_graph.relationships) == initial_rel_count + 1

    # Check entities exist
    assert "e5" in target_graph.entities
    assert "e6" in target_graph.entities


def test_merge_with_duplicates(target_graph, source_graph_with_duplicates):
    """Test merging graphs with duplicate entities."""
    initial_entity_count = len(target_graph.entities)

    # Merge
    stats = target_graph.merge(source_graph_with_duplicates)

    # Check stats
    assert stats["entities_added"] == 1  # Only e4 (Prague) is new
    assert stats["entities_deduplicated"] == 1  # e3 is duplicate of e1
    assert stats["relationships_added"] == 1

    # Check final entity count (only 1 new entity)
    assert len(target_graph.entities) == initial_entity_count + 1

    # Check e3 was mapped to e1
    assert stats["entity_id_remapping"]["e3"] == "e1"

    # Check duplicate entity NOT added
    assert "e3" not in target_graph.entities

    # Check unique entity WAS added
    assert "e4" in target_graph.entities

    # Check source_chunk_ids merged for duplicate
    e1 = target_graph.entities["e1"]
    assert "chunk1" in e1.source_chunk_ids
    assert "chunk2" in e1.source_chunk_ids
    assert "chunk3" in e1.source_chunk_ids  # Merged from e3


def test_merge_relationship_remapping(target_graph, source_graph_with_duplicates):
    """Test that relationships are remapped after entity deduplication."""
    # Merge
    stats = target_graph.merge(source_graph_with_duplicates)

    # Find the merged relationship
    r2 = target_graph.relationships.get("r2")
    assert r2 is not None

    # Check that source_entity_id was remapped from e3 to e1
    assert r2.source_entity_id == "e1"  # Remapped from e3
    assert r2.target_entity_id == "e4"  # No remapping needed


def test_merge_preserves_indexes(target_graph, source_graph_no_duplicates):
    """Test that merge updates all indexes correctly."""
    # Merge
    target_graph.merge(source_graph_no_duplicates)

    # Check entity_by_type index
    persons = target_graph.entity_by_type.get(EntityType.PERSON, set())
    assert "e1" in persons
    assert "e5" in persons

    # Check entity_by_normalized_value index
    jane_key = (EntityType.PERSON, "jane doe")
    assert jane_key in target_graph.entity_by_normalized_value
    assert target_graph.entity_by_normalized_value[jane_key] == "e5"

    # Check relationship indexes
    assert "e5" in target_graph.relationships_by_source
    assert "e6" in target_graph.relationships_by_target


def test_merge_multiple_times(target_graph):
    """Test merging multiple graphs sequentially."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)

    # Create first source graph
    source1 = SimpleGraphBuilder(config)
    source1.add_entities(
        [
            Entity(
                id="e_src1",
                type=EntityType.PERSON,
                value="Alice",
                normalized_value="alice",
                confidence=0.90,
                source_chunk_ids=["chunk_s1"],
            ),
        ]
    )

    # Create second source graph
    source2 = SimpleGraphBuilder(config)
    source2.add_entities(
        [
            Entity(
                id="e_src2",
                type=EntityType.PERSON,
                value="Bob",
                normalized_value="bob",
                confidence=0.88,
                source_chunk_ids=["chunk_s2"],
            ),
        ]
    )

    initial_count = len(target_graph.entities)

    # Merge both
    stats1 = target_graph.merge(source1)
    stats2 = target_graph.merge(source2)

    # Check results
    assert stats1["entities_added"] == 1
    assert stats2["entities_added"] == 1
    assert len(target_graph.entities) == initial_count + 2

    assert "e_src1" in target_graph.entities
    assert "e_src2" in target_graph.entities


def test_merge_empty_graph(target_graph):
    """Test merging an empty graph."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    empty_graph = SimpleGraphBuilder(config)

    initial_count = len(target_graph.entities)

    # Merge empty graph
    stats = target_graph.merge(empty_graph)

    # Should have no changes
    assert stats["entities_added"] == 0
    assert stats["entities_deduplicated"] == 0
    assert stats["relationships_added"] == 0
    assert len(target_graph.entities) == initial_count


def test_merge_self_referential_relationship():
    """Test merging graphs with self-referential relationships."""
    config = GraphStorageConfig(backend=GraphBackend.SIMPLE)

    target = SimpleGraphBuilder(config)
    target.add_entities(
        [
            Entity(
                id="e1",
                type=EntityType.REGULATION,
                value="Article 5",
                normalized_value="article 5",
                confidence=0.95,
                source_chunk_ids=["chunk1"],
            ),
        ]
    )

    source = SimpleGraphBuilder(config)
    source.add_entities(
        [
            Entity(
                id="e2",
                type=EntityType.REGULATION,
                value="Article 5",  # Duplicate
                normalized_value="article 5",
                confidence=0.92,
                source_chunk_ids=["chunk2"],
            ),
        ]
    )
    source.add_relationships(
        [
            Relationship(
                id="r1",
                type=RelationshipType.REFERENCES,  # Self-referential
                source_entity_id="e2",
                target_entity_id="e2",  # Self-referential
                confidence=0.80,
                source_chunk_id="chunk2",
            ),
        ]
    )

    # Merge
    stats = target.merge(source)

    # Check entity was deduplicated
    assert stats["entities_deduplicated"] == 1
    assert stats["entity_id_remapping"]["e2"] == "e1"

    # Check relationship was remapped correctly
    r1 = target.relationships.get("r1")
    assert r1 is not None
    assert r1.source_entity_id == "e1"  # Remapped
    assert r1.target_entity_id == "e1"  # Remapped


def test_merge_updates_statistics(target_graph, source_graph_no_duplicates):
    """Test that merge updates graph statistics correctly."""
    # Get stats before merge
    stats_before = target_graph.get_statistics()

    # Merge
    target_graph.merge(source_graph_no_duplicates)

    # Get stats after merge
    stats_after = target_graph.get_statistics()

    # Check entity count increased
    assert stats_after["total_entities"] > stats_before["total_entities"]
    assert stats_after["total_relationships"] > stats_before["total_relationships"]

    # Check entity type counts
    assert EntityType.PERSON.value in stats_after["entity_type_counts"]
    assert EntityType.REGULATION.value in stats_after["entity_type_counts"]
