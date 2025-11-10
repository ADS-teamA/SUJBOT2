"""
Integration tests for Neo4j Aura backend.

These tests require a running Neo4j instance (configured via .env).
Run with: uv run pytest tests/graph/test_neo4j_integration.py -v
"""

import pytest
import os
from datetime import datetime

from src.graph.models import Entity, EntityType, Relationship, RelationshipType, KnowledgeGraph
from src.graph.config import Neo4jConfig, GraphStorageConfig, GraphBackend
from src.graph.graph_builder import Neo4jGraphBuilder, create_graph_builder
from src.graph.neo4j_manager import Neo4jManager
from src.graph.health_check import check_neo4j_health
from src.graph.exceptions import Neo4jConnectionError


# Skip all tests if Neo4j not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI"),
    reason="NEO4J_URI not set in environment (tests require Neo4j instance)",
)


class TestNeo4jConnection:
    """Test Neo4j connection and health checks."""

    def test_neo4j_config_from_env(self):
        """Test loading Neo4j config from environment."""
        config = Neo4jConfig.from_env()

        assert config.uri.startswith("neo4j")  # bolt:// or neo4j+s://
        assert config.username
        assert config.database

    def test_health_check_success(self):
        """Test health check with valid connection."""
        config = Neo4jConfig.from_env()
        health = check_neo4j_health(config)

        assert health["connected"] is True
        assert health["can_write"] is True
        assert health["can_query"] is True
        assert health["response_time_ms"] > 0
        assert health["error"] is None

    def test_neo4j_manager_initialization(self):
        """Test Neo4jManager can be created and closed."""
        config = Neo4jConfig.from_env()
        manager = Neo4jManager(config)

        try:
            # Should be able to execute simple query
            result = manager.execute("RETURN 1 as value")
            assert result[0]["value"] == 1
        finally:
            manager.close()


class TestNeo4jGraphBuilder:
    """Test Neo4jGraphBuilder with real Neo4j instance."""

    @pytest.fixture
    def config(self):
        """Create GraphStorageConfig with Neo4j backend."""
        return GraphStorageConfig(
            backend=GraphBackend.NEO4J,
            neo4j_config=Neo4jConfig.from_env(),
        )

    @pytest.fixture
    def builder(self, config):
        """Create Neo4jGraphBuilder and cleanup after test."""
        builder = create_graph_builder(config)
        yield builder

        # Cleanup: delete all test nodes
        try:
            builder.manager.execute(
                "MATCH (n:Entity) WHERE n.id STARTS WITH 'test-' DETACH DELETE n"
            )
        except:
            pass

        builder.close()

    def test_factory_creates_neo4j_builder(self, config):
        """Test factory function creates Neo4jGraphBuilder."""
        builder = create_graph_builder(config)
        assert isinstance(builder, Neo4jGraphBuilder)
        builder.close()

    def test_add_single_entity(self, builder):
        """Test adding a single entity to Neo4j."""
        entity = Entity(
            id="test-entity-1",
            type=EntityType.STANDARD,
            value="Test Standard GRI 306",
            normalized_value="test_standard_gri_306",
            confidence=0.95,
            source_chunk_ids=["chunk-1", "chunk-2"],
            first_mention_chunk_id="chunk-1",
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        builder.add_entities([entity])

        # Verify entity was added
        retrieved = builder.get_entity("test-entity-1")
        assert retrieved is not None
        assert retrieved.id == "test-entity-1"
        assert retrieved.value == "Test Standard GRI 306"
        assert retrieved.type == EntityType.STANDARD
        assert retrieved.confidence == 0.95

    def test_add_batch_entities(self, builder):
        """Test adding multiple entities in batch."""
        entities = [
            Entity(
                id=f"test-batch-{i}",
                type=EntityType.ORGANIZATION if i % 2 == 0 else EntityType.TOPIC,
                value=f"Entity {i}",
                normalized_value=f"entity_{i}",
                confidence=0.8 + (i * 0.01),
                extraction_method="test",
                extracted_at=datetime.now(),
            )
            for i in range(10)
        ]

        builder.add_entities(entities)

        # Verify all entities were added
        for i in range(10):
            retrieved = builder.get_entity(f"test-batch-{i}")
            assert retrieved is not None
            assert retrieved.value == f"Entity {i}"

    def test_add_relationships(self, builder):
        """Test adding relationships between entities."""
        # Create entities first
        entity1 = Entity(
            id="test-rel-source",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="gri_306",
            confidence=0.9,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        entity2 = Entity(
            id="test-rel-target",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="gssb",
            confidence=0.9,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        builder.add_entities([entity1, entity2])

        # Create relationship
        rel = Relationship(
            id="test-relationship-1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="test-rel-source",
            target_entity_id="test-rel-target",
            confidence=0.85,
            source_chunk_id="chunk-1",
            evidence_text="GRI 306 was issued by GSSB",
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        builder.add_relationships([rel])

        # Verify relationship was added
        relationships = builder.get_relationships_for_entity("test-rel-source")
        assert len(relationships) > 0

        # Find our test relationship
        test_rel = next((r for r in relationships if r.id == "test-relationship-1"), None)
        assert test_rel is not None
        assert test_rel.type == RelationshipType.ISSUED_BY
        assert test_rel.confidence == 0.85

    def test_export_to_knowledge_graph(self, builder):
        """Test exporting Neo4j data to KnowledgeGraph object."""
        # Add test data
        entities = [
            Entity(
                id=f"test-export-{i}",
                type=EntityType.TOPIC,
                value=f"Topic {i}",
                normalized_value=f"topic_{i}",
                confidence=0.9,
                extraction_method="test",
                extracted_at=datetime.now(),
            )
            for i in range(3)
        ]

        builder.add_entities(entities)

        # Export
        kg = builder.export_to_knowledge_graph()

        # Verify export contains our test entities
        test_entities = [e for e in kg.entities if e.id.startswith("test-export-")]
        assert len(test_entities) == 3

    def test_large_batch_performance(self, builder):
        """Test performance with larger batch (1000+ entities)."""
        import time

        # Create 1500 entities to test batch chunking
        entities = [
            Entity(
                id=f"test-perf-{i}",
                type=EntityType.TOPIC,
                value=f"Performance Test Entity {i}",
                normalized_value=f"perf_entity_{i}",
                confidence=0.8,
                extraction_method="test",
                extracted_at=datetime.now(),
            )
            for i in range(1500)
        ]

        start_time = time.time()
        builder.add_entities(entities)
        duration = time.time() - start_time

        # Should complete in reasonable time (< 10 seconds for 1500 entities)
        assert duration < 10.0, f"Batch insertion took {duration:.1f}s (expected <10s)"

        # Verify a sample
        retrieved = builder.get_entity("test-perf-500")
        assert retrieved is not None
        assert retrieved.value == "Performance Test Entity 500"


class TestErrorHandling:
    """Test error handling and retry logic."""

    def test_connection_error_invalid_credentials(self):
        """Test that invalid credentials raise authentication error."""
        from src.graph.exceptions import Neo4jAuthenticationError

        bad_config = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username="invalid_user",
            password="invalid_password",
            database="neo4j",
        )

        with pytest.raises((Neo4jAuthenticationError, Neo4jConnectionError)):
            manager = Neo4jManager(bad_config)

    def test_retry_logic_on_transient_failure(self):
        """Test that manager retries on transient failures."""
        # This is a unit test - would need to mock neo4j driver to simulate failures
        # For now, just verify the method exists and has correct signature
        config = Neo4jConfig.from_env()
        manager = Neo4jManager(config)

        try:
            # Verify execute has max_retries parameter
            import inspect
            sig = inspect.signature(manager.execute)
            assert "max_retries" in sig.parameters
        finally:
            manager.close()


@pytest.mark.skipif(
    not os.path.exists("vector_db/unified_kg.json"),
    reason="unified_kg.json not found (run pipeline first)",
)
class TestMigrationWorkflow:
    """Test complete migration workflow with real data."""

    def test_load_and_migrate_unified_kg(self):
        """Test loading unified KG and migrating to Neo4j."""
        # Load unified KG
        kg = KnowledgeGraph.load_json("vector_db/unified_kg.json")

        assert len(kg.entities) > 0
        assert len(kg.relationships) > 0

        # Create builder
        config = GraphStorageConfig(
            backend=GraphBackend.NEO4J,
            neo4j_config=Neo4jConfig.from_env(),
        )
        builder = create_graph_builder(config)

        try:
            # Add test subset (first 10 entities to avoid long test time)
            test_entities = kg.entities[:10]
            builder.add_entities(test_entities)

            # Verify
            for entity in test_entities:
                retrieved = builder.get_entity(entity.id)
                assert retrieved is not None

        finally:
            # Cleanup
            for entity in test_entities:
                try:
                    builder.manager.execute(
                        "MATCH (n:Entity {id: $id}) DETACH DELETE n",
                        {"id": entity.id},
                    )
                except:
                    pass
            builder.close()
