"""
Unit tests for Knowledge Graph configuration.

Tests configuration classes and preset configs.
"""

import pytest

from src.graph.config import (
    EntityExtractionConfig,
    RelationshipExtractionConfig,
    Neo4jConfig,
    GraphStorageConfig,
    KnowledgeGraphConfig,
    GraphBackend,
    get_default_config,
    get_development_config,
    get_production_config,
)
from src.graph.models import EntityType, RelationshipType


class TestEntityExtractionConfig:
    """Tests for EntityExtractionConfig."""

    def test_default_config(self):
        """Test default entity extraction config."""
        config = EntityExtractionConfig()

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.min_confidence == 0.6
        assert config.batch_size == 20  # Optimized: 2× faster

    def test_enabled_entity_types(self):
        """Test enabled entity types."""
        config = EntityExtractionConfig()

        assert EntityType.STANDARD in config.enabled_entity_types
        assert EntityType.ORGANIZATION in config.enabled_entity_types
        assert EntityType.TOPIC in config.enabled_entity_types

    def test_custom_config(self):
        """Test custom entity extraction config."""
        config = EntityExtractionConfig(
            llm_model="claude-haiku",
            min_confidence=0.7,
            batch_size=5,
            enabled_entity_types={EntityType.STANDARD, EntityType.ORGANIZATION},
        )

        assert config.llm_model == "claude-haiku"
        assert config.min_confidence == 0.7
        assert config.batch_size == 5
        assert len(config.enabled_entity_types) == 2


class TestRelationshipExtractionConfig:
    """Tests for RelationshipExtractionConfig."""

    def test_default_config(self):
        """Test default relationship extraction config."""
        config = RelationshipExtractionConfig()

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.min_confidence == 0.5
        assert config.extract_within_chunk is True

    def test_enabled_relationship_types(self):
        """Test enabled relationship types."""
        config = RelationshipExtractionConfig()

        assert RelationshipType.SUPERSEDED_BY in config.enabled_relationship_types
        assert RelationshipType.REFERENCES in config.enabled_relationship_types
        assert RelationshipType.ISSUED_BY in config.enabled_relationship_types


class TestNeo4jConfig:
    """Tests for Neo4j configuration."""

    def test_default_config(self):
        """Test default Neo4j config."""
        config = Neo4jConfig()

        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.database == "neo4j"

    def test_custom_config(self):
        """Test custom Neo4j config."""
        config = Neo4jConfig(
            uri="bolt://example.com:7687",
            username="admin",
            password="secret",
        )

        assert config.uri == "bolt://example.com:7687"
        assert config.username == "admin"
        assert config.password == "secret"


class TestGraphStorageConfig:
    """Tests for GraphStorageConfig."""

    def test_default_config(self):
        """Test default graph storage config."""
        config = GraphStorageConfig()

        assert config.backend == GraphBackend.SIMPLE
        assert config.export_json is True
        assert config.deduplicate_entities is True

    def test_neo4j_config(self):
        """Test Neo4j graph storage config."""
        neo4j_config = Neo4jConfig()
        config = GraphStorageConfig(
            backend=GraphBackend.NEO4J,
            neo4j_config=neo4j_config,
        )

        assert config.backend == GraphBackend.NEO4J
        assert config.neo4j_config is not None


class TestKnowledgeGraphConfig:
    """Tests for KnowledgeGraphConfig."""

    def test_default_config(self):
        """Test default knowledge graph config."""
        config = KnowledgeGraphConfig()

        assert config.enable_entity_extraction is True
        assert config.enable_relationship_extraction is True
        assert config.entity_extraction is not None
        assert config.relationship_extraction is not None
        assert config.graph_storage is not None

    def test_custom_config(self):
        """Test custom knowledge graph config."""
        entity_config = EntityExtractionConfig(llm_model="gpt-4o")
        rel_config = RelationshipExtractionConfig(llm_model="gpt-4o")
        storage_config = GraphStorageConfig(backend=GraphBackend.SIMPLE)

        config = KnowledgeGraphConfig(
            entity_extraction=entity_config,
            relationship_extraction=rel_config,
            graph_storage=storage_config,
            openai_api_key="test-key",
        )

        assert config.entity_extraction.llm_model == "gpt-4o"
        assert config.relationship_extraction.llm_model == "gpt-4o"
        assert config.openai_api_key == "test-key"

    def test_validate_success(self):
        """Test validation with valid config."""
        config = KnowledgeGraphConfig(
            openai_api_key="test-key",
        )

        # Should not raise
        config.validate()

    def test_validate_missing_api_key(self):
        """Test validation with missing API key."""
        config = KnowledgeGraphConfig()

        # Should raise ValueError
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            config.validate()

    def test_validate_invalid_confidence(self):
        """Test validation with invalid confidence threshold."""
        config = KnowledgeGraphConfig(
            entity_extraction=EntityExtractionConfig(min_confidence=1.5),
            openai_api_key="test-key",
        )

        with pytest.raises(ValueError, match="min_confidence"):
            config.validate()

    def test_validate_neo4j_missing_config(self):
        """Test validation when Neo4j backend but no config."""
        config = KnowledgeGraphConfig(
            graph_storage=GraphStorageConfig(
                backend=GraphBackend.NEO4J,
                neo4j_config=None,
            ),
            openai_api_key="test-key",
        )

        with pytest.raises(ValueError, match="neo4j_config"):
            config.validate()


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test get_default_config."""
        config = get_default_config()

        assert isinstance(config, KnowledgeGraphConfig)
        assert config.graph_storage.backend == GraphBackend.SIMPLE

    def test_development_config(self):
        """Test get_development_config."""
        config = get_development_config()

        assert isinstance(config, KnowledgeGraphConfig)
        assert config.entity_extraction.llm_model == "gpt-4o-mini"
        assert config.entity_extraction.batch_size == 5
        assert config.graph_storage.backend == GraphBackend.SIMPLE

    def test_production_config(self):
        """Test get_production_config."""
        config = get_production_config()

        assert isinstance(config, KnowledgeGraphConfig)
        assert config.entity_extraction.llm_model == "gpt-4o"
        assert config.entity_extraction.batch_size == 20
        assert config.graph_storage.backend == GraphBackend.NEO4J


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
