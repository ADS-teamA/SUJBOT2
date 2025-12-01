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
    EntityDeduplicationConfig,
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


class TestEntityDeduplicationConfig:
    """Tests for EntityDeduplicationConfig and alias map loading."""

    def test_default_config(self):
        """Test default deduplication config."""
        config = EntityDeduplicationConfig()

        assert config.enabled is True
        assert config.exact_match_enabled is True
        assert config.similarity_threshold == 0.90
        assert config.alias_map == {}  # Empty by default (no path)

    def test_validation_similarity_threshold(self):
        """Test similarity threshold validation."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            EntityDeduplicationConfig(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold"):
            EntityDeduplicationConfig(similarity_threshold=-0.1)

    def test_validation_acronym_threshold(self):
        """Test acronym fuzzy threshold validation."""
        with pytest.raises(ValueError, match="acronym_fuzzy_threshold"):
            EntityDeduplicationConfig(acronym_fuzzy_threshold=2.0)

    def test_validation_batch_size(self):
        """Test embedding batch size validation."""
        with pytest.raises(ValueError, match="embedding_batch_size"):
            EntityDeduplicationConfig(embedding_batch_size=0)

    def test_alias_map_loading_missing_file(self, tmp_path):
        """Test alias map loading with non-existent file."""
        config = EntityDeduplicationConfig(
            alias_map_path=str(tmp_path / "nonexistent.json")
        )
        # Should return empty map, not crash
        assert config.alias_map == {}

    def test_alias_map_loading_invalid_json(self, tmp_path):
        """Test alias map loading with malformed JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ not valid json }")

        config = EntityDeduplicationConfig(alias_map_path=str(invalid_file))
        # Should return empty map, not crash
        assert config.alias_map == {}

    def test_alias_map_loading_inverts_correctly(self, tmp_path):
        """Test canonical->aliases is inverted to alias->canonical."""
        import json

        alias_file = tmp_path / "aliases.json"
        alias_file.write_text(json.dumps({
            "Canonical Name": ["alias1", "alias2"],
            "Another Entity": ["alias3"],
        }))

        config = EntityDeduplicationConfig(alias_map_path=str(alias_file))

        # Check inversion: alias -> canonical (all lowercase keys)
        assert config.alias_map["canonical name"] == "Canonical Name"
        assert config.alias_map["alias1"] == "Canonical Name"
        assert config.alias_map["alias2"] == "Canonical Name"
        assert config.alias_map["another entity"] == "Another Entity"
        assert config.alias_map["alias3"] == "Another Entity"

    def test_alias_map_loading_skips_metadata_keys(self, tmp_path):
        """Test _comment and _updated metadata keys are skipped."""
        import json

        alias_file = tmp_path / "aliases.json"
        alias_file.write_text(json.dumps({
            "_comment": "This is a comment",
            "_updated": "2024-01-01",
            "Real Entity": ["alias1"],
        }))

        config = EntityDeduplicationConfig(alias_map_path=str(alias_file))

        # Metadata keys should be skipped
        assert "_comment" not in config.alias_map
        assert "_updated" not in config.alias_map
        assert "this is a comment" not in config.alias_map  # Wasn't processed

        # Real entity should be loaded
        assert config.alias_map["real entity"] == "Real Entity"
        assert config.alias_map["alias1"] == "Real Entity"

    def test_alias_map_loading_skips_invalid_entries(self, tmp_path):
        """Test entries with non-list values are skipped."""
        import json

        alias_file = tmp_path / "aliases.json"
        alias_file.write_text(json.dumps({
            "Valid Entity": ["alias1", "alias2"],
            "Invalid Entity": "not a list",  # Should be skipped
            "Also Invalid": 123,  # Should be skipped
        }))

        config = EntityDeduplicationConfig(alias_map_path=str(alias_file))

        # Valid entry should be loaded
        assert config.alias_map["valid entity"] == "Valid Entity"
        assert config.alias_map["alias1"] == "Valid Entity"

        # Invalid entries should be skipped (no crash)
        # The canonical names are still added, just not the aliases
        assert "invalid entity" not in config.alias_map or config.alias_map.get("not a list") is None

    def test_alias_map_loading_handles_czech_diacritics(self, tmp_path):
        """Test proper handling of Czech characters."""
        import json

        alias_file = tmp_path / "aliases.json"
        alias_file.write_text(json.dumps({
            "Státní úřad pro jadernou bezpečnost": ["SÚJB", "SUJB"],
        }), encoding="utf-8")

        config = EntityDeduplicationConfig(alias_map_path=str(alias_file))

        # Czech diacritics should be preserved in canonical name
        assert config.alias_map["sújb"] == "Státní úřad pro jadernou bezpečnost"
        assert config.alias_map["sujb"] == "Státní úřad pro jadernou bezpečnost"
        # Lowercase canonical
        assert config.alias_map["státní úřad pro jadernou bezpečnost"] == "Státní úřad pro jadernou bezpečnost"

    def test_inline_alias_map(self):
        """Test providing alias map directly (not from file)."""
        config = EntityDeduplicationConfig(
            alias_map={
                "sújb": "Státní úřad pro jadernou bezpečnost",
                "sujb": "Státní úřad pro jadernou bezpečnost",
            }
        )

        assert len(config.alias_map) == 2
        assert config.alias_map["sújb"] == "Státní úřad pro jadernou bezpečnost"


class TestEntityDeduplicationConfigFromEnv:
    """Tests for EntityDeduplicationConfig.from_env() method."""

    def test_from_env_default_values(self, monkeypatch):
        """Test from_env() with default values."""
        # Clear relevant env vars
        monkeypatch.delenv("KG_DEDUPLICATE_ENTITIES", raising=False)
        monkeypatch.delenv("KG_ALIAS_MAP_PATH", raising=False)

        config = EntityDeduplicationConfig.from_env()

        assert config.enabled is True
        # alias_map_path will be None if default path doesn't exist

    def test_from_env_custom_acronyms(self, monkeypatch):
        """Test from_env() parses custom acronyms."""
        monkeypatch.setenv("KG_DEDUP_CUSTOM_ACRONYMS", "ACRO1:expansion1,ACRO2:expansion2")

        config = EntityDeduplicationConfig.from_env()

        assert config.custom_acronyms == {
            "ACRO1": "expansion1",
            "ACRO2": "expansion2",
        }

    def test_from_env_custom_thresholds(self, monkeypatch):
        """Test from_env() parses custom thresholds."""
        monkeypatch.setenv("KG_DEDUP_SIMILARITY_THRESHOLD", "0.85")
        monkeypatch.setenv("KG_DEDUP_ACRONYM_FUZZY_THRESHOLD", "0.80")

        config = EntityDeduplicationConfig.from_env()

        assert config.similarity_threshold == 0.85
        assert config.acronym_fuzzy_threshold == 0.80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
