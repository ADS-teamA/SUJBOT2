"""
Comprehensive tests for 3-layer entity deduplication system.

Tests:
- EntityDeduplicationConfig
- EntitySimilarityDetector (Layer 2)
- AcronymExpander (Layer 3)
- Enhanced EntityDeduplicator with all 3 layers
- Neo4jDeduplicator (with mocked Neo4j)
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.graph.config import EntityDeduplicationConfig, GraphBackend, GraphStorageConfig
from src.graph.graph_builder import SimpleGraphBuilder
from src.graph.models import Entity, EntityType

# ==================== EntityDeduplicationConfig Tests ====================


def test_config_default_initialization():
    """Test default EntityDeduplicationConfig initialization."""
    config = EntityDeduplicationConfig()

    assert config.enabled is True
    assert config.exact_match_enabled is True
    assert config.use_embeddings is False
    assert config.use_acronym_expansion is False
    assert config.similarity_threshold == 0.90
    assert config.acronym_fuzzy_threshold == 0.85
    assert config.apoc_enabled is True
    assert config.create_uniqueness_constraints is True


def test_config_custom_initialization():
    """Test EntityDeduplicationConfig with custom settings."""
    config = EntityDeduplicationConfig(
        enabled=True,
        use_embeddings=True,
        similarity_threshold=0.95,
        use_acronym_expansion=True,
        acronym_fuzzy_threshold=0.90,
        custom_acronyms={"SUJBOT": "system for unified job bot"},
    )

    assert config.enabled is True
    assert config.use_embeddings is True
    assert config.similarity_threshold == 0.95
    assert config.use_acronym_expansion is True
    assert config.acronym_fuzzy_threshold == 0.90
    assert "SUJBOT" in config.custom_acronyms


def test_config_from_env_defaults(monkeypatch):
    """Test loading config from environment with defaults."""
    # Clear relevant env vars
    for key in [
        "KG_DEDUPLICATE_ENTITIES",
        "KG_DEDUP_USE_EMBEDDINGS",
        "KG_DEDUP_SIMILARITY_THRESHOLD",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = EntityDeduplicationConfig.from_env()

    assert config.enabled is True  # Default
    assert config.use_embeddings is False  # Default


def test_config_from_env_custom(monkeypatch):
    """Test loading config from environment with custom values."""
    monkeypatch.setenv("KG_DEDUPLICATE_ENTITIES", "true")
    monkeypatch.setenv("KG_DEDUP_USE_EMBEDDINGS", "true")
    monkeypatch.setenv("KG_DEDUP_SIMILARITY_THRESHOLD", "0.92")
    monkeypatch.setenv("KG_DEDUP_USE_ACRONYM_EXPANSION", "true")
    monkeypatch.setenv("KG_DEDUP_ACRONYM_FUZZY_THRESHOLD", "0.88")
    monkeypatch.setenv("KG_DEDUP_CUSTOM_ACRONYMS", "ACRO1:expansion1,ACRO2:expansion2")

    config = EntityDeduplicationConfig.from_env()

    assert config.enabled is True
    assert config.use_embeddings is True
    assert config.similarity_threshold == 0.92
    assert config.use_acronym_expansion is True
    assert config.acronym_fuzzy_threshold == 0.88
    assert "ACRO1" in config.custom_acronyms
    assert config.custom_acronyms["ACRO1"] == "expansion1"


# ==================== EntitySimilarityDetector Tests ====================


@pytest.fixture
def mock_embedder():
    """Create mock EmbeddingGenerator."""
    embedder = Mock()

    # Mock embed_texts to return normalized random vectors (batch API)
    def embed_texts(texts):
        results = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(768)
            results.append(vec / np.linalg.norm(vec))  # Normalized
        return results

    embedder.embed_texts = Mock(side_effect=embed_texts)
    return embedder


def test_similarity_detector_initialization(mock_embedder):
    """Test EntitySimilarityDetector initialization."""
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(similarity_threshold=0.90, cache_embeddings=True)
    detector = EntitySimilarityDetector(mock_embedder, config)

    assert detector.similarity_threshold == 0.90
    assert detector.cache_enabled is True
    assert len(detector._embedding_cache) == 0


def test_similarity_detector_find_similar_high_similarity(mock_embedder):
    """Test finding similar entity with high similarity."""
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(similarity_threshold=0.70)
    detector = EntitySimilarityDetector(mock_embedder, config)

    # Create entities with same normalized text (will have same embedding)
    entity1 = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.95,
    )

    entity2 = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.90,
    )

    candidates = [entity1]

    match_id = detector.find_similar(entity2, candidates, EntityType.STANDARD)

    # Should match (same text → same embedding → cosine similarity = 1.0)
    assert match_id == "e1"
    assert detector.stats["matches_found"] == 1


def test_similarity_detector_find_similar_no_match(mock_embedder):
    """Test finding no similar entity when similarity too low."""
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(similarity_threshold=0.95)
    detector = EntitySimilarityDetector(mock_embedder, config)

    entity1 = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.95,
    )

    entity2 = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="Completely Different Text",
        normalized_value="completely different text",
        confidence=0.90,
    )

    candidates = [entity1]

    match_id = detector.find_similar(entity2, candidates, EntityType.STANDARD)

    # Should not match (different text → low similarity)
    assert match_id is None


def test_similarity_detector_type_filtering(mock_embedder):
    """Test that similarity detector filters by entity type."""
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(similarity_threshold=0.70)
    detector = EntitySimilarityDetector(mock_embedder, config)

    # Same text but different types
    entity_org = Entity(
        id="e1",
        type=EntityType.ORGANIZATION,
        value="ISO",
        normalized_value="iso",
        confidence=0.95,
    )

    entity_standard = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="ISO",
        normalized_value="iso",
        confidence=0.90,
    )

    candidates = [entity_org]

    # Search for STANDARD type should not match ORGANIZATION
    match_id = detector.find_similar(entity_standard, candidates, EntityType.STANDARD)

    assert match_id is None


def test_similarity_detector_caching(mock_embedder):
    """Test embedding caching functionality."""
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(cache_embeddings=True)
    detector = EntitySimilarityDetector(mock_embedder, config)

    # First call - cache miss
    embedding1 = detector._get_embedding("test text")
    assert detector.stats["cache_misses"] == 1
    assert detector.stats["cache_hits"] == 0

    # Second call - cache hit
    embedding2 = detector._get_embedding("test text")
    assert detector.stats["cache_misses"] == 1
    assert detector.stats["cache_hits"] == 1

    # Should be same embedding
    assert np.array_equal(embedding1, embedding2)


# ==================== AcronymExpander Tests ====================


def test_acronym_expander_initialization():
    """Test AcronymExpander initialization."""
    from src.graph.acronym_expander import AcronymExpander

    config = EntityDeduplicationConfig(
        acronym_fuzzy_threshold=0.85, custom_acronyms={"SUJBOT": "system for unified job bot"}
    )

    expander = AcronymExpander(config)

    assert expander.fuzzy_threshold == 0.85
    assert "GRI" in expander.acronyms  # Built-in
    assert "SUJBOT" in expander.acronyms  # Custom
    assert expander.acronyms["SUJBOT"] == "system for unified job bot"


def test_acronym_expander_extract_acronyms():
    """Test extracting acronyms from text."""
    from src.graph.acronym_expander import AcronymExpander

    config = EntityDeduplicationConfig()
    expander = AcronymExpander(config)

    acronyms = expander._extract_acronyms("ISO 14001 and GRI standards by UN")

    assert "ISO" in acronyms
    assert "GRI" in acronyms
    assert "UN" in acronyms
    assert len(acronyms) == 3


def test_acronym_expander_expand_entity():
    """Test expanding entity with acronyms."""
    from src.graph.acronym_expander import AcronymExpander

    config = EntityDeduplicationConfig()
    expander = AcronymExpander(config)

    expanded_forms = expander._expand_entity("GRI 306")

    # Should include original and expanded form
    assert "gri 306" in expanded_forms
    assert "global reporting initiative 306" in expanded_forms


def test_acronym_expander_find_acronym_match_exact():
    """Test finding acronym match with exact expansion."""
    from src.graph.acronym_expander import AcronymExpander

    # Lower threshold since SequenceMatcher similarity is low for acronym expansions
    config = EntityDeduplicationConfig(acronym_fuzzy_threshold=0.20)
    expander = AcronymExpander(config)

    entity_acronym = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="GRI 306",
        normalized_value="gri 306",
        confidence=0.95,
    )

    entity_expanded = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="Global Reporting Initiative 306",
        normalized_value="global reporting initiative 306",
        confidence=0.90,
    )

    candidates = [entity_expanded]

    match_id = expander.find_acronym_match(entity_acronym, candidates)

    # Should match via acronym expansion
    assert match_id == "e2"


def test_acronym_expander_find_acronym_match_fuzzy():
    """Test finding acronym match with fuzzy matching."""
    from src.graph.acronym_expander import AcronymExpander

    # Lower threshold for acronym expansion fuzzy matching
    config = EntityDeduplicationConfig(acronym_fuzzy_threshold=0.15)
    expander = AcronymExpander(config)

    entity1 = Entity(
        id="e1",
        type=EntityType.REGULATION,
        value="GDPR Compliance",
        normalized_value="gdpr compliance",
        confidence=0.95,
    )

    entity2 = Entity(
        id="e2",
        type=EntityType.REGULATION,
        value="General Data Protection Regulation Compliance",
        normalized_value="general data protection regulation compliance",
        confidence=0.90,
    )

    candidates = [entity2]

    match_id = expander.find_acronym_match(entity1, candidates)

    # Should match via GDPR expansion + fuzzy match
    assert match_id == "e2"


def test_acronym_expander_no_match_different_type():
    """Test that acronym expander respects entity types."""
    from src.graph.acronym_expander import AcronymExpander

    config = EntityDeduplicationConfig()
    expander = AcronymExpander(config)

    entity_reg = Entity(
        id="e1",
        type=EntityType.REGULATION,
        value="ISO",
        normalized_value="iso",
        confidence=0.95,
    )

    entity_org = Entity(
        id="e2",
        type=EntityType.ORGANIZATION,
        value="International Organization for Standardization",
        normalized_value="international organization for standardization",
        confidence=0.90,
    )

    candidates = [entity_org]

    # Different types should not match
    match_id = expander.find_acronym_match(entity_reg, candidates)

    assert match_id is None


# ==================== Enhanced EntityDeduplicator Tests ====================


def test_enhanced_deduplicator_initialization():
    """Test enhanced EntityDeduplicator initialization."""
    from src.graph.deduplicator import EntityDeduplicator

    config = EntityDeduplicationConfig(
        exact_match_enabled=True, use_embeddings=True, use_acronym_expansion=True
    )

    dedup = EntityDeduplicator(config=config)

    assert dedup.config.exact_match_enabled is True
    assert dedup.config.use_embeddings is True
    assert dedup.config.use_acronym_expansion is True


def test_enhanced_deduplicator_layer1_exact_match():
    """Test Layer 1 exact match deduplication."""
    from src.graph.deduplicator import EntityDeduplicator

    config = EntityDeduplicationConfig(exact_match_enabled=True, use_embeddings=False)
    dedup = EntityDeduplicator(config=config)

    graph_config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(graph_config)

    entity1 = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.95,
    )

    entity2 = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.90,
    )

    graph.add_entities([entity1])

    dup_id = dedup.find_duplicate(entity2, graph)

    # Should find exact match
    assert dup_id == "e1"
    assert dedup.stats["layer1_matches"] == 1


def test_enhanced_deduplicator_layer2_semantic(mock_embedder):
    """Test Layer 2 semantic similarity deduplication."""
    from src.graph.deduplicator import EntityDeduplicator
    from src.graph.similarity_detector import EntitySimilarityDetector

    config = EntityDeduplicationConfig(
        exact_match_enabled=False,  # Disable Layer 1
        use_embeddings=True,
        similarity_threshold=0.70,
    )

    similarity_detector = EntitySimilarityDetector(mock_embedder, config)
    dedup = EntityDeduplicator(config=config, similarity_detector=similarity_detector)

    graph_config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(graph_config)

    entity1 = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="GRI 306",
        normalized_value="gri 306",
        confidence=0.95,
    )

    entity2 = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="GRI 306",
        normalized_value="gri 306",
        confidence=0.90,
    )

    graph.add_entities([entity1])

    dup_id = dedup.find_duplicate(entity2, graph)

    # Should find semantic match
    assert dup_id == "e1"
    assert dedup.stats["layer2_matches"] == 1


def test_enhanced_deduplicator_layer3_acronym():
    """Test Layer 3 acronym expansion deduplication."""
    from src.graph.acronym_expander import AcronymExpander
    from src.graph.deduplicator import EntityDeduplicator

    config = EntityDeduplicationConfig(
        exact_match_enabled=False,  # Disable Layer 1
        use_embeddings=False,  # Disable Layer 2
        use_acronym_expansion=True,
        acronym_fuzzy_threshold=0.20,  # Lower threshold for acronym matching
    )

    acronym_expander = AcronymExpander(config)
    dedup = EntityDeduplicator(config=config, acronym_expander=acronym_expander)

    graph_config = GraphStorageConfig(backend=GraphBackend.SIMPLE)
    graph = SimpleGraphBuilder(graph_config)

    entity_expanded = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="Global Reporting Initiative 306",
        normalized_value="global reporting initiative 306",
        confidence=0.95,
    )

    entity_acronym = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="GRI 306",
        normalized_value="gri 306",
        confidence=0.90,
    )

    graph.add_entities([entity_expanded])

    dup_id = dedup.find_duplicate(entity_acronym, graph)

    # Should find acronym match
    assert dup_id == "e1"
    assert dedup.stats["layer3_matches"] == 1


def test_enhanced_deduplicator_merge_properties():
    """Test merging entity properties."""
    from src.graph.deduplicator import EntityDeduplicator

    dedup = EntityDeduplicator()

    primary = Entity(
        id="e1",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.92,
        source_chunk_ids=["chunk1", "chunk2"],
        first_mention_chunk_id="chunk1",
        metadata={"version": "2015"},
    )

    duplicate = Entity(
        id="e2",
        type=EntityType.STANDARD,
        value="ISO 14001",
        normalized_value="iso 14001",
        confidence=0.95,  # Higher confidence
        source_chunk_ids=["chunk3"],  # New chunk
        metadata={"document_id": "doc2"},
    )

    merged = dedup.merge_entity_properties(primary, duplicate)

    # Confidence should be MAX
    assert merged.confidence == 0.95

    # Chunks should be UNION
    assert set(merged.source_chunk_ids) == {"chunk1", "chunk2", "chunk3"}

    # First mention should be preserved
    assert merged.first_mention_chunk_id == "chunk1"

    # Metadata should be merged
    assert merged.metadata["version"] == "2015"
    assert merged.metadata["document_id"] == "doc2"
    assert "e2" in merged.metadata["merged_from"]


# ==================== Neo4jDeduplicator Tests ====================


@pytest.fixture
def mock_neo4j_manager():
    """Create mock Neo4jManager."""
    manager = Mock()
    manager.execute = Mock(return_value=[{"created_count": 5, "merged_count": 3}])
    return manager


def test_neo4j_deduplicator_initialization(mock_neo4j_manager):
    """Test Neo4jDeduplicator initialization."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig(apoc_enabled=True)

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=True) as mock_check_apoc:
        dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

        assert dedup.manager == mock_neo4j_manager
        assert dedup.config == config
        assert dedup.apoc_available is True
        mock_check_apoc.assert_called_once()


def test_neo4j_deduplicator_check_apoc_available(mock_neo4j_manager):
    """Test APOC availability check when available."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    mock_neo4j_manager.execute = Mock(return_value=[{"version": "5.0.0"}])

    config = EntityDeduplicationConfig()
    dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

    assert dedup._check_apoc() is True


def test_neo4j_deduplicator_check_apoc_unavailable(mock_neo4j_manager):
    """Test APOC availability check when unavailable."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    mock_neo4j_manager.execute = Mock(side_effect=Exception("APOC not found"))

    config = EntityDeduplicationConfig()
    dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

    assert dedup._check_apoc() is False


def test_neo4j_deduplicator_add_entities_with_apoc(mock_neo4j_manager):
    """Test adding entities with APOC."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig()

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=True):
        dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

        entities = [
            Entity(
                id=f"e{i}",
                type=EntityType.STANDARD,
                value=f"Entity {i}",
                normalized_value=f"entity {i}",
                confidence=0.9,
            )
            for i in range(10)
        ]

        stats = dedup.add_entities_with_dedup(entities, batch_size=5)

        assert stats["entities_added"] == 10  # 5 + 5
        assert stats["entities_merged"] == 6  # 3 + 3
        assert mock_neo4j_manager.execute.call_count == 2  # 2 batches


def test_neo4j_deduplicator_add_entities_without_apoc(mock_neo4j_manager):
    """Test adding entities without APOC (pure Cypher fallback)."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig()

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=False):
        dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

        entities = [
            Entity(
                id=f"e{i}",
                type=EntityType.STANDARD,
                value=f"Entity {i}",
                normalized_value=f"entity {i}",
                confidence=0.9,
            )
            for i in range(10)
        ]

        stats = dedup.add_entities_with_dedup(entities, batch_size=5)

        assert stats["entities_added"] == 10
        assert stats["entities_merged"] == 6
        assert mock_neo4j_manager.execute.call_count == 2


def test_neo4j_deduplicator_entity_to_dict():
    """Test converting Entity to dict for Neo4j."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig()
    manager = Mock()

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=False):
        dedup = Neo4jDeduplicator(manager, config)

        entity = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="ISO 14001",
            normalized_value="iso 14001",
            confidence=0.95,
            source_chunk_ids=["chunk1", "chunk2"],
            first_mention_chunk_id="chunk1",
            document_id="doc1",
            section_path=["Section 1"],
            extraction_method="llm",
        )

        entity_dict = dedup._entity_to_dict(entity)

        assert entity_dict["id"] == "e1"
        assert entity_dict["type"] == "standard"
        assert entity_dict["value"] == "ISO 14001"
        assert entity_dict["normalized_value"] == "iso 14001"
        assert entity_dict["confidence"] == 0.95
        assert entity_dict["source_chunk_ids"] == ["chunk1", "chunk2"]
        assert entity_dict["document_id"] == "doc1"


def test_neo4j_deduplicator_create_constraints(mock_neo4j_manager):
    """Test creating uniqueness constraints."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig(create_uniqueness_constraints=True)

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=False):
        dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

        dedup.create_uniqueness_constraints()

        # Should execute constraint creation query
        assert mock_neo4j_manager.execute.called


def test_neo4j_deduplicator_skip_constraints(mock_neo4j_manager):
    """Test skipping constraint creation when disabled."""
    from src.graph.neo4j_deduplicator import Neo4jDeduplicator

    config = EntityDeduplicationConfig(create_uniqueness_constraints=False)

    with patch.object(Neo4jDeduplicator, "_check_apoc", return_value=False):
        dedup = Neo4jDeduplicator(mock_neo4j_manager, config)

        dedup.create_uniqueness_constraints()

        # Should not execute any queries
        assert not mock_neo4j_manager.execute.called
