"""
Tests for TIER 3 Analysis Tools.

Tests all 6 analysis tools:
1. ExplainEntityTool
2. GetEntityRelationshipsTool
3. TimelineViewTool
4. SummarizeSectionTool
5. GetStatisticsTool
6. GetIndexStatisticsTool (NEW in Phase 7B)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from collections import Counter

from src.agent.tools.tier3_analysis import (
    # ExplainEntityTool,  # REMOVED - functionality moved to GraphSearchTool (tier 2)
    # GetEntityRelationshipsTool,  # REMOVED - functionality moved to GraphSearchTool (tier 2)
    TimelineViewTool,
    SummarizeSectionTool,
    GetStatsTool,  # Was GetStatisticsTool
    # GetIndexStatisticsTool,  # TODO: Check if this exists
)
from src.agent.tools.base import ToolResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = Mock()
    embedder.embed_texts.return_value = [[0.1] * 10]  # 10D embedding
    embedder.model_name = "test-embedding-model"
    embedder.dimensions = 10
    embedder.model_type = "test"

    # Mock cache stats for GetIndexStatisticsTool
    embedder.get_cache_stats.return_value = {
        "hits": 42,
        "misses": 58,
        "hit_rate": 0.42,
        "size": 100,
    }

    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    vs = Mock()

    # Default hierarchical_search returns empty results
    vs.hierarchical_search.return_value = {
        "layer1": [],
        "layer2": [],
        "layer3": [],
    }

    # get_stats method
    vs.get_stats.return_value = {
        "layer1_count": 10,
        "layer2_count": 50,
        "layer3_count": 200,
        "hybrid_enabled": True,
    }

    return vs


@pytest.fixture
def mock_entity():
    """Create mock entity."""
    entity = Mock()
    entity.id = "entity_1"
    entity.name = "GRI 306"
    entity.type = "STANDARD"
    entity.properties = {"version": "2016"}
    entity.confidence = 0.95
    return entity


@pytest.fixture
def mock_relationship():
    """Create mock relationship."""
    rel = Mock()
    rel.id = "rel_1"
    rel.type = "REFERENCES"
    rel.source = "entity_1"
    rel.target = "entity_2"
    rel.source_entity_id = "entity_1"
    rel.target_entity_id = "entity_2"
    rel.confidence = 0.90
    rel.properties = {}
    return rel


@pytest.fixture
def mock_knowledge_graph(mock_entity, mock_relationship):
    """Create mock knowledge graph with dict interface."""
    kg = Mock()

    # Entities as dict (matching tool expectations)
    entity1 = Mock()
    entity1.id = "entity_1"
    entity1.name = "GRI 306"
    entity1.type = "STANDARD"
    entity1.properties = {}
    entity1.confidence = 0.95

    entity2 = Mock()
    entity2.id = "entity_2"
    entity2.name = "GRI 303"
    entity2.type = "STANDARD"
    entity2.properties = {}
    entity2.confidence = 0.90

    kg.entities = {
        "entity_1": entity1,
        "entity_2": entity2,
        "gri_306": entity1,  # Allow lookup by name (lowercase)
    }

    # Relationships as dict (matching tool expectations)
    rel1 = Mock()
    rel1.id = "rel_1"
    rel1.type = "REFERENCES"
    rel1.source = "entity_1"
    rel1.target = "entity_2"
    rel1.source_entity_id = "entity_1"
    rel1.target_entity_id = "entity_2"
    rel1.confidence = 0.90
    rel1.properties = {}

    kg.relationships = {
        "rel_1": rel1,
    }

    # Methods
    kg.get_outgoing_relationships.return_value = [rel1]
    kg.get_incoming_relationships.return_value = []

    return kg


@pytest.fixture
def mock_context_assembler():
    """Create mock context assembler."""
    assembler = Mock()
    assembled = Mock()
    assembled.context = "Formatted section content..."
    assembler.assemble.return_value = assembled
    return assembler


@pytest.fixture
def base_tool_kwargs(
    mock_vector_store, mock_embedder, mock_knowledge_graph, mock_context_assembler
):
    """Common kwargs for tool initialization."""
    return {
        "vector_store": mock_vector_store,
        "embedder": mock_embedder,
        "knowledge_graph": mock_knowledge_graph,
        "context_assembler": mock_context_assembler,
        "config": Mock(),
    }


# ============================================================================
# Test ExplainEntityTool
# ============================================================================


class TestExplainEntityTool:
    """Test ExplainEntityTool."""

    def test_explain_entity_by_id(self, base_tool_kwargs, mock_vector_store):
        """Test explaining entity by ID."""
        # Setup
        tool = ExplainEntityTool(**base_tool_kwargs)

        # Mock vector store to return chunks
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "content": "GRI 306 covers waste disposal...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "chunk_id": "chunk1",
                    "score": 0.95,
                }
            ]
        }

        # Execute
        result = tool.execute(entity_id="entity_1")

        # Assert
        assert result.success
        assert result.data["entity_id"] == "entity_1"
        assert result.data["name"] == "GRI 306"
        assert result.data["type"] == "STANDARD"
        assert len(result.data["outgoing_relationships"]) == 1
        assert result.data["mention_count"] >= 1

    def test_explain_entity_by_name(self, base_tool_kwargs, mock_vector_store):
        """Test explaining entity by name (case-insensitive)."""
        # Setup
        tool = ExplainEntityTool(**base_tool_kwargs)
        kg = base_tool_kwargs["knowledge_graph"]

        # Create mock entities dict that supports lookup
        entity1 = kg.entities["entity_1"]
        entity2 = kg.entities["entity_2"]

        # Replace entities dict with one that supports iteration
        entities_dict = {
            "entity_1": entity1,
            "entity_2": entity2,
        }

        # Mock __contains__ to return False for non-exact matches
        kg.entities = Mock()
        kg.entities.__contains__ = Mock(return_value=False)
        kg.entities.__getitem__ = Mock(side_effect=lambda k: entities_dict.get(k, KeyError(k)))
        kg.entities.items = Mock(return_value=list(entities_dict.items()))

        # Mock vector store to return chunks
        mock_vector_store.hierarchical_search.return_value = {"layer3": []}

        # Keep the same relationship mocks
        kg.get_outgoing_relationships.return_value = []
        kg.get_incoming_relationships.return_value = []

        # Execute (should find by name case-insensitive)
        result = tool.execute(entity_id="GRI 306")

        # Assert - should find by name
        assert result.success
        assert result.data["name"] == "GRI 306"

    def test_explain_entity_not_found(self, base_tool_kwargs):
        """Test explaining non-existent entity."""
        # Setup
        tool = ExplainEntityTool(**base_tool_kwargs)
        kg = base_tool_kwargs["knowledge_graph"]

        # Replace entities dict with empty mock
        kg.entities = Mock()
        kg.entities.__contains__ = Mock(return_value=False)
        kg.entities.items = Mock(return_value=[])

        # Execute
        result = tool.execute(entity_id="nonexistent")

        # Assert
        assert not result.success
        assert "not found" in result.error.lower()

    def test_explain_entity_no_kg(self, base_tool_kwargs):
        """Test graceful degradation when KG not available."""
        # Setup
        base_tool_kwargs["knowledge_graph"] = None
        tool = ExplainEntityTool(**base_tool_kwargs)

        # Execute
        result = tool.execute(entity_id="entity_1")

        # Assert
        assert not result.success
        assert "not available" in result.error.lower()


# ============================================================================
# Test GetEntityRelationshipsTool
# ============================================================================


class TestGetEntityRelationshipsTool:
    """Test GetEntityRelationshipsTool."""

    def test_get_relationships_both_directions(self, base_tool_kwargs):
        """Test getting relationships in both directions."""
        # Setup
        tool = GetEntityRelationshipsTool(**base_tool_kwargs)

        # Execute
        result = tool.execute(entity_id="entity_1", direction="both")

        # Assert
        assert result.success
        assert result.data["entity_name"] == "GRI 306"
        assert result.data["direction"] == "both"
        assert len(result.data["relationships"]) == 1
        assert result.data["relationships"][0]["direction"] == "outgoing"

    def test_get_relationships_outgoing_only(self, base_tool_kwargs):
        """Test getting outgoing relationships only."""
        # Setup
        tool = GetEntityRelationshipsTool(**base_tool_kwargs)

        # Execute
        result = tool.execute(entity_id="entity_1", direction="outgoing")

        # Assert
        assert result.success
        assert result.data["direction"] == "outgoing"
        assert all(r["direction"] == "outgoing" for r in result.data["relationships"])

    def test_get_relationships_with_type_filter(self, base_tool_kwargs):
        """Test filtering relationships by type."""
        # Setup
        tool = GetEntityRelationshipsTool(**base_tool_kwargs)
        kg = base_tool_kwargs["knowledge_graph"]

        # Mock filtered relationships
        ref_rel = kg.relationships["rel_1"]
        ref_rel.type = "REFERENCES"
        kg.get_outgoing_relationships.return_value = [ref_rel]

        # Execute
        result = tool.execute(
            entity_id="entity_1", relationship_type="REFERENCES", direction="outgoing"
        )

        # Assert
        assert result.success
        assert result.data["relationship_type_filter"] == "REFERENCES"

    def test_get_relationships_no_kg(self, base_tool_kwargs):
        """Test graceful degradation when KG not available."""
        # Setup
        base_tool_kwargs["knowledge_graph"] = None
        tool = GetEntityRelationshipsTool(**base_tool_kwargs)

        # Execute
        result = tool.execute(entity_id="entity_1")

        # Assert
        assert not result.success
        assert "not available" in result.error.lower()


# ============================================================================
# Test TimelineViewTool
# ============================================================================


class TestTimelineViewTool:
    """Test TimelineViewTool."""

    def test_timeline_view_with_dates(self, base_tool_kwargs, mock_vector_store):
        """Test extracting timeline from chunks with dates."""
        # Setup
        tool = TimelineViewTool(**base_tool_kwargs)

        # Mock chunks with dates
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "content": "Standard effective from 2018-07-01...",
                    "raw_content": "Standard effective from 2018-07-01...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Effective Date",
                    "chunk_id": "chunk1",
                },
                {
                    "content": "Published on January 15, 2018...",
                    "raw_content": "Published on January 15, 2018...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec2",
                    "section_title": "Publication",
                    "chunk_id": "chunk2",
                },
            ]
        }

        # Execute
        result = tool.execute(query="effective dates", k=10)

        # Assert
        assert result.success
        assert result.data["event_count"] > 0
        assert len(result.data["timeline_events"]) > 0

        # Check that dates were extracted
        event = result.data["timeline_events"][0]
        assert "dates" in event
        assert len(event["dates"]) > 0

    def test_timeline_view_with_metadata_dates(self, base_tool_kwargs, mock_vector_store):
        """Test timeline with dates from chunk metadata."""
        # Setup
        tool = TimelineViewTool(**base_tool_kwargs)

        # Mock chunks with date metadata
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "content": "Standard content...",
                    "raw_content": "Standard content...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "date": "2018-07-01",
                    "section_id": "sec1",
                    "section_title": "Section",
                    "chunk_id": "chunk1",
                }
            ]
        }

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert result.data["event_count"] > 0

    def test_timeline_view_no_dates(self, base_tool_kwargs, mock_vector_store):
        """Test timeline when no dates found."""
        # Setup
        tool = TimelineViewTool(**base_tool_kwargs)

        # Mock chunks without dates
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "content": "Standard content without dates...",
                    "raw_content": "Standard content without dates...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Section",
                    "chunk_id": "chunk1",
                }
            ]
        }

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert result.data["event_count"] == 0


# ============================================================================
# Test SummarizeSectionTool
# ============================================================================


class TestSummarizeSectionTool:
    """Test SummarizeSectionTool."""

    def test_summarize_section_found(self, base_tool_kwargs, mock_vector_store):
        """Test summarizing a section that exists."""
        # Setup
        tool = SummarizeSectionTool(**base_tool_kwargs)

        # Mock section and chunks
        mock_vector_store.hierarchical_search.return_value = {
            "layer2": [
                {
                    "content": "Section summary...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "chunk_id": "sec1",
                }
            ],
            "layer3": [
                {
                    "content": "Chunk 1 content...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "chunk_id": "chunk1",
                },
                {
                    "content": "Chunk 2 content...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "chunk_id": "chunk2",
                },
            ],
        }

        # Execute
        result = tool.execute(doc_id="doc1", section_id="sec1")

        # Assert
        assert result.success
        assert result.data["doc_id"] == "doc1"
        assert result.data["section_id"] == "sec1"
        assert result.data["chunk_count"] == 2
        assert result.data["section_summary"] is not None

    def test_summarize_section_with_context_assembler(
        self, base_tool_kwargs, mock_vector_store, mock_context_assembler
    ):
        """Test summarizing section with context assembler formatting."""
        # Setup
        tool = SummarizeSectionTool(**base_tool_kwargs)

        # Mock section chunks
        mock_vector_store.hierarchical_search.return_value = {
            "layer2": [],
            "layer3": [
                {
                    "content": "Chunk content...",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Section",
                    "chunk_id": "chunk1",
                }
            ],
        }

        # Execute
        result = tool.execute(doc_id="doc1", section_id="sec1")

        # Assert
        assert result.success
        assert "formatted_content" in result.data
        mock_context_assembler.assemble.assert_called_once()

    def test_summarize_section_not_found(self, base_tool_kwargs, mock_vector_store):
        """Test summarizing non-existent section."""
        # Setup
        tool = SummarizeSectionTool(**base_tool_kwargs)

        # Mock empty results
        mock_vector_store.hierarchical_search.return_value = {
            "layer2": [],
            "layer3": [],
        }

        # Execute
        result = tool.execute(doc_id="doc1", section_id="nonexistent")

        # Assert
        assert not result.success
        assert "not found" in result.error.lower()


# ============================================================================
# Test GetStatisticsTool
# ============================================================================


class TestGetStatisticsTool:
    """Test GetStatisticsTool (legacy)."""

    def test_get_corpus_statistics(self, base_tool_kwargs, mock_vector_store):
        """Test getting corpus-level statistics."""
        # Setup
        tool = GetStatisticsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {
            "layer1": [
                {"doc_id": "doc1"},
                {"doc_id": "doc2"},
                {"doc_id": "doc1"},  # Duplicate
            ]
        }

        # Execute
        result = tool.execute(stat_type="corpus")

        # Assert
        assert result.success
        assert "unique_documents" in result.data
        assert result.data["unique_documents"] == 2
        assert "layer1_count" in result.data

    def test_get_statistics_with_kg(self, base_tool_kwargs, mock_vector_store):
        """Test statistics including knowledge graph."""
        # Setup
        tool = GetStatisticsTool(**base_tool_kwargs)
        kg = base_tool_kwargs["knowledge_graph"]

        # Mock KG entities/relationships
        entity1 = Mock()
        entity1.type = "STANDARD"
        entity1.confidence = 0.95

        entity2 = Mock()
        entity2.type = "ORGANIZATION"
        entity2.confidence = 0.90

        kg.entities = {"e1": entity1, "e2": entity2}

        rel1 = Mock()
        rel1.type = "REFERENCES"

        kg.relationships = {"r1": rel1}

        # Mock layer1 search
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"doc_id": "doc1"}]}

        # Execute
        result = tool.execute(stat_type="entity")

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is not None
        assert result.data["knowledge_graph"]["total_entities"] == 2
        assert result.data["knowledge_graph"]["total_relationships"] == 1

    def test_get_statistics_no_kg(self, base_tool_kwargs, mock_vector_store):
        """Test statistics without knowledge graph."""
        # Setup
        base_tool_kwargs["knowledge_graph"] = None
        tool = GetStatisticsTool(**base_tool_kwargs)

        # Mock layer1 search
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"doc_id": "doc1"}]}

        # Execute
        result = tool.execute(stat_type="corpus")

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is None


# ============================================================================
# Test GetIndexStatisticsTool (NEW in Phase 7B)
# ============================================================================


class TestGetIndexStatisticsTool:
    """Test GetIndexStatisticsTool (comprehensive statistics)."""

    def test_get_index_statistics_basic(self, base_tool_kwargs, mock_vector_store):
        """Test getting basic index statistics."""
        # Setup
        tool = GetIndexStatisticsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {
            "layer1": [
                {"document_id": "doc1"},
                {"document_id": "doc2"},
            ]
        }

        # Execute
        result = tool.execute(include_cache_stats=False)

        # Assert
        assert result.success
        assert "vector_store" in result.data
        assert "embedding_model" in result.data
        assert "documents" in result.data
        assert result.data["documents"]["count"] == 2
        assert "hybrid_search_enabled" in result.data
        assert result.data["hybrid_search_enabled"] is True

    def test_get_index_statistics_with_cache(self, base_tool_kwargs, mock_vector_store):
        """Test index statistics with embedding cache stats."""
        # Setup
        tool = GetIndexStatisticsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute(include_cache_stats=True)

        # Assert
        assert result.success
        assert "embedding_cache" in result.data
        assert result.data["embedding_cache"]["hits"] == 42
        assert result.data["embedding_cache"]["hit_rate"] == 0.42

    def test_get_index_statistics_with_kg(self, base_tool_kwargs, mock_vector_store):
        """Test index statistics with knowledge graph."""
        # Setup
        tool = GetIndexStatisticsTool(**base_tool_kwargs)
        kg = base_tool_kwargs["knowledge_graph"]

        # Mock KG entities/relationships
        entity1 = Mock()
        entity1.type = "STANDARD"

        entity2 = Mock()
        entity2.type = "STANDARD"

        entity3 = Mock()
        entity3.type = "ORGANIZATION"

        kg.entities = {"e1": entity1, "e2": entity2, "e3": entity3}

        rel1 = Mock()
        rel1.type = "REFERENCES"

        rel2 = Mock()
        rel2.type = "ISSUED_BY"

        kg.relationships = {"r1": rel1, "r2": rel2}

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is not None
        assert result.data["knowledge_graph"]["total_entities"] == 3
        assert result.data["knowledge_graph"]["total_relationships"] == 2
        assert "entity_types" in result.data["knowledge_graph"]
        assert "relationship_types" in result.data["knowledge_graph"]

    def test_get_index_statistics_embedding_model_info(self, base_tool_kwargs, mock_vector_store):
        """Test that embedding model information is included."""
        # Setup
        tool = GetIndexStatisticsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert result.data["embedding_model"]["model_name"] == "test-embedding-model"
        assert result.data["embedding_model"]["dimensions"] == 10
        assert result.data["embedding_model"]["model_type"] == "test"

    def test_get_index_statistics_configuration(self, base_tool_kwargs, mock_vector_store):
        """Test that system configuration is included."""
        # Setup
        tool = GetIndexStatisticsTool(**base_tool_kwargs)
        tool.reranker = Mock()  # Reranker available

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert "configuration" in result.data
        assert result.data["configuration"]["reranking_enabled"] is True
        assert result.data["configuration"]["context_assembler_enabled"] is True

    def test_get_index_statistics_no_kg(self, base_tool_kwargs, mock_vector_store):
        """Test index statistics without knowledge graph."""
        # Setup
        base_tool_kwargs["knowledge_graph"] = None
        tool = GetIndexStatisticsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute()

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is None
        assert "embedding_model" in result.data
        assert "vector_store" in result.data
