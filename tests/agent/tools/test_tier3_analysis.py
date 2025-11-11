"""
Tests for TIER 3 Analysis Tools.

Tests for remaining TIER 3 tools:
1. GetStatsTool (unified statistics tool)

Removed tools (2025-01):
- ExplainEntityTool → GraphSearchTool (tier 2)
- GetEntityRelationshipsTool → GraphSearchTool (tier 2)
- TimelineViewTool → Use filtered_search with temporal filter
- SummarizeSectionTool → Use get_document_info or filtered_search
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from collections import Counter

from src.agent.tools.tier3_analysis import (
    GetStatsTool,  # Was GetStatisticsTool
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


# ============================================================================
# Tests for ExplainEntityTool and GetEntityRelationshipsTool removed (2025-01)
# - Functionality moved to GraphSearchTool (tier 2) with unified interface
# - Use graph_search(mode='entity_details') instead of ExplainEntityTool
# - Use graph_search(mode='relationships') instead of GetEntityRelationshipsTool
# ============================================================================


# ============================================================================
# Test TimelineViewTool
# ============================================================================
# Tests for TimelineViewTool and SummarizeSectionTool removed (tools deleted 2025-01)
# - TimelineViewTool: Replaced by filtered_search with temporal filter
# - SummarizeSectionTool: False advertising (returned chunks not LLM summary)
# ============================================================================


# ============================================================================
# Test GetStatsTool (Unified Statistics Tool)
# ============================================================================


class TestGetStatsTool:
    """Test GetStatsTool (unified statistics tool)."""

    def test_get_corpus_statistics(self, base_tool_kwargs, mock_vector_store):
        """Test getting corpus-level statistics."""
        # Setup
        tool = GetStatsTool(**base_tool_kwargs)

        # Mock document list
        mock_vector_store.hierarchical_search.return_value = {
            "layer1": [
                {"document_id": "doc1"},
                {"document_id": "doc2"},
                {"document_id": "doc1"},  # Duplicate
            ]
        }

        # Execute
        result = tool.execute(stat_scope="corpus")

        # Assert
        assert result.success
        assert "unique_documents" in result.data
        assert result.data["unique_documents"] == 2

    def test_get_statistics_with_kg(self, base_tool_kwargs, mock_vector_store):
        """Test statistics including knowledge graph."""
        # Setup
        tool = GetStatsTool(**base_tool_kwargs)
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
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute(stat_scope="entity")

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is not None
        assert result.data["knowledge_graph"]["total_entities"] == 2
        assert result.data["knowledge_graph"]["total_relationships"] == 1

    def test_get_statistics_no_kg(self, base_tool_kwargs, mock_vector_store):
        """Test statistics without knowledge graph."""
        # Setup
        base_tool_kwargs["knowledge_graph"] = None
        tool = GetStatsTool(**base_tool_kwargs)

        # Mock layer1 search
        mock_vector_store.hierarchical_search.return_value = {"layer1": [{"document_id": "doc1"}]}

        # Execute
        result = tool.execute(stat_scope="corpus")

        # Assert
        assert result.success
        assert result.data["knowledge_graph"] is None


# ============================================================================
# Test for GetIndexStatisticsTool removed (2025-01) - tool never existed
# - GetStatsTool with stat_scope='index' provides comprehensive statistics
# - Includes embedding model, cache stats, KG stats, and configuration
# ============================================================================
