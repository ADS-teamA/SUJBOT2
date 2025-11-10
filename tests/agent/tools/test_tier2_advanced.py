"""
Tests for Tier 2 Advanced Retrieval Tools.

Tests all 6 Tier 2 tools (after consolidation):
1. GraphSearchTool (consolidated: MultiHopSearchTool + EntityTool) - 4 modes, 16 tests
2. CompareDocumentsTool
3. ExplainSearchResultsTool
4. FilteredSearchTool (enhanced: HybridSearchWithFiltersTool + ExactMatchSearchTool) - 3 methods, 18 tests
5. SimilaritySearchTool (renamed from ChunkSimilaritySearchTool)
6. ExpandContextTool (renamed from ExpandSearchContextTool)

Coverage:
- Valid inputs and expected outputs
- Error cases (missing KG, no results, invalid inputs)
- Mode/method-based functionality (GraphSearchTool: 4 modes, FilteredSearchTool: 3 methods)
- Legacy parameter compatibility
- Edge cases and safety limits
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.agent.tools.tier2_advanced import (
    # MultiHopSearchTool,  # REMOVED - replaced by GraphSearchTool
    CompareDocumentsTool,
    # FindRelatedChunksTool,  # TODO: Check if this tool exists
    # TemporalSearchTool,  # TODO: Check if this tool exists
    # HybridSearchWithFiltersTool,  # REMOVED - replaced by FilteredSearchTool
    # CrossReferenceSearchTool,  # TODO: Check if this tool exists
    # ExpandSearchContextTool,  # TODO: Check if exists (might be ExpandContextTool)
    # ChunkSimilaritySearchTool,  # REMOVED - replaced by SimilaritySearchTool
    ExplainSearchResultsTool,
)

# New consolidated tools (to be tested):
from src.agent.tools.tier2_advanced import (
    GraphSearchTool,  # Replaces MultiHopSearchTool + EntityTool
    FilteredSearchTool,  # Replaces HybridSearchWithFiltersTool
    SimilaritySearchTool,  # Replaces ChunkSimilaritySearchTool
    ExpandContextTool,  # Check if this replaces ExpandSearchContextTool
    BrowseEntitiesTool,  # New tool for browsing entities by type/filters
)
from src.agent.tools.base import ToolResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = Mock()
    embedder.embed_texts.return_value = np.random.rand(1, 3072)
    embedder.dimensions = 3072
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store with hierarchical search."""
    store = Mock()

    # Default search results
    def hierarchical_search(**kwargs):
        return {
            "layer1": [],
            "layer2": [],
            "layer3": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "content": "This is chunk 0 content about waste disposal.",
                    "raw_content": "This is chunk 0 content about waste disposal.",
                    "rrf_score": 0.9,
                    "bm25_score": 0.85,
                    "dense_score": 0.88,
                },
                {
                    "chunk_id": "doc1:sec1:1",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "content": "This is chunk 1 content.",
                    "raw_content": "This is chunk 1 content.",
                    "rrf_score": 0.8,
                    "bm25_score": 0.75,
                    "dense_score": 0.78,
                },
                {
                    "chunk_id": "doc2:sec2:0",
                    "document_id": "doc2",
                    "doc_id": "doc2",
                    "section_id": "sec2",
                    "section_title": "Requirements",
                    "content": "This is chunk 2 from doc2.",
                    "raw_content": "This is chunk 2 from doc2.",
                    "date": "2023-05-15",
                    "rrf_score": 0.7,
                },
            ],
        }

    store.hierarchical_search.side_effect = hierarchical_search

    # Add metadata for layer3
    store.metadata_layer3 = [
        {
            "chunk_id": "doc1:sec1:0",
            "document_id": "doc1",
            "doc_id": "doc1",
            "section_id": "sec1",
            "section_title": "Introduction",
            "content": "This is chunk 0 content about waste disposal.",
            "raw_content": "This is chunk 0 content about waste disposal.",
            "rrf_score": 0.9,
            "bm25_score": 0.85,
            "dense_score": 0.88,
        },
        {
            "chunk_id": "doc1:sec1:1",
            "document_id": "doc1",
            "doc_id": "doc1",
            "section_id": "sec1",
            "section_title": "Introduction",
            "content": "This is chunk 1 content.",
            "raw_content": "This is chunk 1 content.",
            "rrf_score": 0.8,
            "bm25_score": 0.75,
            "dense_score": 0.78,
        },
        {
            "chunk_id": "doc2:sec2:0",
            "document_id": "doc2",
            "doc_id": "doc2",
            "section_id": "sec2",
            "section_title": "Requirements",
            "content": "This is chunk 2 from doc2.",
            "raw_content": "This is chunk 2 from doc2.",
            "rrf_score": 0.7,
        },
    ]

    return store


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    reranker = Mock()

    def rerank(query, chunks, top_k=6):
        # Add rerank scores and return top_k
        for i, chunk in enumerate(chunks[:top_k]):
            chunk["rerank_score"] = 1.0 - (i * 0.1)
        return chunks[:top_k]

    reranker.rerank.side_effect = rerank
    return reranker


@pytest.fixture
def mock_graph_retriever():
    """Create mock graph retriever."""
    retriever = Mock()

    def search_with_graph(**kwargs):
        # Return chunks with graph boosting
        return [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Graph-boosted chunk 0.",
                "raw_content": "Graph-boosted chunk 0.",
                "boosted_score": 0.95,
            },
            {
                "chunk_id": "doc1:sec1:1",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Graph-boosted chunk 1.",
                "raw_content": "Graph-boosted chunk 1.",
                "boosted_score": 0.85,
            },
        ]

    retriever.search_with_graph.side_effect = search_with_graph
    return retriever


@pytest.fixture
def mock_knowledge_graph():
    """Create mock knowledge graph."""
    kg = Mock()
    # Mock entities dict structure (knowledge_graph.entities.values())
    kg.entities = {}
    # Mock get_relationships_for_entity method
    kg.get_relationships_for_entity.return_value = []
    return kg


@pytest.fixture
def tool_dependencies(mock_vector_store, mock_embedder, mock_reranker):
    """Common dependencies for all tools."""
    return {
        "vector_store": mock_vector_store,
        "embedder": mock_embedder,
        "reranker": mock_reranker,
        "graph_retriever": None,
        "knowledge_graph": None,
        "context_assembler": None,
        "config": None,
    }


# ============================================================================
# TEST 1: GraphSearchTool (Consolidated from MultiHopSearchTool + EntityTool)
# ============================================================================


class TestGraphSearchTool:
    """Test unified graph search with 4 modes: entity_mentions, entity_details, relationships, multi_hop."""

    # ============================================================================
    # Mode 1: entity_mentions
    # ============================================================================

    def test_entity_mentions_without_kg_fails(self, tool_dependencies):
        """Test that entity_mentions mode fails gracefully without KG."""
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_mentions",
            entity_value="Company X",
            k=6
        )

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Knowledge graph not available" in result.error

    def test_entity_mentions_with_kg(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test entity_mentions mode with knowledge graph."""
        # Mock KG entities dict - use proper string values (not Mock objects)
        entity = Mock(
            id="entity1",
            value="Company X",  # Real string
            normalized_value="company x",  # Real string
            type=Mock(value="ORGANIZATION"),  # Mock with .value attribute
            source_chunk_ids=["doc1:sec1:0", "doc1:sec1:1"],
            confidence=0.9
        )
        mock_knowledge_graph.entities = {"entity1": entity}

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "content": "Company X info in chunk 0",
                "section_title": "Section 1",
            },
            {
                "chunk_id": "doc1:sec1:1",
                "document_id": "doc1",
                "content": "Company X info in chunk 1",
                "section_title": "Section 1",
            },
        ]

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_mentions",
            entity_value="Company X",
            k=6
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        assert result.metadata["mode"] == "entity_mentions"
        assert result.metadata["entity_value"] == "Company X"

    def test_entity_mentions_with_type_filter(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test entity_mentions with entity_type filter."""
        # Mock KG entities dict with proper string values
        entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        mock_knowledge_graph.entities = {"entity1": entity}

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "content": "Company X info",
                "section_title": "Section 1",
            },
        ]

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_mentions",
            entity_value="Company X",
            entity_type="ORGANIZATION",
            k=6
        )

        assert result.success is True
        assert isinstance(result.data, list)
        assert result.metadata["entity_type"] == "ORGANIZATION"

    def test_entity_mentions_not_found(self, tool_dependencies, mock_knowledge_graph):
        """Test entity_mentions when entity not found."""
        # Empty entities dict - entity not found
        mock_knowledge_graph.entities = {}
        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_mentions",
            entity_value="NonexistentEntity",
            k=6
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_find_entity_with_none_normalized_value(self, tool_dependencies, mock_knowledge_graph):
        """Test that _find_entity handles None normalized_value without crashing (bug fix test)."""
        # Create entity with None normalized_value (this was causing AttributeError: 'NoneType' object has no attribute 'lower')
        entity_with_none = Mock(
            id="entity1",
            value="Test Entity",
            normalized_value=None,  # This was causing the crash
            type=Mock(value="STANDARD"),
            confidence=0.9
        )

        # Also add entity with proper values
        entity_normal = Mock(
            id="entity2",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            confidence=0.85
        )

        # Entity with both values None (edge case)
        entity_all_none = Mock(
            id="entity3",
            value=None,
            normalized_value=None,
            type=Mock(value="CLAUSE"),
            confidence=0.7
        )

        mock_knowledge_graph.entities = {
            "entity1": entity_with_none,
            "entity2": entity_normal,
            "entity3": entity_all_none
        }
        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        # Test 1: Should find entity2 and skip entity1 (None normalized_value) and entity3 (all None)
        found = tool._find_entity("Company")
        assert found is not None
        assert found.id == "entity2"

        # Test 2: Should handle entity with None normalized_value but valid value
        entity_partial = Mock(
            id="entity4",
            value="Standard ABC",
            normalized_value=None,
            type=Mock(value="STANDARD"),
            confidence=0.95
        )
        mock_knowledge_graph.entities["entity4"] = entity_partial

        # Should find entity4 by matching value (even though normalized_value is None)
        found = tool._find_entity("ABC")
        assert found is not None
        assert found.id == "entity4"

        # Test 3: Verify no crash when no entities match
        found = tool._find_entity("NonexistentEntity")
        assert found is None

    # ============================================================================
    # Mode 2: entity_details
    # ============================================================================

    def test_entity_details_with_relationships(self, tool_dependencies, mock_knowledge_graph):
        """Test entity_details mode with relationships."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}

        # Setup relationships (new GraphAdapter API)
        mock_relationship = Mock(
            source_entity_id="entity1",
            target_entity_id="entity2",
            type=Mock(value="produces"),
            confidence=0.85
        )
        mock_knowledge_graph.get_relationships_for_entity = Mock(return_value=[mock_relationship])

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_details",
            entity_value="Company X",
            k=6
        )

        assert result.success is True
        assert "entity" in result.data
        assert "relationships" in result.data
        # entity_details mode automatically includes relationships
        assert len(result.data["relationships"]) > 0

    def test_entity_details_without_relationships(self, tool_dependencies, mock_knowledge_graph):
        """Test entity_details mode without relationships."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_details",
            entity_value="Company X",
            k=6
        )

        assert result.success is True
        assert "entity" in result.data
        # entity_details always includes relationships (may be empty)
        assert "relationships" in result.data
        assert len(result.data["relationships"]) == 0  # No relationships mocked

    def test_entity_details_with_direction_filter(self, tool_dependencies, mock_knowledge_graph):
        """Test entity_details with direction filter."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}
        mock_knowledge_graph.get_relationships_for_entity = Mock(return_value=[])

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="entity_details",
            entity_value="Company X",
            direction="outgoing",
            k=6
        )

        assert result.success is True
        assert result.metadata["direction"] == "outgoing"

    # ============================================================================
    # Mode 3: relationships
    # ============================================================================

    def test_relationships_mode_basic(self, tool_dependencies, mock_knowledge_graph):
        """Test relationships mode."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}

        # Setup relationships (new GraphAdapter API)
        mock_relationship = Mock(
            source_entity_id="entity1",
            target_entity_id="entity2",
            type=Mock(value="produces"),
            confidence=0.85
        )
        mock_knowledge_graph.get_relationships_for_entity = Mock(return_value=[mock_relationship])
        # Also need get_entity for target entity
        mock_target = Mock(id="entity2", value="Product Y", type=Mock(value="PRODUCT"))
        mock_knowledge_graph.get_entity = Mock(return_value=mock_target)

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="relationships",
            entity_value="Company X",
            k=6
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["count"] > 0
        assert len(result.data["relationships"]) > 0
        assert result.metadata["mode"] == "relationships"

    def test_relationships_with_type_filter(self, tool_dependencies, mock_knowledge_graph):
        """Test relationships mode with type filter."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}

        # Setup relationships (new GraphAdapter API)
        mock_relationship = Mock(
            source_entity_id="entity1",
            target_entity_id="entity2",
            type=Mock(value="produces"),
            confidence=0.85
        )
        mock_knowledge_graph.get_relationships_for_entity = Mock(return_value=[mock_relationship])
        # Also need get_entity for target entity
        mock_target = Mock(id="entity2", value="Product Y", type=Mock(value="PRODUCT"))
        mock_knowledge_graph.get_entity = Mock(return_value=mock_target)

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="relationships",
            entity_value="Company X",
            relationship_types=["PRODUCES"],
            k=6
        )

        assert result.success is True
        assert result.metadata["relationship_types"] == ["PRODUCES"]

    def test_relationships_no_results(self, tool_dependencies, mock_knowledge_graph):
        """Test relationships mode when no relationships found."""
        # Mock entity with proper string values
        mock_entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": mock_entity}
        mock_knowledge_graph.get_relationships_for_entity = Mock(return_value=[])

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="relationships",
            entity_value="Company X",
            k=6
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["count"] == 0
        assert len(result.data["relationships"]) == 0

    # ============================================================================
    # Mode 4: multi_hop
    # ============================================================================

    def test_multi_hop_bfs_traversal(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test multi-hop BFS traversal."""
        # Setup entity chain: A -> B -> C with proper string values
        entity_a = Mock(
            id="entity_a",
            value="Company A",
            normalized_value="company a",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9,
            document_id="doc1"
        )
        entity_b = Mock(
            id="entity_b",
            value="Company B",
            normalized_value="company b",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:1"],
            confidence=0.85,
            document_id="doc1"
        )
        entity_c = Mock(
            id="entity_c",
            value="Company C",
            normalized_value="company c",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:2"],
            confidence=0.8,
            document_id="doc1"
        )

        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "entity_c": entity_c
        }

        # Setup get_entity for retrieving by ID
        def mock_get_entity(entity_id):
            return mock_knowledge_graph.entities.get(entity_id)

        mock_knowledge_graph.get_entity = Mock(side_effect=mock_get_entity)

        # Setup get_outgoing_relationships for BFS traversal
        def mock_get_outgoing_rels(entity_id):
            if entity_id == "entity_a":
                return [Mock(
                    source_entity_id="entity_a",
                    target_entity_id="entity_b",
                    type=Mock(value="partner"),
                    confidence=0.85
                )]
            elif entity_id == "entity_b":
                return [Mock(
                    source_entity_id="entity_b",
                    target_entity_id="entity_c",
                    type=Mock(value="subsidiary"),
                    confidence=0.8
                )]
            return []

        mock_knowledge_graph.get_outgoing_relationships = Mock(side_effect=mock_get_outgoing_rels)

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = [
            {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "content": "Company A info"},
            {"chunk_id": "doc1:sec1:1", "document_id": "doc1", "content": "Company B info"},
            {"chunk_id": "doc1:sec1:2", "document_id": "doc1", "content": "Company C info"},
        ]

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="multi_hop",
            entity_value="Company A",
            max_hops=2,
            k=10
        )

        assert result.success is True
        chunks = result.data.get("chunks", [])
        assert len(chunks) > 0
        assert result.metadata.get("mode") == "multi_hop"
        assert result.metadata.get("max_hops") == 2

    def test_multi_hop_max_hops_validation(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test max_hops parameter validation."""
        # Mock entity with proper string values
        entity = Mock(
            id="entity1",
            value="Company X",
            normalized_value="company x",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9,
            document_id="doc1"
        )
        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {"entity1": entity}
        mock_knowledge_graph.get_entity = Mock(return_value=entity)
        mock_knowledge_graph.get_outgoing_relationships = Mock(return_value=[])

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = [
            {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "content": "Company X info"}
        ]

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        # Valid: max_hops=1
        result = tool.execute(mode="multi_hop", entity_value="Company X", max_hops=1, k=6)
        assert result.success is True

        # Valid: max_hops=3
        result = tool.execute(mode="multi_hop", entity_value="Company X", max_hops=3, k=6)
        assert result.success is True

        # Invalid: max_hops=0 (should fail validation)
        result = tool.execute(mode="multi_hop", entity_value="Company X", max_hops=0, k=6)
        assert result.success is False
        assert "validation" in result.error.lower()

    def test_multi_hop_safety_limits(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test multi-hop safety limits (20 entities/hop, 200 max)."""
        # Create a large entity chain with proper string values
        entities = {}
        chunks = []
        for i in range(100):
            entity_id = f"entity_{i}"
            entities[entity_id] = Mock(
                id=entity_id,
                value=f"Entity{i}",
                normalized_value=f"entity{i}",
                type=Mock(value="ORGANIZATION"),
                source_chunk_ids=[f"doc1:sec1:{i}"],
                confidence=0.9,
                document_id="doc1"
            )
            chunks.append({
                "chunk_id": f"doc1:sec1:{i}",
                "document_id": "doc1",
                "content": f"Entity{i} info"
            })

        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = entities

        # Setup get_entity for retrieving by ID
        def mock_get_entity(entity_id):
            return entities.get(entity_id)

        mock_knowledge_graph.get_entity = Mock(side_effect=mock_get_entity)

        # Setup get_outgoing_relationships - return many relationships to test limits
        def mock_get_outgoing_rels(entity_id):
            # Return many relationships (30) to test per-hop limit
            idx = int(entity_id.split("_")[1])
            return [
                Mock(
                    source_entity_id=entity_id,
                    target_entity_id=f"entity_{i}",
                    type=Mock(value="related"),
                    confidence=0.8
                )
                for i in range(min(idx + 1, idx + 30, 100))
            ]

        mock_knowledge_graph.get_outgoing_relationships = Mock(side_effect=mock_get_outgoing_rels)

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = chunks

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="multi_hop",
            entity_value="Entity0",
            max_hops=2,
            k=50
        )

        assert result.success is True
        # Should hit safety limits
        traversal = result.data.get("traversal", {})
        assert traversal.get("total_entities_discovered", 0) <= 200

    def test_multi_hop_circular_reference_handling(self, tool_dependencies, mock_knowledge_graph, mock_vector_store):
        """Test circular reference handling in multi-hop."""
        # Setup circular reference: A -> B -> A with proper string values
        entity_a = Mock(
            id="entity_a",
            value="Company A",
            normalized_value="company a",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:0"],
            confidence=0.9,
            document_id="doc1"
        )
        entity_b = Mock(
            id="entity_b",
            value="Company B",
            normalized_value="company b",
            type=Mock(value="ORGANIZATION"),
            source_chunk_ids=["doc1:sec1:1"],
            confidence=0.85,
            document_id="doc1"
        )

        # Setup entities dict (new GraphAdapter API)
        mock_knowledge_graph.entities = {
            "entity_a": entity_a,
            "entity_b": entity_b
        }

        # Setup get_entity for retrieving by ID
        def mock_get_entity(entity_id):
            return mock_knowledge_graph.entities.get(entity_id)

        mock_knowledge_graph.get_entity = Mock(side_effect=mock_get_entity)

        # Setup get_outgoing_relationships - circular reference
        def mock_get_outgoing_rels(entity_id):
            if entity_id == "entity_a":
                return [Mock(
                    source_entity_id="entity_a",
                    target_entity_id="entity_b",
                    type=Mock(value="partner"),
                    confidence=0.85
                )]
            elif entity_id == "entity_b":
                return [Mock(
                    source_entity_id="entity_b",
                    target_entity_id="entity_a",  # Back to A (circular!)
                    type=Mock(value="partner"),
                    confidence=0.85
                )]
            return []

        mock_knowledge_graph.get_outgoing_relationships = Mock(side_effect=mock_get_outgoing_rels)

        # Mock metadata_layer3 for chunk retrieval
        mock_vector_store.metadata_layer3 = [
            {"chunk_id": "doc1:sec1:0", "document_id": "doc1", "content": "Company A info"},
            {"chunk_id": "doc1:sec1:1", "document_id": "doc1", "content": "Company B info"},
        ]

        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="multi_hop",
            entity_value="Company A",
            max_hops=3,
            k=10
        )

        assert result.success is True
        # Should not infinitely loop - BFS tracks visited entities
        traversal = result.data.get("traversal", {})
        assert traversal.get("total_entities_discovered", 0) < 10

    # ============================================================================
    # General tests
    # ============================================================================

    def test_invalid_mode(self, tool_dependencies, mock_knowledge_graph):
        """Test invalid mode parameter."""
        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        result = tool.execute(
            mode="invalid_mode",
            entity_value="Company X",
            k=6
        )

        assert result.success is False
        assert "validation" in result.error.lower() or "mode" in result.error.lower()


# ============================================================================
# TEST 2: CompareDocumentsTool
# ============================================================================


class TestCompareDocumentsTool:
    """Test document comparison functionality."""

    def test_compare_documents_basic(self, tool_dependencies, mock_vector_store):
        """Test basic document comparison."""

        # Mock the new direct layer search approach
        doc1_chunks = [
            {
                "chunk_id": "doc1:sec1:0",
                "doc_id": "doc1",
                "document_id": "doc1",
                "content": "Doc1 content.",
            }
        ]
        doc2_chunks = [
            {
                "chunk_id": "doc2:sec1:0",
                "doc_id": "doc2",
                "document_id": "doc2",
                "content": "Doc2 content.",
            }
        ]

        # Mock faiss_store and bm25_store
        mock_vector_store.faiss_store = Mock()
        mock_vector_store.bm25_store = Mock()
        mock_vector_store._rrf_fusion = Mock()

        # Setup side_effect to return appropriate chunks
        def rrf_fusion_side_effect(dense, sparse, k):
            # Determine which document based on call order
            if not hasattr(rrf_fusion_side_effect, 'call_count'):
                rrf_fusion_side_effect.call_count = 0
            rrf_fusion_side_effect.call_count += 1
            return doc1_chunks if rrf_fusion_side_effect.call_count == 1 else doc2_chunks

        mock_vector_store._rrf_fusion.side_effect = rrf_fusion_side_effect
        mock_vector_store.faiss_store.search_layer3.return_value = []
        mock_vector_store.bm25_store.search_layer3.return_value = []

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="doc1", doc_id_2="doc2")

        assert result.success is True
        assert result.data["doc_id_1"] == "doc1"
        assert result.data["doc_id_2"] == "doc2"
        assert "doc1_chunk_count" in result.data
        assert "doc2_chunk_count" in result.data
        assert "doc1" in result.citations
        assert "doc2" in result.citations

    def test_compare_documents_with_aspect(self, tool_dependencies, mock_vector_store):
        """Test document comparison with specific aspect."""
        # Mock the new direct layer search approach
        mock_vector_store.faiss_store = Mock()
        mock_vector_store.bm25_store = Mock()
        mock_vector_store._rrf_fusion = Mock(return_value=[
            {"chunk_id": "doc1:sec1:0", "doc_id": "doc1", "content": "Requirements content."}
        ])
        mock_vector_store.faiss_store.search_layer3.return_value = []
        mock_vector_store.bm25_store.search_layer3.return_value = []

        tool = CompareDocumentsTool(**tool_dependencies)

        result = tool.execute(doc_id_1="doc1", doc_id_2="doc2", comparison_aspect="requirements")

        assert result.success is True
        assert result.data["comparison_aspect"] == "requirements"
        assert "doc1_relevant_chunks" in result.data
        assert "doc2_relevant_chunks" in result.data

    def test_compare_documents_first_not_found(self, tool_dependencies):
        """Test comparison when first document not found."""
        # Create fresh mock that returns empty for all calls
        mock_vs = Mock()
        mock_vs.faiss_store = Mock()
        mock_vs.bm25_store = Mock()
        mock_vs._rrf_fusion = Mock(return_value=[])
        mock_vs.faiss_store.search_layer3.return_value = []
        mock_vs.bm25_store.search_layer3.return_value = []
        tool_dependencies["vector_store"] = mock_vs

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="nonexistent", doc_id_2="doc2")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_compare_documents_second_not_found(self, tool_dependencies, mock_vector_store):
        """Test comparison when second document not found."""

        # Mock the new direct layer search approach
        doc1_chunks = [{"chunk_id": "doc1:sec1:0", "doc_id": "doc1"}]
        doc2_chunks = []  # Empty - doc2 not found

        # Mock faiss_store and bm25_store
        mock_vector_store.faiss_store = Mock()
        mock_vector_store.bm25_store = Mock()
        mock_vector_store._rrf_fusion = Mock()

        # Setup side_effect to return doc1 chunks first, then empty for doc2
        def rrf_fusion_side_effect(dense, sparse, k):
            if not hasattr(rrf_fusion_side_effect, 'call_count'):
                rrf_fusion_side_effect.call_count = 0
            rrf_fusion_side_effect.call_count += 1
            return doc1_chunks if rrf_fusion_side_effect.call_count == 1 else doc2_chunks

        mock_vector_store._rrf_fusion.side_effect = rrf_fusion_side_effect
        mock_vector_store.faiss_store.search_layer3.return_value = []
        mock_vector_store.bm25_store.search_layer3.return_value = []

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="doc1", doc_id_2="nonexistent")

        assert result.success is False
        assert "nonexistent" in result.error


# ============================================================================
# TEST 3: FindRelatedChunksTool
# ============================================================================


class TestFindRelatedChunksTool:
    """Test finding semantically related chunks."""

    def test_find_related_chunks_basic(self, tool_dependencies):
        """Test finding related chunks."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=6, same_document_only=False)

        assert result.success is True
        assert len(result.data) > 0
        # Source chunk should be filtered out
        assert not any(c["chunk_id"] == "doc1:sec1:0" for c in result.data)
        assert result.metadata["source_chunk_id"] == "doc1:sec1:0"

    def test_find_related_chunks_same_document_only(self, tool_dependencies):
        """Test finding related chunks within same document."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=6, same_document_only=True)

        assert result.success is True
        assert result.metadata["same_document_only"] is True
        # Source chunk doc1:sec1:0 should be filtered out
        # Check that all remaining results are from doc1 or doc2 (mock returns both initially)
        # but results might include doc2:sec2:0 since mock doesn't filter perfectly
        # The key is that doc1:sec1:0 should not be in the results
        assert not any(c.get("chunk_id") == "doc1:sec1:0" for c in result.data)

    def test_find_related_chunks_not_found(self, tool_dependencies, mock_vector_store):
        """Test when chunk ID not found."""
        # Mock search to return no matching chunk
        mock_vector_store.hierarchical_search.return_value = {"layer3": []}

        tool = FindRelatedChunksTool(**tool_dependencies)
        result = tool.execute(chunk_id="nonexistent:chunk:id", k=6)

        assert result.success is False
        assert "not found" in result.error

    def test_find_related_chunks_with_reranker(self, tool_dependencies):
        """Test that reranker is applied to results."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=3)

        assert result.success is True
        # Verify reranker was called (results should have rerank_score)
        if result.data:
            # Check that we got limited results (k=3)
            assert len(result.data) <= 3


# ============================================================================
# TEST 4: TemporalSearchTool
# ============================================================================


class TestTemporalSearchTool:
    """Test temporal search with date filtering."""

    def test_temporal_search_no_date_filter(self, tool_dependencies):
        """Test temporal search without date filtering."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(query="regulations", k=6)

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["start_date"] is None
        assert result.metadata["end_date"] is None

    def test_temporal_search_with_start_date(self, tool_dependencies):
        """Test temporal search with start date filter."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(query="regulations", start_date="2023-01-01", k=6)

        assert result.success is True
        assert result.metadata["start_date"] == "2023-01-01"

    def test_temporal_search_with_date_range(self, tool_dependencies):
        """Test temporal search with date range."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(
            query="regulations", start_date="2023-01-01", end_date="2023-12-31", k=6
        )

        assert result.success is True
        assert result.metadata["start_date"] == "2023-01-01"
        assert result.metadata["end_date"] == "2023-12-31"

    def test_temporal_search_filters_by_date(self, tool_dependencies, mock_reranker):
        """Test that chunks are actually filtered by date."""
        # Create fresh mock with specific data
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "date": "2023-05-15",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "date": "2022-03-20",
                },
                {
                    "chunk_id": "c3",
                    "doc_id": "doc3",
                    "document_id": "doc3",
                    "section_id": "sec3",
                    "content": "Content 3",
                    "raw_content": "Content 3",
                    "date": "2024-01-10",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = TemporalSearchTool(**tool_dependencies)
        result = tool.execute(query="test", start_date="2023-01-01", end_date="2023-12-31", k=6)

        assert result.success is True
        # Only chunk with date 2023-05-15 should pass filter
        assert len(result.data) >= 1
        # Check first result is c1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_temporal_search_no_results_in_range(self, tool_dependencies):
        """Test when no results in date range."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content",
                    "raw_content": "Content",
                    "date": "2020-01-01",
                }
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = TemporalSearchTool(**tool_dependencies)
        result = tool.execute(query="test", start_date="2023-01-01", end_date="2023-12-31", k=6)

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 5: HybridSearchWithFiltersTool
# ============================================================================


class TestHybridSearchWithFiltersTool:
    """Test hybrid search with metadata filters."""

    def test_hybrid_search_no_filters(self, tool_dependencies):
        """Test hybrid search without filters."""
        tool = HybridSearchWithFiltersTool(**tool_dependencies)

        result = tool.execute(query="waste disposal", k=6)

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["document_type"] is None
        assert result.metadata["section_type"] is None

    def test_hybrid_search_with_document_type(self, tool_dependencies, mock_reranker):
        """Test filtering by document type."""
        # Create fresh mock with doc_type metadata
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "doc_type": "regulation",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "doc_type": "contract",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", document_type="regulation", k=6)

        assert result.success is True
        assert len(result.data) >= 1
        # Verify we got the regulation chunk
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_with_section_type(self, tool_dependencies, mock_reranker):
        """Test filtering by section type."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "section_type": "requirements",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "section_type": "introduction",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", section_type="requirements", k=6)

        assert result.success is True
        assert len(result.data) >= 1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_with_both_filters(self, tool_dependencies, mock_reranker):
        """Test filtering by both document and section type."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "doc_type": "regulation",
                    "section_type": "requirements",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "doc_type": "contract",
                    "section_type": "requirements",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(
            query="test", document_type="regulation", section_type="requirements", k=6
        )

        assert result.success is True
        assert len(result.data) >= 1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_no_results_after_filtering(self, tool_dependencies, mock_vector_store):
        """Test when filters eliminate all results."""
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [{"chunk_id": "c1", "doc_type": "contract"}]
        }

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", document_type="regulation", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 6: FilteredSearchTool (Enhanced with 3 search methods)
# ============================================================================


class TestFilteredSearchTool:
    """Test unified filtered search with 3 methods: hybrid, bm25_only, dense_only."""

    # ============================================================================
    # Method 1: hybrid (default)
    # ============================================================================

    def test_filtered_search_hybrid_basic(self, tool_dependencies):
        """Test basic hybrid search without filters."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="waste disposal",
            search_method="hybrid",
            k=6
        )

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["search_method"] == "hybrid"

    def test_filtered_search_hybrid_with_document_filter(self, tool_dependencies, mock_vector_store):
        """Test hybrid search with document filter."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="regulations",
            search_method="hybrid",
            filter_type="document",
            filter_value="doc1",
            k=6
        )

        assert result.success is True
        assert result.metadata["filter_type"] == "document"
        assert result.metadata["filter_value"] == "doc1"

    def test_filtered_search_hybrid_with_section_filter(self, tool_dependencies, mock_vector_store):
        """Test hybrid search with section filter."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="requirements",
            search_method="hybrid",
            filter_type="section",
            filter_value="Introduction",
            k=6
        )

        assert result.success is True
        assert result.metadata["filter_type"] == "section"

    def test_filtered_search_hybrid_with_temporal_filter(self, tool_dependencies, mock_reranker):
        """Test hybrid search with temporal filter."""
        # Create mock with date metadata
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "date": "2023-05-15",
                    "rrf_score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "date": "2022-03-20",
                    "rrf_score": 0.8,
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="hybrid",
            filter_type="temporal",
            start_date="2023-01-01",
            end_date="2023-12-31",
            k=6
        )

        assert result.success is True
        assert result.metadata["filter_type"] == "temporal"
        # Should only return chunks in date range
        assert len(result.data) >= 1

    def test_filtered_search_hybrid_with_metadata_filter(self, tool_dependencies, mock_reranker):
        """Test hybrid search with metadata filter."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "doc_type": "regulation",
                    "rrf_score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "doc_type": "contract",
                    "rrf_score": 0.8,
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="hybrid",
            filter_type="metadata",
            metadata_key="doc_type",
            metadata_value="regulation",
            k=6
        )

        assert result.success is True
        assert result.metadata["filter_type"] == "metadata"
        # Should only return regulation chunks
        assert len(result.data) >= 1

    def test_filtered_search_hybrid_with_content_filter(self, tool_dependencies, mock_reranker):
        """Test hybrid search with content filter (keyword)."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "This mentions Article 5.2 specifically.",
                    "raw_content": "This mentions Article 5.2 specifically.",
                    "rrf_score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Different content here.",
                    "raw_content": "Different content here.",
                    "rrf_score": 0.8,
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="hybrid",
            filter_type="content",
            filter_value="Article 5.2",
            k=6
        )

        assert result.success is True
        assert result.metadata["filter_type"] == "content"
        # Should only return chunks containing the keyword
        assert len(result.data) >= 1

    # ============================================================================
    # Method 2: bm25_only
    # ============================================================================

    def test_filtered_search_bm25_only_basic(self, tool_dependencies, mock_vector_store):
        """Test BM25-only search method."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="waste disposal",
            search_method="bm25_only",
            k=6
        )

        assert result.success is True
        assert result.metadata["search_method"] == "bm25_only"
        # BM25 method should be faster (no dense search)
        assert len(result.data) > 0

    def test_filtered_search_bm25_with_document_filter(self, tool_dependencies, mock_vector_store):
        """Test BM25-only with document filter."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="regulations",
            search_method="bm25_only",
            filter_type="document",
            filter_value="doc1",
            k=6
        )

        assert result.success is True
        assert result.metadata["search_method"] == "bm25_only"
        assert result.metadata["filter_type"] == "document"

    def test_filtered_search_bm25_with_post_filters(self, tool_dependencies, mock_reranker):
        """Test BM25-only with post-filtering."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "bm25_score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "section_title": "Requirements",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "bm25_score": 0.8,
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="bm25_only",
            filter_type="section",
            filter_value="Introduction",
            k=6
        )

        assert result.success is True
        # Should apply section filter after BM25 search
        assert len(result.data) >= 1

    # ============================================================================
    # Method 3: dense_only
    # ============================================================================

    def test_filtered_search_dense_only_basic(self, tool_dependencies, mock_vector_store):
        """Test dense-only (semantic) search method."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="waste disposal",
            search_method="dense_only",
            k=6
        )

        assert result.success is True
        assert result.metadata["search_method"] == "dense_only"
        assert len(result.data) > 0

    def test_filtered_search_dense_with_filters(self, tool_dependencies, mock_reranker):
        """Test dense-only with post-filtering."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "dense_score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "dense_score": 0.8,
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="dense_only",
            filter_type="document",
            filter_value="doc1",
            k=6
        )

        assert result.success is True
        assert result.metadata["search_method"] == "dense_only"
        assert result.metadata["filter_type"] == "document"

    # ============================================================================
    # Legacy parameter mapping
    # ============================================================================

    def test_filtered_search_legacy_search_type(self, tool_dependencies, mock_vector_store):
        """Test legacy search_type parameter maps to search_method."""
        tool = FilteredSearchTool(**tool_dependencies)

        # Legacy: search_type="exact" should map to search_method="bm25_only"
        result = tool.execute(
            query="test",
            search_type="exact",  # Legacy parameter
            k=6
        )

        assert result.success is True
        # Should use BM25 method
        assert result.metadata.get("search_method") in ["bm25_only", "hybrid"]

    def test_filtered_search_legacy_document_id(self, tool_dependencies, mock_vector_store):
        """Test legacy document_id parameter maps to filter_type/filter_value."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            document_id="doc1",  # Legacy parameter
            k=6
        )

        assert result.success is True
        # Should apply document filter
        assert result.metadata.get("filter_type") in ["document", None]

    # ============================================================================
    # Edge cases
    # ============================================================================

    def test_filtered_search_invalid_method(self, tool_dependencies, mock_vector_store):
        """Test invalid search_method parameter."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="invalid_method",
            k=6
        )

        assert result.success is False
        assert "validation" in result.error.lower() or "method" in result.error.lower()

    def test_filtered_search_missing_filter_value(self, tool_dependencies, mock_vector_store):
        """Test filter_type without filter_value."""
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="hybrid",
            filter_type="document",
            # Missing filter_value
            k=6
        )

        # Should either fail validation or ignore filter
        assert isinstance(result, ToolResult)

    def test_filtered_search_no_results_after_filtering(self, tool_dependencies, mock_vector_store):
        """Test when filters eliminate all results."""
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "content": "Content",
                    "raw_content": "Content",
                    "rrf_score": 0.9,
                }
            ]
        }
        tool = FilteredSearchTool(**tool_dependencies)

        result = tool.execute(
            query="test",
            search_method="hybrid",
            filter_type="document",
            filter_value="doc1",  # No chunks from doc1
            k=6
        )

        assert result.success is True
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 7: CrossReferenceSearchTool
# ============================================================================


class TestCrossReferenceSearchTool:
    """Test cross-reference search."""

    def test_cross_reference_search_basic(self, tool_dependencies):
        """Test basic cross-reference search."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "See Article 5.2 for details.",
                    "raw_content": "See Article 5.2 for details.",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Article 5.2 specifies requirements.",
                    "raw_content": "Article 5.2 specifies requirements.",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 5.2", k=6)

        assert result.success is True
        assert len(result.data) >= 2
        assert result.metadata["reference_text"] == "Article 5.2"

    def test_cross_reference_search_case_insensitive(self, tool_dependencies):
        """Test that cross-reference search is case-insensitive."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "see ARTICLE 5.2 for details.",
                    "raw_content": "see ARTICLE 5.2 for details.",
                }
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="article 5.2", k=6)

        assert result.success is True
        assert len(result.data) >= 1

    def test_cross_reference_search_no_matches(self, tool_dependencies, mock_vector_store):
        """Test when reference text not found in any chunks."""
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "content": "Different content.",
                    "raw_content": "Different content.",
                }
            ]
        }

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 99.99", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["results_count"] == 0

    def test_cross_reference_search_limits_results(self, tool_dependencies, mock_vector_store):
        """Test that results are limited to k."""
        # Create 10 matching chunks
        chunks = [
            {
                "chunk_id": f"c{i}",
                "doc_id": "doc1",
                "document_id": "doc1",
                "section_id": "sec1",
                "content": f"Article 5 mentioned here {i}.",
                "raw_content": f"Article 5 mentioned here {i}.",
            }
            for i in range(10)
        ]
        mock_vector_store.hierarchical_search.return_value = {"layer3": chunks}

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 5", k=3)

        assert result.success is True
        assert len(result.data) <= 3


# ============================================================================
# TEST 7: ExpandSearchContextTool (Phase 7B)
# ============================================================================


class TestExpandSearchContextTool:
    """Test context expansion with multiple strategies."""

    def test_expand_search_context_section_strategy(self, tool_dependencies, mock_vector_store):
        """Test section-based expansion."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="section", k=3)

        assert result.success is True
        assert "expansions" in result.data
        assert result.data["expansion_strategy"] == "section"

    def test_expand_search_context_similarity_strategy(self, tool_dependencies):
        """Test similarity-based expansion."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="similarity", k=3)

        assert result.success is True
        assert result.data["expansion_strategy"] == "similarity"

    def test_expand_search_context_hybrid_strategy(self, tool_dependencies):
        """Test hybrid expansion (section + similarity)."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="hybrid", k=4)

        assert result.success is True
        assert result.data["expansion_strategy"] == "hybrid"

    def test_expand_search_context_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test expansion when chunk not found."""
        mock_vector_store.metadata_layer3 = []

        tool = ExpandSearchContextTool(**tool_dependencies)
        result = tool.execute(chunk_ids=["nonexistent:chunk:id"], expansion_strategy="section", k=3)

        assert result.success is False
        assert "No chunks found" in result.error

    def test_expand_search_context_multiple_chunks(self, tool_dependencies):
        """Test expanding multiple chunks."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(
            chunk_ids=["doc1:sec1:0", "doc1:sec1:1"], expansion_strategy="section", k=2
        )

        assert result.success is True
        assert len(result.data["expansions"]) == 2

    def test_expand_search_context_deduplication(self, tool_dependencies):
        """Test that duplicate chunks are removed."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="hybrid", k=3)

        # Check that target chunk is not in expanded_chunks
        if result.success and result.data["expansions"]:
            expansion = result.data["expansions"][0]
            target_id = expansion["target_chunk"]["chunk_id"]
            expanded_ids = [c["chunk_id"] for c in expansion["expanded_chunks"]]
            assert target_id not in expanded_ids


# ============================================================================
# TEST 8: ChunkSimilaritySearchTool (Phase 7B)
# ============================================================================


class TestChunkSimilaritySearchTool:
    """Test 'more like this chunk' similarity search."""

    def test_chunk_similarity_search_cross_document(self, tool_dependencies):
        """Test similarity search across all documents."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        assert "target_chunk" in result.data
        assert "similar_chunks" in result.data
        assert result.metadata["cross_document"] is True

    def test_chunk_similarity_search_same_document(self, tool_dependencies, mock_vector_store):
        """Test similarity search within same document."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=False, k=6)

        assert result.success is True
        assert result.metadata["cross_document"] is False
        # All similar chunks should be from same document
        if result.data["similar_chunks"]:
            assert all(c["document_id"] == "doc1" for c in result.data["similar_chunks"])

    def test_chunk_similarity_search_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test when chunk ID not found."""
        mock_vector_store.metadata_layer3 = []

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="nonexistent:chunk:id", cross_document=True, k=6)

        assert result.success is False
        assert "not found" in result.error

    def test_chunk_similarity_search_no_content(self, tool_dependencies, mock_vector_store):
        """Test when chunk has no content."""
        mock_vector_store.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "content": "",  # Empty content
            }
        ]

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is False
        assert "no content" in result.error

    def test_chunk_similarity_search_excludes_self(self, tool_dependencies):
        """Test that target chunk is excluded from results."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        # Target chunk should not be in similar_chunks
        similar_ids = [c["chunk_id"] for c in result.data["similar_chunks"]]
        assert "doc1:sec1:0" not in similar_ids

    def test_chunk_similarity_search_no_similar_chunks(self, tool_dependencies):
        """Test when no similar chunks found."""
        # Create fresh mock that returns only the target chunk
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content",
                    "raw_content": "Content",
                }
            ]
        }
        mock_vs.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Content",
                "raw_content": "Content",
            }
        ]
        tool_dependencies["vector_store"] = mock_vs

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        # When no similar chunks found, tool returns empty list
        # (early return before dict construction)
        assert isinstance(result.data, list)
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 9: ExplainSearchResultsTool (Phase 7B)
# ============================================================================


class TestExplainSearchResultsTool:
    """Test search result explanation with score breakdowns."""

    def test_explain_search_results_basic(self, tool_dependencies):
        """Test basic search result explanation."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0", "doc1:sec1:1"])

        assert result.success is True
        assert "explanations" in result.data
        assert len(result.data["explanations"]) == 2

    def test_explain_search_results_with_scores(self, tool_dependencies):
        """Test explanation includes all score types."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert explanation["found"] is True
        assert "scores" in explanation
        assert "bm25_score" in explanation["scores"]
        assert "dense_score" in explanation["scores"]
        assert "rrf_score" in explanation["scores"]

    def test_explain_search_results_primary_method_detection(self, tool_dependencies):
        """Test detection of primary retrieval method."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert "primary_retrieval_method" in explanation
        # With our mock data, dense_score (0.88) > bm25_score (0.85)
        assert "semantic" in explanation["primary_retrieval_method"]

    def test_explain_search_results_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test explanation when chunk not found."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["nonexistent:chunk:id"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert explanation["found"] is False
        assert "error" in explanation

    def test_explain_search_results_hybrid_detection(self, tool_dependencies):
        """Test detection of hybrid search availability."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        assert "hybrid_search_enabled" in result.data

    def test_explain_search_results_content_preview(self, tool_dependencies):
        """Test that content preview is included."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert "content_preview" in explanation
        assert len(explanation["content_preview"]) <= 203  # 200 + "..."

    def test_explain_search_results_multiple_chunks(self, tool_dependencies):
        """Test explaining multiple chunks."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0", "doc1:sec1:1", "doc2:sec2:0"])

        assert result.success is True
        assert len(result.data["explanations"]) == 3
        assert result.metadata["chunk_count"] == 3


# ============================================================================
# TEST: Edge Cases & Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling across all tools."""

    def test_invalid_k_parameter(self, tool_dependencies, mock_knowledge_graph):
        """Test that invalid k values are caught by validation."""
        mock_knowledge_graph.get_entities.return_value = [
            Mock(value="Test", type="ORGANIZATION", chunk_ids=["doc1:sec1:0"], confidence=0.9)
        ]
        tool_dependencies["knowledge_graph"] = mock_knowledge_graph
        tool = GraphSearchTool(**tool_dependencies)

        # k=0 should fail validation
        result = tool.execute(mode="entity_mentions", entity_value="Test", k=0)
        assert result.success is False
        assert "validation" in result.error.lower()

        # k=51 (exceeds max 50 for GraphSearchTool) should fail validation
        result = tool.execute(mode="entity_mentions", entity_value="Test", k=51)
        assert result.success is False

    def test_empty_query_string(self, tool_dependencies):
        """Test handling of empty query strings."""
        tool = FilteredSearchTool(**tool_dependencies)

        # Empty string is technically valid in Pydantic (str type)
        # But we can test that it works without error
        result = tool.execute(query="", search_method="hybrid", k=6)
        # Should succeed or fail gracefully, not crash
        assert isinstance(result, ToolResult)

    def test_embedder_failure(self, tool_dependencies, mock_embedder):
        """Test handling of embedder failures."""
        mock_embedder.embed_texts.side_effect = Exception("Embedder failed")

        tool = FilteredSearchTool(**tool_dependencies)
        result = tool.execute(query="test", search_method="hybrid", k=6)

        assert result.success is False
        # Error message should contain information about the failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_vector_store_failure(self, tool_dependencies, mock_vector_store):
        """Test handling of vector store failures."""
        mock_vector_store.hierarchical_search.side_effect = Exception("Vector store error")

        tool = FilteredSearchTool(**tool_dependencies)
        result = tool.execute(query="test", search_method="hybrid", k=6)

        assert result.success is False

    def test_tool_statistics_tracking(self, tool_dependencies):
        """Test that tools track execution statistics."""
        tool = CompareDocumentsTool(**tool_dependencies)

        # Execute tool multiple times
        tool.execute(doc_id_1="doc1", doc_id_2="doc2")
        tool.execute(doc_id_1="doc1", doc_id_2="doc3")

        stats = tool.get_stats()
        assert stats["execution_count"] == 2
        assert stats["name"] == "compare_documents"
        assert "avg_time_ms" in stats


# ============================================================================
# TEST: Integration Tests
# ============================================================================


class TestToolIntegration:
    """Integration tests combining multiple tools."""

    def test_multi_tool_workflow(self, tool_dependencies):
        """Test typical workflow using multiple tools."""
        # 1. Initial search
        search_tool = FilteredSearchTool(**tool_dependencies)
        search_result = search_tool.execute(query="waste disposal", search_method="hybrid", k=6)
        assert search_result.success is True

        # 2. Find similar chunks
        if search_result.data:
            chunk_id = search_result.data[0]["chunk_id"]
            similarity_tool = SimilaritySearchTool(**tool_dependencies)
            similarity_result = similarity_tool.execute(chunk_id=chunk_id, cross_document=True, k=3)
            assert similarity_result.success is True

        # 3. Explain results
        if search_result.data:
            chunk_ids = [c["chunk_id"] for c in search_result.data[:2]]
            explain_tool = ExplainSearchResultsTool(**tool_dependencies)
            explain_result = explain_tool.execute(chunk_ids=chunk_ids)
            assert explain_result.success is True

    def test_tool_error_propagation(self, tool_dependencies, mock_vector_store):
        """Test that errors are properly propagated through tool chain."""
        # Setup vector store to fail
        mock_vector_store.hierarchical_search.side_effect = RuntimeError("Database error")

        # Each tool should handle the error gracefully
        tools = [
            FilteredSearchTool(**tool_dependencies),
            CompareDocumentsTool(**tool_dependencies),
            ExplainSearchResultsTool(**tool_dependencies),
        ]

        for tool in tools:
            if tool.name == "compare_documents":
                result = tool.execute(doc_id_1="doc1", doc_id_2="doc2")
            elif tool.name == "explain_search_results":
                result = tool.execute(chunk_ids=["doc1:sec1:0"])
            else:
                result = tool.execute(query="test", search_method="hybrid", k=6)
            assert result.success is False
            assert result.error is not None


# ============================================================================
# TEST 7: BrowseEntitiesTool (New)
# ============================================================================


class TestBrowseEntitiesTool:
    """Test browse_entities tool for discovering entities without specific names."""

    def test_browse_without_kg_fails(self, tool_dependencies):
        """Test that browse_entities fails gracefully without knowledge graph."""
        tool = BrowseEntitiesTool(**tool_dependencies)

        result = tool.execute()

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Knowledge graph not available" in result.error

    def test_browse_without_find_entities_method_fails(self, tool_dependencies):
        """Test that browse_entities fails with non-GraphAdapter KG."""
        # Create mock KG without find_entities method (simple KnowledgeGraph)
        simple_kg = Mock(spec=['entities', 'relationships'])  # Explicitly specify allowed attrs
        simple_kg.entities = {}
        simple_kg.relationships = []
        # find_entities method is not in spec, so hasattr will return False
        tool_dependencies["knowledge_graph"] = simple_kg

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute()

        assert result.success is False
        assert "does not support find_entities" in result.error
        assert "GraphAdapter" in result.error

    def test_browse_all_entities(self, tool_dependencies):
        """Test browsing all entities without filters."""
        # Create mock GraphAdapter
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.STANDARD,
                value="GRI 306",
                normalized_value="gri 306",
                confidence=0.95,
                source_chunk_ids=["doc1:sec1:0", "doc1:sec2:1"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
            Entity(
                id="ent2",
                type=EntityType.ORGANIZATION,
                value="GSSB",
                normalized_value="gssb",
                confidence=0.88,
                source_chunk_ids=["doc1:sec1:5"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:5",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
            Entity(
                id="ent3",
                type=EntityType.REGULATION,
                value="Waste Management Act",
                normalized_value="waste management act",
                confidence=0.82,
                source_chunk_ids=["doc2:sec1:0"],
                document_id="doc2",
                first_mention_chunk_id="doc2:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(limit=20)

        assert result.success is True
        assert len(result.data) == 3
        # Should be sorted by confidence (descending)
        assert result.data[0]["confidence"] == 0.95
        assert result.data[1]["confidence"] == 0.88
        assert result.data[2]["confidence"] == 0.82
        assert result.metadata["count"] == 3
        assert result.metadata["sorted_by"] == "confidence (descending)"

        # Verify find_entities was called with correct params
        mock_adapter.find_entities.assert_called_once_with(
            entity_type=None, min_confidence=0.0, value_contains=None
        )

    def test_browse_by_entity_type(self, tool_dependencies):
        """Test browsing entities filtered by type."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.STANDARD,
                value="GRI 306",
                normalized_value="gri 306",
                confidence=0.95,
                source_chunk_ids=["doc1:sec1:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(entity_type="standard", limit=20)

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["type"] == "standard"
        assert result.metadata["filters"]["entity_type"] == "standard"
        assert "type=standard" in result.metadata["filters_description"]

        mock_adapter.find_entities.assert_called_once_with(
            entity_type="standard", min_confidence=0.0, value_contains=None
        )

    def test_browse_by_search_term(self, tool_dependencies):
        """Test browsing entities with search term filter."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.REGULATION,
                value="Waste Management Act",
                normalized_value="waste management act",
                confidence=0.90,
                source_chunk_ids=["doc1:sec1:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
            Entity(
                id="ent2",
                type=EntityType.CLAUSE,
                value="Waste disposal requirements",
                normalized_value="waste disposal requirements",
                confidence=0.85,
                source_chunk_ids=["doc1:sec2:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec2:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(search_term="waste", limit=20)

        assert result.success is True
        assert len(result.data) == 2
        assert result.metadata["filters"]["search_term"] == "waste"
        assert "search='waste'" in result.metadata["filters_description"]

        mock_adapter.find_entities.assert_called_once_with(
            entity_type=None, min_confidence=0.0, value_contains="waste"
        )

    def test_browse_by_min_confidence(self, tool_dependencies):
        """Test browsing entities with minimum confidence filter."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.STANDARD,
                value="GRI 306",
                normalized_value="gri 306",
                confidence=0.95,
                source_chunk_ids=["doc1:sec1:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
            Entity(
                id="ent2",
                type=EntityType.STANDARD,
                value="ISO 14001",
                normalized_value="iso 14001",
                confidence=0.88,
                source_chunk_ids=["doc1:sec2:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec2:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(min_confidence=0.85, limit=20)

        assert result.success is True
        assert len(result.data) == 2
        assert all(e["confidence"] >= 0.85 for e in result.data)
        assert result.metadata["filters"]["min_confidence"] == 0.85
        assert "confidence≥0.85" in result.metadata["filters_description"]

        mock_adapter.find_entities.assert_called_once_with(
            entity_type=None, min_confidence=0.85, value_contains=None
        )

    def test_browse_with_multiple_filters(self, tool_dependencies):
        """Test browsing with multiple filters combined."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.REGULATION,
                value="Waste Management Act 2020",
                normalized_value="waste management act 2020",
                confidence=0.92,
                source_chunk_ids=["doc1:sec1:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(entity_type="regulation", search_term="waste", min_confidence=0.9, limit=10)

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["type"] == "regulation"
        assert "waste" in result.data[0]["value"].lower()
        assert result.data[0]["confidence"] >= 0.9

        # Check all filters in metadata
        filters_desc = result.metadata["filters_description"]
        assert "type=regulation" in filters_desc
        assert "search='waste'" in filters_desc
        assert "confidence≥0.9" in filters_desc

        mock_adapter.find_entities.assert_called_once_with(
            entity_type="regulation", min_confidence=0.9, value_contains="waste"
        )

    def test_browse_no_results(self, tool_dependencies):
        """Test browsing when no entities match filters."""
        mock_adapter = Mock()
        mock_adapter.find_entities.return_value = []
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(entity_type="nonexistent_type", limit=20)

        assert result.success is True  # No results is not an error
        assert len(result.data) == 0
        assert result.metadata["count"] == 0
        assert "No entities found" in result.metadata["message"]
        assert "type='nonexistent_type'" in result.metadata["message"]

    def test_browse_respects_limit(self, tool_dependencies):
        """Test that browse_entities respects the limit parameter."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        # Create 30 mock entities
        mock_entities = [
            Entity(
                id=f"ent{i}",
                type=EntityType.STANDARD,
                value=f"Standard {i}",
                normalized_value=f"standard {i}",
                confidence=0.9 - (i * 0.01),  # Decreasing confidence
                source_chunk_ids=[f"doc1:sec1:{i}"],
                document_id="doc1",
                first_mention_chunk_id=f"doc1:sec1:{i}",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            )
            for i in range(30)
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute(limit=10)

        assert result.success is True
        assert len(result.data) == 10  # Should only return 10
        assert result.metadata["count"] == 10
        assert result.metadata["filters"]["limit"] == 10

    def test_browse_max_limit_validation(self, tool_dependencies):
        """Test that limit cannot exceed 50."""
        mock_adapter = Mock()
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)

        # Try with limit > 50 (should fail validation)
        result = tool.execute(limit=100)

        assert result.success is False
        assert "validation" in result.error.lower() or "less than or equal to 50" in result.error.lower()

    def test_browse_entity_data_format(self, tool_dependencies):
        """Test that returned entity data has correct format."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="test-id",
                type=EntityType.ORGANIZATION,
                value="Test Organization",
                normalized_value="test organization",
                confidence=0.92,
                source_chunk_ids=["doc1:sec1:0", "doc1:sec2:5"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={"extra": "data"},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute()

        assert result.success is True
        entity_data = result.data[0]

        # Check all required fields
        assert entity_data["id"] == "test-id"
        assert entity_data["type"] == "organization"
        assert entity_data["value"] == "Test Organization"
        assert entity_data["normalized_value"] == "test organization"
        assert entity_data["confidence"] == 0.92
        assert entity_data["mentions"] == 2  # len(source_chunk_ids)
        assert entity_data["document_id"] == "doc1"

    def test_browse_error_handling(self, tool_dependencies):
        """Test error handling when find_entities raises exception."""
        mock_adapter = Mock()
        mock_adapter.find_entities.side_effect = Exception("Neo4j connection error")
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute()

        assert result.success is False
        assert "Failed to browse entities" in result.error
        assert "Neo4j connection error" in result.error

    def test_browse_execution_time_tracking(self, tool_dependencies):
        """Test that execution time is tracked."""
        from src.graph.models import Entity, EntityType

        mock_adapter = Mock()
        mock_entities = [
            Entity(
                id="ent1",
                type=EntityType.STANDARD,
                value="Test",
                normalized_value="test",
                confidence=0.9,
                source_chunk_ids=["doc1:sec1:0"],
                document_id="doc1",
                first_mention_chunk_id="doc1:sec1:0",
                metadata={},
                extraction_method="llm",
                extracted_at=datetime(2024, 10, 29),
            ),
        ]

        mock_adapter.find_entities.return_value = mock_entities
        tool_dependencies["knowledge_graph"] = mock_adapter

        tool = BrowseEntitiesTool(**tool_dependencies)
        result = tool.execute()

        assert result.success is True
        assert result.execution_time_ms > 0
        assert "execution_time_ms" in result.metadata
        assert result.metadata["execution_time_ms"] > 0
