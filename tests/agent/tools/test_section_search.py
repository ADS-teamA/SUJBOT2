"""
Tests for SectionSearchTool - Layer 2 Section-Level Search.

Tests cover:
- Input validation (including cross-field validation)
- FusionRetriever integration
- Section filtering (path prefix, level range)
- Chunk count enrichment
- Error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import ValidationError

from src.agent.tools.section_search import (
    SectionSearchInput,
    SectionSearchTool,
    SectionSearchResult,
)
from src.agent.tools._base import ToolResult


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestSectionSearchInput:
    """Test input schema validation."""

    def test_valid_input_minimal(self):
        """Test minimal valid input."""
        input_data = SectionSearchInput(query="What is GDPR?")
        assert input_data.query == "What is GDPR?"
        assert input_data.k == 5  # default
        assert input_data.document_filter is None
        assert input_data.section_path_prefix is None
        assert input_data.min_section_level is None
        assert input_data.max_section_level is None

    def test_valid_input_full(self):
        """Test full valid input with all parameters."""
        input_data = SectionSearchInput(
            query="Privacy requirements",
            k=10,
            document_filter="doc_123",
            section_path_prefix="Chapter 3 > ",
            min_section_level=1,
            max_section_level=3,
        )
        assert input_data.query == "Privacy requirements"
        assert input_data.k == 10
        assert input_data.document_filter == "doc_123"
        assert input_data.section_path_prefix == "Chapter 3 > "
        assert input_data.min_section_level == 1
        assert input_data.max_section_level == 3

    def test_query_required_and_non_empty(self):
        """Test that query is required and cannot be empty."""
        with pytest.raises(ValidationError):
            SectionSearchInput()  # query missing

        with pytest.raises(ValidationError):
            SectionSearchInput(query="")  # empty query

    def test_k_bounds(self):
        """Test k validation (1-50)."""
        # Valid bounds
        SectionSearchInput(query="test", k=1)
        SectionSearchInput(query="test", k=50)

        # Invalid bounds
        with pytest.raises(ValidationError):
            SectionSearchInput(query="test", k=0)

        with pytest.raises(ValidationError):
            SectionSearchInput(query="test", k=51)

    def test_section_level_bounds(self):
        """Test section level validation (1-10)."""
        # Valid bounds
        SectionSearchInput(query="test", min_section_level=1)
        SectionSearchInput(query="test", max_section_level=10)

        # Invalid bounds
        with pytest.raises(ValidationError):
            SectionSearchInput(query="test", min_section_level=0)

        with pytest.raises(ValidationError):
            SectionSearchInput(query="test", max_section_level=11)

    def test_cross_field_validation_min_max_level(self):
        """Test that min_section_level <= max_section_level."""
        # Valid: min < max
        input_data = SectionSearchInput(
            query="test",
            min_section_level=1,
            max_section_level=3,
        )
        assert input_data.min_section_level < input_data.max_section_level

        # Valid: min == max
        input_data = SectionSearchInput(
            query="test",
            min_section_level=2,
            max_section_level=2,
        )
        assert input_data.min_section_level == input_data.max_section_level

        # Invalid: min > max
        with pytest.raises(ValidationError) as exc_info:
            SectionSearchInput(
                query="test",
                min_section_level=5,
                max_section_level=2,
            )
        assert "min_section_level" in str(exc_info.value)
        assert "cannot be greater than" in str(exc_info.value)

    def test_cross_field_validation_only_applies_when_both_set(self):
        """Test that cross-field validation only runs when both levels are set."""
        # Only min set - should be valid
        input_data = SectionSearchInput(query="test", min_section_level=5)
        assert input_data.min_section_level == 5

        # Only max set - should be valid
        input_data = SectionSearchInput(query="test", max_section_level=2)
        assert input_data.max_section_level == 2


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestSectionSearchTool:
    """Test SectionSearchTool execution."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store with Layer 2 and Layer 3 metadata."""
        store = Mock()
        store.metadata_layer2 = [
            {
                "document_id": "doc_123",
                "section_id": "sec_1",
                "section_title": "Chapter 1",
                "section_path": "Chapter 1",
                "section_level": 1,
            },
            {
                "document_id": "doc_123",
                "section_id": "sec_2",
                "section_title": "Chapter 2",
                "section_path": "Chapter 2",
                "section_level": 1,
            },
        ]
        store.metadata_layer3 = [
            {"document_id": "doc_123", "section_id": "sec_1", "chunk_id": "c1"},
            {"document_id": "doc_123", "section_id": "sec_1", "chunk_id": "c2"},
        ]
        return store

    @pytest.fixture
    def mock_fusion_retriever(self):
        """Create mock FusionRetriever."""
        retriever = Mock()
        retriever.search_layer2 = Mock(return_value=[
            {
                "section_id": "sec_1",
                "document_id": "doc_123",
                "section_title": "Chapter 1",
                "section_path": "Chapter 1",
                "section_level": 1,
                "content": "Chapter 1 content summary...",
                "score": 0.95,
            },
            {
                "section_id": "sec_2",
                "document_id": "doc_123",
                "section_title": "Chapter 2",
                "section_path": "Chapter 2",
                "section_level": 1,
                "content": "Chapter 2 content summary...",
                "score": 0.85,
            },
        ])
        return retriever

    @pytest.fixture
    def tool(self, mock_vector_store, mock_fusion_retriever):
        """Create tool instance with mocks."""
        mock_embedder = Mock()
        with patch(
            "src.agent.tools.section_search.create_fusion_retriever",
            return_value=mock_fusion_retriever
        ):
            tool = SectionSearchTool(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
            )
            tool._fusion_retriever = mock_fusion_retriever
            return tool

    def test_execute_returns_sections_with_scores(self, tool, mock_fusion_retriever):
        """Test basic search returns sections with fusion scores."""
        result = tool.execute_impl(query="Chapter content", k=5)

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2
        # Check first result has expected fields
        # Note: format_chunk_result() transforms the output
        first_result = result.data[0]
        assert "section_title" in first_result
        assert "score" in first_result
        assert first_result["score"] == 0.95
        # section_level and section_path are augmented
        assert "section_level" in first_result
        assert "section_path" in first_result

    def test_execute_with_document_filter(self, tool, mock_fusion_retriever):
        """Test search with document filter."""
        result = tool.execute_impl(
            query="Chapter content",
            document_filter="doc_123",
        )

        assert result.success is True
        # Verify filter was passed to retriever
        mock_fusion_retriever.search_layer2.assert_called_once()
        call_kwargs = mock_fusion_retriever.search_layer2.call_args[1]
        assert call_kwargs["document_filter"] == "doc_123"

    def test_execute_generates_citations(self, tool):
        """Test that citations are generated for results."""
        result = tool.execute_impl(query="Test query", k=5)

        assert result.success is True
        assert result.citations is not None
        assert len(result.citations) == len(result.data)

    def test_execute_includes_metadata(self, tool):
        """Test that result metadata contains search parameters."""
        result = tool.execute_impl(
            query="Test query",
            k=10,
            document_filter="doc_123",
        )

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["query"] == "Test query"
        assert result.metadata["k"] == 10
        assert result.metadata["document_filter"] == "doc_123"


# =============================================================================
# Section Filtering Tests
# =============================================================================


class TestSectionFiltering:
    """Test section path and level filtering."""

    @pytest.fixture
    def tool_with_hierarchy(self):
        """Create tool with hierarchical sections."""
        store = Mock()
        store.metadata_layer3 = []
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)
        return tool

    def test_filter_by_path_prefix(self, tool_with_hierarchy):
        """Test filtering sections by path prefix."""
        sections = [
            {"section_path": "Chapter 1", "section_level": 1},
            {"section_path": "Chapter 1 > Section 1.1", "section_level": 2},
            {"section_path": "Chapter 2", "section_level": 1},
            {"section_path": "Chapter 2 > Section 2.1", "section_level": 2},
        ]

        filtered = tool_with_hierarchy._apply_section_filters(
            sections, section_path_prefix="Chapter 1"
        )

        assert len(filtered) == 2
        assert all("Chapter 1" in s["section_path"] for s in filtered)

    def test_filter_by_min_level(self, tool_with_hierarchy):
        """Test filtering by minimum section level."""
        sections = [
            {"section_path": "Chapter 1", "section_level": 1},
            {"section_path": "Section 1.1", "section_level": 2},
            {"section_path": "SubSection 1.1.1", "section_level": 3},
        ]

        filtered = tool_with_hierarchy._apply_section_filters(
            sections, min_level=2
        )

        assert len(filtered) == 2
        assert all(s["section_level"] >= 2 for s in filtered)

    def test_filter_by_max_level(self, tool_with_hierarchy):
        """Test filtering by maximum section level."""
        sections = [
            {"section_path": "Chapter 1", "section_level": 1},
            {"section_path": "Section 1.1", "section_level": 2},
            {"section_path": "SubSection 1.1.1", "section_level": 3},
        ]

        filtered = tool_with_hierarchy._apply_section_filters(
            sections, max_level=2
        )

        assert len(filtered) == 2
        assert all(s["section_level"] <= 2 for s in filtered)

    def test_filter_combined(self, tool_with_hierarchy):
        """Test combining multiple filters."""
        sections = [
            {"section_path": "Chapter 1", "section_level": 1},
            {"section_path": "Chapter 1 > Section 1.1", "section_level": 2},
            {"section_path": "Chapter 1 > Section 1.1 > Sub", "section_level": 3},
            {"section_path": "Chapter 2", "section_level": 1},
        ]

        filtered = tool_with_hierarchy._apply_section_filters(
            sections,
            section_path_prefix="Chapter 1",
            min_level=2,
            max_level=2,
        )

        assert len(filtered) == 1
        assert filtered[0]["section_path"] == "Chapter 1 > Section 1.1"


# =============================================================================
# Chunk Count Enrichment Tests
# =============================================================================


class TestChunkCountEnrichment:
    """Test chunk count enrichment from Layer 3."""

    def test_enrichment_adds_chunk_counts(self):
        """Test that chunk counts are added from Layer 3 metadata."""
        store = Mock()
        store.metadata_layer3 = [
            {"document_id": "doc1", "section_id": "sec1", "chunk_id": "c1"},
            {"document_id": "doc1", "section_id": "sec1", "chunk_id": "c2"},
            {"document_id": "doc1", "section_id": "sec2", "chunk_id": "c3"},
        ]
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        sections = [
            {"section_id": "sec1", "document_id": "doc1"},
            {"section_id": "sec2", "document_id": "doc1"},
        ]

        enriched = tool._enrich_with_chunk_counts(sections)

        assert enriched[0]["chunk_count"] == 2  # sec1 has 2 chunks
        assert enriched[1]["chunk_count"] == 1  # sec2 has 1 chunk

    def test_enrichment_unknown_when_no_metadata(self):
        """Test that chunk_count is -1 when Layer 3 metadata unavailable."""
        store = Mock()
        store.metadata_layer3 = None  # No metadata
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        sections = [{"section_id": "sec1", "document_id": "doc1"}]

        enriched = tool._enrich_with_chunk_counts(sections)

        assert enriched[0]["chunk_count"] == -1  # -1 = unknown

    def test_enrichment_handles_error_gracefully(self):
        """Test graceful handling when enrichment fails."""
        store = Mock()
        # Make metadata_layer3 raise TypeError
        type(store).metadata_layer3 = property(
            lambda self: (_ for _ in ()).throw(TypeError("Test error"))
        )
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        sections = [{"section_id": "sec1", "document_id": "doc1"}]

        enriched = tool._enrich_with_chunk_counts(sections)

        # Should mark as unknown, not crash
        assert enriched[0]["chunk_count"] == -1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSectionSearchErrorHandling:
    """Test error handling in SectionSearchTool."""

    def test_connection_error_returns_failure(self):
        """Test that ConnectionError is handled gracefully."""
        store = Mock()
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        with patch(
            "src.agent.tools.section_search.create_fusion_retriever",
            side_effect=ConnectionError("Database unavailable")
        ):
            result = tool.execute_impl(query="test")

        assert result.success is False
        assert "connection" in result.error.lower()

    def test_configuration_error_returns_failure(self):
        """Test that ValueError (config error) is handled gracefully."""
        store = Mock()
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        with patch(
            "src.agent.tools.section_search.create_fusion_retriever",
            side_effect=ValueError("Missing API key")
        ):
            result = tool.execute_impl(query="test")

        assert result.success is False
        assert "Configuration error" in result.error

    def test_unexpected_error_logged_and_returned(self):
        """Test that unexpected errors are logged and returned."""
        store = Mock()
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        with patch(
            "src.agent.tools.section_search.create_fusion_retriever",
            side_effect=RuntimeError("Unexpected error")
        ):
            result = tool.execute_impl(query="test")

        assert result.success is False
        assert "RuntimeError" in result.error


# =============================================================================
# TypedDict Compliance Tests
# =============================================================================


class TestSectionSearchResultTypedDict:
    """Test that results match SectionSearchResult TypedDict."""

    def test_result_has_expected_fields(self):
        """Test that results include all TypedDict fields."""
        store = Mock()
        store.metadata_layer3 = []
        mock_embedder = Mock()
        tool = SectionSearchTool(vector_store=store, embedder=mock_embedder)

        mock_retriever = Mock()
        mock_retriever.search_layer2 = Mock(return_value=[
            {
                "chunk_id": "sec1",
                "section_id": "sec1",
                "document_id": "doc1",
                "section_title": "Test Section",
                "section_path": "Test Section",
                "section_level": 1,
                "content": "Content...",
                "score": 0.9,
            },
        ])

        with patch(
            "src.agent.tools.section_search.create_fusion_retriever",
            return_value=mock_retriever
        ):
            tool._fusion_retriever = mock_retriever
            result = tool.execute_impl(query="test")

        assert result.success is True
        section = result.data[0]

        # Verify expected fields are present
        # Note: format_chunk_result transforms output, section_id becomes chunk_id
        assert "section_level" in section
        assert "section_path" in section
        assert "chunk_count" in section
        assert "section_title" in section
        assert "score" in section


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
