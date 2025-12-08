"""
Tests for BrowseSectionsTool - Hierarchical Section Navigation.

Tests cover:
- Input validation (Pydantic model with Literal type)
- Tree building from flat section list
- Section filtering (path, level)
- Chunk count enrichment (including -1 for unknown)
- Error handling (ToolExecutionError propagation)
"""

import pytest
from unittest.mock import Mock, MagicMock
from pydantic import ValidationError

from src.agent.tools.browse_sections import (
    BrowseSectionsInput,
    BrowseSectionsTool,
    SectionNode,
    BrowseSectionsResult,
)
from src.agent.tools._base import ToolResult
from src.exceptions import ToolExecutionError


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestBrowseSectionsInput:
    """Test input schema validation."""

    def test_valid_input_minimal(self):
        """Test minimal valid input."""
        input_data = BrowseSectionsInput(document_id="doc_123")
        assert input_data.document_id == "doc_123"
        assert input_data.parent_section_path is None
        assert input_data.max_depth == 2  # default
        assert input_data.include_summaries is False  # default
        assert input_data.sort_by == "path"  # default

    def test_valid_input_full(self):
        """Test full valid input with all parameters."""
        input_data = BrowseSectionsInput(
            document_id="doc_123",
            parent_section_path="Chapter 1",
            max_depth=3,
            include_summaries=True,
            sort_by="page",
        )
        assert input_data.document_id == "doc_123"
        assert input_data.parent_section_path == "Chapter 1"
        assert input_data.max_depth == 3
        assert input_data.include_summaries is True
        assert input_data.sort_by == "page"

    def test_sort_by_literal_validation(self):
        """Test that sort_by only accepts valid Literal values."""
        # Valid values
        for sort_by in ["path", "page", "size"]:
            input_data = BrowseSectionsInput(document_id="doc", sort_by=sort_by)
            assert input_data.sort_by == sort_by

        # Invalid value should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            BrowseSectionsInput(document_id="doc", sort_by="invalid")
        assert "sort_by" in str(exc_info.value)

    def test_max_depth_bounds(self):
        """Test max_depth validation (1-5)."""
        # Valid bounds
        BrowseSectionsInput(document_id="doc", max_depth=1)
        BrowseSectionsInput(document_id="doc", max_depth=5)

        # Invalid bounds
        with pytest.raises(ValidationError):
            BrowseSectionsInput(document_id="doc", max_depth=0)

        with pytest.raises(ValidationError):
            BrowseSectionsInput(document_id="doc", max_depth=6)

    def test_document_id_required(self):
        """Test that document_id is required."""
        with pytest.raises(ValidationError) as exc_info:
            BrowseSectionsInput()
        assert "document_id" in str(exc_info.value)


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestBrowseSectionsTool:
    """Test BrowseSectionsTool execution."""

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
                "page_number": 1,
            },
            {
                "document_id": "doc_123",
                "section_id": "sec_1_1",
                "section_title": "Section 1.1",
                "section_path": "Chapter 1 > Section 1.1",
                "section_level": 2,
                "page_number": 5,
            },
            {
                "document_id": "doc_123",
                "section_id": "sec_2",
                "section_title": "Chapter 2",
                "section_path": "Chapter 2",
                "section_level": 1,
                "page_number": 20,
            },
        ]
        store.metadata_layer3 = [
            {"document_id": "doc_123", "section_id": "sec_1", "chunk_id": "c1"},
            {"document_id": "doc_123", "section_id": "sec_1", "chunk_id": "c2"},
            {"document_id": "doc_123", "section_id": "sec_1_1", "chunk_id": "c3"},
        ]
        return store

    @pytest.fixture
    def tool(self, mock_vector_store):
        """Create tool instance with mock vector store."""
        mock_embedder = Mock()
        tool = BrowseSectionsTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        return tool

    def test_execute_returns_top_level_sections(self, tool):
        """Test that execute returns top-level sections for document."""
        result = tool.execute_impl(document_id="doc_123", max_depth=1)

        assert result.success is True
        assert result.data is not None
        assert result.data["document_id"] == "doc_123"
        assert len(result.data["sections"]) == 2  # Chapter 1, Chapter 2

    def test_execute_with_parent_path(self, tool):
        """Test drilling down into a parent section."""
        result = tool.execute_impl(
            document_id="doc_123",
            parent_section_path="Chapter 1",
            max_depth=1,
        )

        assert result.success is True
        # Should return children of Chapter 1
        sections = result.data["sections"]
        assert len(sections) >= 1
        # Section 1.1 is a child of Chapter 1
        section_paths = [s["section_path"] for s in sections]
        assert any("Section 1.1" in p for p in section_paths)

    def test_execute_with_chunk_counts(self, tool):
        """Test that chunk counts are added from Layer 3."""
        result = tool.execute_impl(document_id="doc_123", max_depth=2)

        assert result.success is True
        sections = result.data["sections"]

        # Find Chapter 1 and check chunk count
        chapter1 = next(
            (s for s in sections if s["section_title"] == "Chapter 1"), None
        )
        assert chapter1 is not None
        assert chapter1["chunk_count"] == 2  # sec_1 has 2 chunks

    def test_execute_empty_document(self, tool):
        """Test with document that has no sections."""
        result = tool.execute_impl(document_id="nonexistent_doc")

        assert result.success is True
        assert result.data["sections"] == []
        assert result.data["total_sections"] == 0

    def test_execute_sort_by_page(self, tool, mock_vector_store):
        """Test sorting by page number."""
        # Add section with later page but earlier in path
        mock_vector_store.metadata_layer2.append({
            "document_id": "doc_123",
            "section_id": "sec_0",
            "section_title": "Preface",
            "section_path": "Preface",
            "section_level": 1,
            "page_number": 100,  # Late page
        })

        result = tool.execute_impl(
            document_id="doc_123", max_depth=1, sort_by="page"
        )

        assert result.success is True
        sections = result.data["sections"]
        page_numbers = [s.get("page_number", 0) for s in sections]
        # Should be sorted by page
        assert page_numbers == sorted(page_numbers)

    def test_execute_sort_by_size(self, tool):
        """Test sorting by chunk count (size)."""
        result = tool.execute_impl(
            document_id="doc_123", max_depth=1, sort_by="size"
        )

        assert result.success is True
        sections = result.data["sections"]
        chunk_counts = [s.get("chunk_count", 0) for s in sections]
        # Should be sorted descending (largest first)
        assert chunk_counts == sorted(chunk_counts, reverse=True)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBrowseSectionsErrorHandling:
    """Test error handling in BrowseSectionsTool."""

    def test_vector_store_not_initialized(self):
        """Test error when vector store is None."""
        mock_embedder = Mock()
        tool = BrowseSectionsTool(vector_store=None, embedder=mock_embedder)

        result = tool.execute_impl(document_id="doc_123")

        assert result.success is False
        assert "Vector store not initialized" in result.error

    def test_chunk_count_enrichment_failure(self):
        """Test graceful handling when chunk count enrichment fails."""
        store = Mock()
        store.metadata_layer2 = [
            {
                "document_id": "doc_123",
                "section_id": "sec_1",
                "section_title": "Chapter 1",
                "section_path": "Chapter 1",
                "section_level": 1,
            },
        ]
        store.metadata_layer3 = None  # Will cause enrichment to fail
        mock_embedder = Mock()
        tool = BrowseSectionsTool(vector_store=store, embedder=mock_embedder)

        result = tool.execute_impl(document_id="doc_123")

        assert result.success is True
        # Chunk count should be -1 (unknown) not 0
        sections = result.data["sections"]
        assert sections[0]["chunk_count"] == -1

    def test_metadata_access_error_propagates(self):
        """Test that metadata access errors are properly handled."""
        store = Mock()
        # Make metadata_layer2 raise an error
        type(store).metadata_layer2 = property(
            lambda self: (_ for _ in ()).throw(TypeError("Test error"))
        )
        mock_embedder = Mock()
        tool = BrowseSectionsTool(vector_store=store, embedder=mock_embedder)

        result = tool.execute_impl(document_id="doc_123")

        # Should return error, not crash
        assert result.success is False
        assert "TypeError" in result.error


# =============================================================================
# Tree Building Tests
# =============================================================================


class TestTreeBuilding:
    """Test hierarchical tree building logic."""

    @pytest.fixture
    def tool_with_deep_hierarchy(self):
        """Create tool with deeply nested sections."""
        store = Mock()
        store.metadata_layer2 = [
            {"document_id": "doc", "section_id": "1", "section_title": "Ch 1",
             "section_path": "Ch 1", "section_level": 1},
            {"document_id": "doc", "section_id": "1.1", "section_title": "Sec 1.1",
             "section_path": "Ch 1 > Sec 1.1", "section_level": 2},
            {"document_id": "doc", "section_id": "1.1.1", "section_title": "SubSec 1.1.1",
             "section_path": "Ch 1 > Sec 1.1 > SubSec 1.1.1", "section_level": 3},
            {"document_id": "doc", "section_id": "1.1.2", "section_title": "SubSec 1.1.2",
             "section_path": "Ch 1 > Sec 1.1 > SubSec 1.1.2", "section_level": 3},
        ]
        store.metadata_layer3 = []
        mock_embedder = Mock()
        tool = BrowseSectionsTool(vector_store=store, embedder=mock_embedder)
        return tool

    def test_max_depth_limits_recursion(self, tool_with_deep_hierarchy):
        """Test that max_depth limits tree depth."""
        result = tool_with_deep_hierarchy.execute_impl(
            document_id="doc", max_depth=1
        )

        assert result.success is True
        sections = result.data["sections"]
        # Should have Ch 1 but NOT recurse into children
        assert len(sections) == 1
        ch1 = sections[0]
        assert ch1["has_children"] is True
        assert "children" not in ch1  # max_depth=1 means no children

    def test_full_depth_returns_all_levels(self, tool_with_deep_hierarchy):
        """Test that max_depth=5 returns full hierarchy."""
        result = tool_with_deep_hierarchy.execute_impl(
            document_id="doc", max_depth=5
        )

        assert result.success is True
        ch1 = result.data["sections"][0]
        assert "children" in ch1
        sec_1_1 = ch1["children"][0]
        assert "children" in sec_1_1
        assert len(sec_1_1["children"]) == 2  # SubSec 1.1.1 and 1.1.2

    def test_children_count_accurate(self, tool_with_deep_hierarchy):
        """Test that children_count reflects actual children."""
        result = tool_with_deep_hierarchy.execute_impl(
            document_id="doc", max_depth=5
        )

        ch1 = result.data["sections"][0]
        assert ch1["children_count"] == 1  # Only Sec 1.1 is direct child

        sec_1_1 = ch1["children"][0]
        assert sec_1_1["children_count"] == 2  # SubSec 1.1.1 and 1.1.2


# =============================================================================
# TypedDict Compliance Tests
# =============================================================================


class TestTypedDictCompliance:
    """Test that return types match TypedDict definitions."""

    def test_section_node_has_required_fields(self):
        """Test that SectionNode TypedDict fields are present."""
        store = Mock()
        store.metadata_layer2 = [
            {
                "document_id": "doc",
                "section_id": "sec1",
                "section_title": "Test Section",
                "section_path": "Test Section",
                "section_level": 1,
                "page_number": 5,
            },
        ]
        store.metadata_layer3 = []
        mock_embedder = Mock()
        tool = BrowseSectionsTool(vector_store=store, embedder=mock_embedder)

        result = tool.execute_impl(document_id="doc")

        section = result.data["sections"][0]
        # Check all TypedDict fields exist
        assert "section_id" in section
        assert "section_title" in section
        assert "section_path" in section
        assert "section_level" in section
        assert "has_children" in section
        assert "children_count" in section
        assert "chunk_count" in section


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
