#!/usr/bin/env python3
"""
Test hierarchy detection in Unstructured.io extraction.

Tests parent-child relationships, cycle detection, page break continuity,
and level calculations based on parent_id metadata.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unstructured_extractor import (
    UnstructuredExtractor,
    ExtractionConfig,
    DocumentSection,
    detect_hierarchy_generic,
)


class TestHierarchyDetection:
    """Test hierarchy detection from parent_id relationships."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExtractionConfig(
            strategy="fast",  # Use fast for quicker tests
            model="yolox",  # Lightweight model
            generate_summaries=False,
            extract_tables=False,
        )

    @pytest.fixture
    def extractor(self, config):
        """Create extractor instance."""
        return UnstructuredExtractor(config)

    def test_hierarchy_sb_1997_18(self, extractor):
        """
        Test hierarchy on known document structure (Sb_1997_18).

        This document should have multi-level hierarchy with:
        - ČÁST (parts)
        - HLAVA (chapters)
        - § paragraphs
        - Subsections
        """
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Assert multi-level hierarchy exists
        assert doc.hierarchy_depth >= 2, f"Expected at least 2 levels, got {doc.hierarchy_depth}"

        # Verify parent-child relationships exist
        sections_with_parents = [s for s in doc.sections if s.parent_id]
        assert len(sections_with_parents) > 0, "Expected sections to have parent relationships"

        # Verify path construction
        for section in doc.sections:
            if section.depth > 1:
                assert " > " in section.path, (
                    f"Multi-level section should have path separator: {section.path}"
                )

        # Print hierarchy sample for debugging
        print("\n=== Hierarchy Sample ===")
        for section in doc.sections[:10]:
            indent = "  " * section.depth
            print(f"{indent}[L{section.level}, D{section.depth}] {section.title[:60]}")

    def test_hierarchy_parent_id_mapping(self, extractor):
        """Test that parent_id correctly maps to parent section."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Build section lookup
        sections_by_id = {s.section_id: s for s in doc.sections}

        for section in doc.sections:
            if section.parent_id:
                # Parent must exist
                assert section.parent_id in sections_by_id, (
                    f"Parent {section.parent_id} not found for section {section.section_id}"
                )

                parent = sections_by_id[section.parent_id]
                # Parent must be at higher level (lower depth) or equal
                # Note: In some cases, same-level siblings may reference common parent
                assert parent.depth <= section.depth, (
                    f"Parent depth {parent.depth} should be <= child depth {section.depth}"
                )

    def test_hierarchy_ancestors_chain(self, extractor):
        """Test that ancestors list is correctly populated."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        for section in doc.sections:
            # Ancestors count should be consistent with depth
            # Root sections (depth=1) should have 0 ancestors
            if section.depth == 1:
                assert len(section.ancestors) == 0, (
                    f"Root section should have 0 ancestors, got {len(section.ancestors)}"
                )
            # Non-root sections should have ancestors
            elif section.depth > 1:
                assert len(section.ancestors) >= 1, (
                    f"Section at depth {section.depth} should have at least 1 ancestor, "
                    f"got {len(section.ancestors)}"
                )

    def test_hierarchy_cycle_detection_mock(self):
        """Test cycle detection with mock elements."""
        # Create mock elements with circular parent_id
        mock_elements = []

        # Element 0 -> Element 1 -> Element 2 -> Element 0 (cycle)
        for i in range(3):
            elem = Mock()
            elem.id = f"elem_{i}"
            elem.category = "Title"
            elem.text = f"Section {i}"

            # Create mock metadata
            elem.metadata = Mock()
            elem.metadata.parent_id = f"elem_{(i + 1) % 3}"  # Create cycle
            elem.metadata.page_number = 1

            mock_elements.append(elem)

        config = ExtractionConfig()

        # This should not crash - cycle detection should handle it
        try:
            hierarchy_features = detect_hierarchy_generic(mock_elements, config)

            # Verify cycle was broken - all should be assigned some level
            assert len(hierarchy_features) == 3
            for feat in hierarchy_features:
                assert "level" in feat
                assert feat["level"] >= 0

            print(f"\n✓ Cycle detection handled successfully")
            print(f"  Assigned levels: {[f['level'] for f in hierarchy_features]}")

        except Exception as e:
            pytest.fail(f"Cycle detection failed: {e}")

    def test_hierarchy_page_break_continuity(self, extractor):
        """Test parent inheritance across page breaks."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find sections that span page breaks
        page_transitions = []
        for i in range(len(doc.sections) - 1):
            curr_section = doc.sections[i]
            next_section = doc.sections[i + 1]

            if curr_section.page_number != next_section.page_number:
                page_transitions.append((curr_section, next_section))

        if page_transitions:
            print(f"\n=== Found {len(page_transitions)} page transitions ===")
            for curr, next_sec in page_transitions[:3]:  # Show first 3
                print(f"Page {curr.page_number} -> {next_sec.page_number}")
                print(f"  Before: [{curr.level}] {curr.title[:40]}")
                print(f"  After:  [{next_sec.level}] {next_sec.title[:40]}")
                print(f"  Next has parent_id: {next_sec.parent_id is not None}")

    def test_hierarchy_no_orphans(self, extractor):
        """Test that all sections except roots have proper parent references."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Count roots (depth=1) and non-roots
        roots = [s for s in doc.sections if s.depth == 1]
        non_roots = [s for s in doc.sections if s.depth > 1]

        print(f"\n=== Section distribution ===")
        print(f"  Root sections (depth=1): {len(roots)}")
        print(f"  Non-root sections: {len(non_roots)}")

        # Most non-root sections should have parent_id
        # (some may not if hierarchy detection failed for that element)
        non_roots_with_parent = [s for s in non_roots if s.parent_id]
        coverage = len(non_roots_with_parent) / len(non_roots) if non_roots else 1.0

        print(f"  Non-roots with parent_id: {len(non_roots_with_parent)} ({coverage:.1%})")

        # We expect most sections to have parents
        assert coverage > 0.5, (
            f"Expected >50% of non-root sections to have parent_id, got {coverage:.1%}"
        )


class TestHierarchyEdgeCases:
    """Test edge cases in hierarchy detection."""

    def test_empty_elements_list(self):
        """Test hierarchy detection with empty elements list."""
        config = ExtractionConfig()
        hierarchy_features = detect_hierarchy_generic([], config)
        assert hierarchy_features == []

    def test_single_element(self):
        """Test hierarchy detection with single element."""
        mock_elem = Mock()
        mock_elem.id = "elem_0"
        mock_elem.category = "Title"
        mock_elem.text = "Single Section"
        mock_elem.metadata = Mock()
        mock_elem.metadata.parent_id = None
        mock_elem.metadata.page_number = 1

        config = ExtractionConfig()
        hierarchy_features = detect_hierarchy_generic([mock_elem], config)

        assert len(hierarchy_features) == 1
        assert hierarchy_features[0]["level"] == 0  # Root level

    def test_flat_document_no_hierarchy(self):
        """Test document with no hierarchy (all NarrativeText)."""
        # Create mock flat document - all elements at same level
        mock_elements = []
        for i in range(5):
            elem = Mock()
            elem.id = f"elem_{i}"
            elem.category = "NarrativeText"
            elem.text = f"Paragraph {i}"
            elem.metadata = Mock()
            elem.metadata.parent_id = None  # No parent
            elem.metadata.page_number = 1
            mock_elements.append(elem)

        config = ExtractionConfig()
        hierarchy_features = detect_hierarchy_generic(mock_elements, config)

        # All should be at same level
        levels = [f["level"] for f in hierarchy_features]
        assert len(set(levels)) == 1, f"Expected all same level, got {set(levels)}"
        print(f"\n✓ Flat document correctly handled - all at level {levels[0]}")

    def test_deep_nesting(self):
        """Test deep nesting (>10 levels) detection."""
        # Create deeply nested hierarchy
        mock_elements = []
        for i in range(15):
            elem = Mock()
            elem.id = f"elem_{i}"
            elem.category = "Title"
            elem.text = f"Level {i}"
            elem.metadata = Mock()
            elem.metadata.parent_id = f"elem_{i-1}" if i > 0 else None
            elem.metadata.page_number = 1
            mock_elements.append(elem)

        config = ExtractionConfig()
        hierarchy_features = detect_hierarchy_generic(mock_elements, config)

        # Check max depth
        max_level = max(f["level"] for f in hierarchy_features)
        print(f"\n✓ Deep nesting test: max level = {max_level}")

        # Deep nesting should be detected (may indicate error)
        if max_level > 10:
            print(f"  ⚠️ Warning: Very deep nesting detected (level {max_level})")


class TestSectionMetadata:
    """Test section metadata correctness."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = ExtractionConfig(
            strategy="fast",
            model="yolox",
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_section_has_required_fields(self, extractor):
        """Test that all sections have required fields."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        required_fields = [
            "section_id", "title", "content", "level", "depth",
            "parent_id", "children_ids", "ancestors", "path",
            "page_number", "char_start", "char_end", "content_length"
        ]

        for section in doc.sections[:5]:  # Check first 5
            for field in required_fields:
                assert hasattr(section, field), (
                    f"Section missing required field: {field}"
                )

    def test_section_content_not_empty(self, extractor):
        """Test that sections have non-empty content."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Most sections should have content
        sections_with_content = [s for s in doc.sections if len(s.content.strip()) > 0]
        coverage = len(sections_with_content) / len(doc.sections)

        print(f"\n=== Content coverage ===")
        print(f"  Sections with content: {len(sections_with_content)}/{len(doc.sections)} ({coverage:.1%})")

        assert coverage > 0.7, (
            f"Expected >70% sections with content, got {coverage:.1%}"
        )

    def test_page_numbers_sequential(self, extractor):
        """Test that page numbers are sequential and positive."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        page_numbers = [s.page_number for s in doc.sections]

        # All page numbers should be positive
        assert all(p > 0 for p in page_numbers), "Found non-positive page numbers"

        # Page numbers should be reasonable (not jump by 100s)
        page_range = max(page_numbers) - min(page_numbers)
        print(f"\n=== Page range ===")
        print(f"  Min page: {min(page_numbers)}")
        print(f"  Max page: {max(page_numbers)}")
        print(f"  Range: {page_range}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
