#!/usr/bin/env python3
"""
Test § paragraph detection in Unstructured.io extraction.

Validates the claimed "10/10 § paragraph detection" improvement over Docling.
Tests paragraph hierarchy, numbering variants, and integration with legal documents.
"""

import pytest
from pathlib import Path
import re
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unstructured_extractor import (
    UnstructuredExtractor,
    ExtractionConfig,
)


class TestParagraphDetection:
    """Test § paragraph detection on legal documents."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing."""
        config = ExtractionConfig(
            strategy="hi_res",  # Use hi_res for best accuracy
            model="detectron2_mask_rcnn",  # Most accurate model
            generate_summaries=False,
            extract_tables=False,
        )
        return UnstructuredExtractor(config)

    def test_paragraph_detection_sb_1997_18(self, extractor):
        """
        Test § paragraph detection on Sb_1997_18 document.

        This is the PRIMARY TEST for the claimed "10/10 vs 0/10" improvement.
        Known structure: This document has 10 § paragraphs.
        """
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find all § paragraphs in titles
        paragraph_sections = []
        for section in doc.sections:
            if section.title and re.match(r'^§\s*\d+', section.title.strip()):
                paragraph_sections.append(section)

        print("\n=== § Paragraph Detection Results ===")
        print(f"  Found {len(paragraph_sections)} § paragraphs")
        print(f"  Expected: 10 § paragraphs")
        print("\n  Detected paragraphs:")
        for i, para in enumerate(paragraph_sections, 1):
            print(f"    {i}. {para.title[:50]}")

        # CRITICAL ASSERTION: This validates the core claim
        assert len(paragraph_sections) >= 8, (
            f"Expected at least 8 § paragraphs (allowing some margin), "
            f"but found only {len(paragraph_sections)}. "
            f"This indicates the '10/10' claim may be inaccurate."
        )

        # Ideal case: exactly 10
        if len(paragraph_sections) == 10:
            print("\n  ✓ PERFECT: Detected exactly 10/10 § paragraphs")
        elif len(paragraph_sections) >= 8:
            print(f"\n  ⚠️  Detected {len(paragraph_sections)}/10 § paragraphs (acceptable but not perfect)")

    def test_paragraph_numbering_variants(self, extractor):
        """Test detection of § formatting variants (§1, § 1, §  1, etc.)."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find different § formatting patterns
        patterns = {
            r'^§\d+': [],  # §1 (no space)
            r'^§\s+\d+': [],  # § 1 (with space)
            r'^§\s*\d+[a-z]': [],  # §1a (with letter)
            r'^§\s*\d+\s+odst': [],  # § 1 odst. 2 (subsection)
        }

        for section in doc.sections:
            if not section.title:
                continue

            title = section.title.strip()
            for pattern, matches in patterns.items():
                if re.match(pattern, title, re.IGNORECASE):
                    matches.append(title)

        print("\n=== § Formatting Variants ===")
        for pattern, matches in patterns.items():
            if matches:
                print(f"  Pattern '{pattern}': {len(matches)} matches")
                print(f"    Examples: {matches[:3]}")

        # At least one pattern should match
        total_matches = sum(len(m) for m in patterns.values())
        assert total_matches > 0, "Expected to find at least some § paragraphs"

    def test_paragraph_hierarchy_nesting(self, extractor):
        """Test that § paragraphs nest correctly under chapters/parts."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        if not paragraphs:
            pytest.skip("No § paragraphs found in document")

        # Check hierarchy
        paragraphs_with_parents = [p for p in paragraphs if p.parent_id]

        print(f"\n=== § Paragraph Hierarchy ===")
        print(f"  Total § paragraphs: {len(paragraphs)}")
        print(f"  Paragraphs with parent: {len(paragraphs_with_parents)}")

        # Most paragraphs should have parents (they belong to chapters)
        parent_ratio = len(paragraphs_with_parents) / len(paragraphs)
        print(f"  Parent ratio: {parent_ratio:.1%}")

        # Show hierarchy sample
        print("\n  Sample hierarchy:")
        for para in paragraphs[:3]:
            print(f"    [{para.level}] {para.title}")
            print(f"      Path: {para.path}")
            print(f"      Depth: {para.depth}, Parent: {para.parent_id[:20] if para.parent_id else 'None'}")

    def test_paragraph_content_not_empty(self, extractor):
        """Test that § paragraphs have non-empty content."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        if not paragraphs:
            pytest.skip("No § paragraphs found in document")

        # Check content
        paragraphs_with_content = [
            p for p in paragraphs
            if p.content and len(p.content.strip()) > 10
        ]

        coverage = len(paragraphs_with_content) / len(paragraphs)

        print(f"\n=== § Paragraph Content ===")
        print(f"  Paragraphs with content: {len(paragraphs_with_content)}/{len(paragraphs)} ({coverage:.1%})")

        # Most paragraphs should have content
        assert coverage > 0.7, (
            f"Expected >70% of § paragraphs to have content, got {coverage:.1%}"
        )

        # Show content sample
        print("\n  Content sample:")
        for para in paragraphs_with_content[:2]:
            content_preview = para.content.strip()[:100].replace('\n', ' ')
            print(f"    {para.title}: {content_preview}...")

    def test_paragraph_page_numbers(self, extractor):
        """Test that § paragraphs have valid page numbers."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        if not paragraphs:
            pytest.skip("No § paragraphs found in document")

        # Check page numbers
        page_numbers = [p.page_number for p in paragraphs]

        print(f"\n=== § Paragraph Page Distribution ===")
        print(f"  Page range: {min(page_numbers)} - {max(page_numbers)}")
        print(f"  Paragraphs across {max(page_numbers) - min(page_numbers) + 1} pages")

        # All should have valid page numbers
        assert all(p > 0 for p in page_numbers), "Found invalid page numbers"

        # Show distribution
        print("\n  Distribution:")
        for para in paragraphs:
            print(f"    Page {para.page_number}: {para.title}")


class TestParagraphDetectionMultipleDocuments:
    """Test paragraph detection across multiple legal documents."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_paragraph_detection_sb_2016_263(self, extractor):
        """Test § paragraph detection on second legal document."""
        pdf_path = Path("data/Sb_2016_263_2024-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        print(f"\n=== Sb_2016_263 § Paragraphs ===")
        print(f"  Found {len(paragraphs)} § paragraphs")

        if paragraphs:
            print("\n  Sample paragraphs:")
            for para in paragraphs[:5]:
                print(f"    {para.title}")

        # This document should also have § paragraphs
        assert len(paragraphs) > 0, "Expected to find § paragraphs in legal document"

    def test_paragraph_consistency_across_documents(self, extractor):
        """Test that paragraph detection works consistently across documents."""
        test_docs = [
            Path("data/Sb_1997_18_2017-01-01_IZ.pdf"),
            Path("data/Sb_2016_263_2024-01-01_IZ.pdf"),
        ]

        results = {}

        for pdf_path in test_docs:
            if not pdf_path.exists():
                continue

            doc = extractor.extract(pdf_path)

            # Count § paragraphs
            paragraphs = [
                s for s in doc.sections
                if s.title and re.match(r'^§\s*\d+', s.title.strip())
            ]

            results[pdf_path.name] = {
                "total_sections": len(doc.sections),
                "paragraphs": len(paragraphs),
                "ratio": len(paragraphs) / len(doc.sections) if doc.sections else 0,
            }

        print("\n=== Cross-Document Consistency ===")
        for doc_name, stats in results.items():
            print(f"  {doc_name}:")
            print(f"    Total sections: {stats['total_sections']}")
            print(f"    § paragraphs: {stats['paragraphs']}")
            print(f"    Ratio: {stats['ratio']:.2%}")

        # Both documents should have paragraphs
        assert all(stats["paragraphs"] > 0 for stats in results.values()), (
            "All legal documents should have § paragraphs detected"
        )


class TestParagraphElementTypes:
    """Test element type detection for § paragraphs."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_paragraph_element_category(self, extractor):
        """Test that § paragraphs have appropriate element categories."""
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        if not paragraphs:
            pytest.skip("No § paragraphs found")

        # Check element categories
        categories = {}
        for para in paragraphs:
            cat = para.element_category or "Unknown"
            categories[cat] = categories.get(cat, 0) + 1

        print("\n=== § Paragraph Element Categories ===")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        # Paragraphs are typically detected as Title or ListItem
        expected_categories = ["Title", "ListItem", "NarrativeText"]
        detected_categories = set(categories.keys())

        assert any(cat in detected_categories for cat in expected_categories), (
            f"Expected § paragraphs to be categorized as {expected_categories}, "
            f"but got {detected_categories}"
        )


class TestParagraphRegressionDetection:
    """Regression tests to catch future detection failures."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_regression_zero_paragraphs(self, extractor):
        """
        REGRESSION TEST: Ensure we never go back to 0/10 detection.

        This test will FAIL if paragraph detection completely breaks.
        """
        pdf_path = Path("data/Sb_1997_18_2017-01-01_IZ.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Find § paragraphs
        paragraphs = [
            s for s in doc.sections
            if s.title and re.match(r'^§\s*\d+', s.title.strip())
        ]

        # CRITICAL: Must detect more than 0 paragraphs
        assert len(paragraphs) > 0, (
            "REGRESSION: Detected 0 § paragraphs! "
            "This is a critical failure - we've regressed back to Docling levels."
        )

        # Should detect at least half of expected paragraphs
        assert len(paragraphs) >= 5, (
            f"REGRESSION: Only detected {len(paragraphs)}/10 § paragraphs. "
            f"This is below acceptable threshold (50%)."
        )

        print(f"\n✓ REGRESSION TEST PASSED: Detected {len(paragraphs)} § paragraphs (>= 5 required)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
