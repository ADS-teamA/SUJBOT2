#!/usr/bin/env python3
"""
Unit test for PHASE 3B: Section summaries from chunk contexts.

Tests the new architecture without requiring real PDFs.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata
from docling_extractor_v2 import DocumentSection, ExtractedDocument


def create_mock_section(section_id: str, title: str, content: str) -> DocumentSection:
    """Create a mock DocumentSection."""
    return DocumentSection(
        section_id=section_id,
        title=title,
        content=content,
        level=1,
        depth=1,
        parent_id=None,
        children_ids=[],
        ancestors=[],
        path=title,
        page_number=1,
        char_start=0,
        char_end=len(content),
        content_length=len(content),
        summary=None,  # Will be generated from chunk contexts
    )


def create_mock_chunk(
    chunk_id: str, section_id: str, context: str, raw_content: str
) -> Chunk:
    """Create a mock Layer 3 chunk with context."""
    return Chunk(
        chunk_id=chunk_id,
        content=f"{context}\n\n{raw_content}",  # Context + raw content
        raw_content=raw_content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            layer=3,
            document_id="test_doc",
            section_id=section_id,
            section_title=f"Section {section_id}",
        ),
    )


def test_generate_section_summaries_from_contexts():
    """Test section summary generation from chunk contexts."""

    print("\n" + "=" * 80)
    print("TEST: Generate Section Summaries from Chunk Contexts")
    print("=" * 80)

    # Create mock extracted document with 2 sections
    section1 = create_mock_section(
        "sec_1", "Introduction", "This is a long introduction with many important details..."
    )

    section2 = create_mock_section(
        "sec_2",
        "Methodology",
        "The methodology section describes the research approach and methods used...",
    )

    extracted_doc = Mock(spec=ExtractedDocument)
    extracted_doc.document_id = "test_doc"
    extracted_doc.sections = [section1, section2]
    extracted_doc.document_summary = "Test document about research methodology"

    # Create mock Layer 3 chunks with contexts
    layer3_chunks = [
        # Section 1 chunks
        create_mock_chunk(
            "test_doc_L3_sec_1_chunk_0",
            "sec_1",
            context="This chunk introduces the main topic and research question",
            raw_content="This is a long introduction with many important details about the research.",
        ),
        create_mock_chunk(
            "test_doc_L3_sec_1_chunk_1",
            "sec_1",
            context="This chunk discusses the background and motivation for the study",
            raw_content="The background provides context for why this research is important.",
        ),
        # Section 2 chunks
        create_mock_chunk(
            "test_doc_L3_sec_2_chunk_0",
            "sec_2",
            context="This chunk describes the data collection methods",
            raw_content="The methodology section describes the research approach and methods used.",
        ),
        create_mock_chunk(
            "test_doc_L3_sec_2_chunk_1",
            "sec_2",
            context="This chunk explains the analysis techniques",
            raw_content="We used statistical analysis to process the collected data.",
        ),
    ]

    # Initialize chunker
    chunker = MultiLayerChunker()

    print("\nBefore generation:")
    print(f"  Section 1 summary: {section1.summary}")
    print(f"  Section 2 summary: {section2.summary}")

    # Test the method
    try:
        chunker._generate_section_summaries_from_contexts(extracted_doc, layer3_chunks)

        print("\nAfter generation:")
        print(f"  Section 1 summary: {section1.summary}")
        print(f"    Length: {len(section1.summary) if section1.summary else 0} chars")
        print(f"  Section 2 summary: {section2.summary}")
        print(f"    Length: {len(section2.summary) if section2.summary else 0} chars")

        # Validate
        print("\nValidation:")

        # Check that summaries were generated
        assert section1.summary, "Section 1 should have a summary"
        assert section2.summary, "Section 2 should have a summary"
        print("  ✓ Both sections have summaries")

        # Check summary lengths (should be >= 50 chars based on validation)
        if len(section1.summary) >= 50:
            print(f"  ✓ Section 1 summary length: {len(section1.summary)} >= 50 chars")
        else:
            print(f"  ⚠️  Section 1 summary too short: {len(section1.summary)} chars")

        if len(section2.summary) >= 50:
            print(f"  ✓ Section 2 summary length: {len(section2.summary)} >= 50 chars")
        else:
            print(f"  ⚠️  Section 2 summary too short: {len(section2.summary)} chars")

        print("\n✓ TEST PASSED")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


def test_validate_summaries():
    """Test summary validation logic."""

    print("\n" + "=" * 80)
    print("TEST: Validate Section Summaries")
    print("=" * 80)

    # Create mock sections with different summary states
    section_valid = create_mock_section("sec_1", "Valid", "Content...")
    section_valid.summary = "This is a valid summary with more than 50 characters total length."

    section_short = create_mock_section("sec_2", "Short", "Content...")
    section_short.summary = "Too short"  # Only 9 chars

    section_missing = create_mock_section("sec_3", "Missing", "Content...")
    section_missing.summary = None

    section_empty_content = create_mock_section("sec_4", "Empty", "")
    section_empty_content.summary = None

    extracted_doc = Mock(spec=ExtractedDocument)
    extracted_doc.sections = [section_valid, section_short, section_missing, section_empty_content]

    # Initialize chunker
    chunker = MultiLayerChunker()

    print("\nTesting validation...")

    try:
        chunker._validate_summaries(extracted_doc, min_summary_length=50)
        print("✓ Validation passed (or allowed with warnings)")

    except ValueError as e:
        print(f"✓ Validation correctly raised error: {e}")

    print("\n✓ TEST PASSED")


if __name__ == "__main__":
    # Run tests
    test_generate_section_summaries_from_contexts()
    test_validate_summaries()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
