"""
Test breadcrumb path functionality in agent tools.

Verifies that hierarchical document paths are correctly displayed in tool outputs.
"""

import pytest
from src.agent.tools.utils import format_chunk_result, generate_citation


def test_format_chunk_result_with_hierarchical_path():
    """Test that format_chunk_result includes breadcrumb from hierarchical_path."""
    chunk = {
        "content": "Chlazení reaktoru VR-1 je zajištěno lehkou vodou.",
        "document_id": "BZ_VR1",
        "section_title": "Chlazení aktivní zóny",
        "section_path": "1.2.3 Chlazení aktivní zóny",
        "hierarchical_path": "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny",
        "chunk_id": "BZ_VR1_L3_1322",
        "score": 0.7948,
        "page_number": 105,
    }

    result = format_chunk_result(chunk, include_score=True, max_content_length=100)

    assert "breadcrumb" in result
    assert result["breadcrumb"] == "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny"
    assert result["document_id"] == "BZ_VR1"
    assert result["section_title"] == "Chlazení aktivní zóny"
    assert result["chunk_id"] == "BZ_VR1_L3_1322"
    assert result["score"] == 0.7948
    assert result["page"] == 105


def test_format_chunk_result_extracts_section_title_from_breadcrumb():
    """Test that section_title is extracted from breadcrumb when empty."""
    chunk = {
        "content": "Test content.",
        "document_id": "BZ_VR1",
        "section_title": "",  # Empty!
        "hierarchical_path": "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení",
        "chunk_id": "BZ_VR1_L3_100",
    }

    result = format_chunk_result(chunk, include_score=False)

    assert "breadcrumb" in result
    assert result["breadcrumb"] == "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení"
    assert result["section_title"] == "1.2.3 Chlazení"  # Extracted from breadcrumb!


def test_format_chunk_result_fallback_section_path():
    """Test breadcrumb fallback to section_path when hierarchical_path missing."""
    chunk = {
        "content": "Testovací obsah.",
        "document_id": "BZ_VR1",
        "section_title": "Test sekce",
        "section_path": "1.2.3 Test sekce",
        "chunk_id": "BZ_VR1_L3_100",
    }

    result = format_chunk_result(chunk, include_score=False)

    assert "breadcrumb" in result
    assert result["breadcrumb"] == "BZ_VR1 > 1.2.3 Test sekce"


def test_format_chunk_result_fallback_section_title():
    """Test breadcrumb fallback to section_title when section_path missing."""
    chunk = {
        "content": "Testovací obsah.",
        "document_id": "BZ_VR1",
        "section_title": "Test sekce",
        "chunk_id": "BZ_VR1_L3_100",
    }

    result = format_chunk_result(chunk, include_score=False)

    assert "breadcrumb" in result
    assert result["breadcrumb"] == "BZ_VR1 > Test sekce"


def test_format_chunk_result_no_section():
    """Test chunk without section information (document level)."""
    chunk = {
        "content": "Dokumentový obsah.",
        "document_id": "BZ_VR1",
        "chunk_id": "BZ_VR1_L1",
    }

    result = format_chunk_result(chunk, include_score=False)

    # Should not have breadcrumb if no hierarchical info available
    assert "breadcrumb" not in result


def test_generate_citation_with_hierarchical_path():
    """Test that citation includes full hierarchical breadcrumb."""
    chunk = {
        "hierarchical_path": "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny",
        "document_id": "BZ_VR1",
        "page_number": 105,
    }

    citation = generate_citation(chunk, chunk_number=1, format="inline")
    assert citation == "[1] BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny"

    citation_detailed = generate_citation(chunk, chunk_number=1, format="detailed")
    assert "[1] BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny" in citation_detailed
    assert "(Page 105)" in citation_detailed

    citation_footnote = generate_citation(chunk, chunk_number=1, format="footnote")
    assert "[1] BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Chlazení aktivní zóny" in citation_footnote
    assert "p. 105" in citation_footnote


def test_generate_citation_fallback():
    """Test citation fallback when hierarchical_path missing."""
    chunk = {
        "document_id": "BZ_VR1",
        "section_path": "1.2.3 Chlazení",
        "page_number": 105,
    }

    citation = generate_citation(chunk, chunk_number=2, format="inline")
    assert citation == "[2] BZ_VR1 > 1.2.3 Chlazení"


def test_generate_citation_multiple_chunks():
    """Test citations for multiple chunks to verify numbering."""
    chunks = [
        {
            "hierarchical_path": "BZ_VR1 > 1.2 Specifikace",
            "page_number": 10,
        },
        {
            "hierarchical_path": "Sb_1997_18 > 2.1 Bezpečnost",
            "page_number": 25,
        },
        {
            "hierarchical_path": "BZ_VR1 > 1.3 Provoz",
            "page_number": 15,
        },
    ]

    citations = [generate_citation(chunk, i + 1, format="inline") for i, chunk in enumerate(chunks)]

    assert citations[0] == "[1] BZ_VR1 > 1.2 Specifikace"
    assert citations[1] == "[2] Sb_1997_18 > 2.1 Bezpečnost"
    assert citations[2] == "[3] BZ_VR1 > 1.3 Provoz"


def test_breadcrumb_with_czech_characters():
    """Test breadcrumb with Czech diacritics."""
    chunk = {
        "content": "Testovací obsah s diakritikou.",
        "document_id": "BZ_VR1",
        "hierarchical_path": "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Řízení reaktivity",
        "chunk_id": "BZ_VR1_L3_200",
        "page_number": 50,
    }

    result = format_chunk_result(chunk, include_score=False)

    assert result["breadcrumb"] == "BZ_VR1 > 1.2 Technické specifikace > 1.2.3 Řízení reaktivity"

    citation = generate_citation(chunk, chunk_number=1, format="detailed")
    assert "1.2.3 Řízení reaktivity" in citation
    assert "(Page 50)" in citation


def test_format_chunk_result_replaces_untitled_with_page():
    """Test that 'Untitled' is replaced with page number in breadcrumb."""
    chunk = {
        "content": "Content from untitled section.",
        "document_id": "BZ_VR1",
        "section_title": "",
        "hierarchical_path": "BZ_VR1 > Untitled",
        "chunk_id": "BZ_VR1_L3_300",
        "page_number": 18,
    }

    result = format_chunk_result(chunk, include_score=False)

    # Should replace " > Untitled" with " (page 18)"
    assert result["breadcrumb"] == "BZ_VR1 (page 18)"
    assert result["section_title"] == "Page 18"

    # Test citation
    citation = generate_citation(chunk, chunk_number=1, format="inline")
    assert "BZ_VR1 (page 18)" in citation or "Page 18" in citation


def test_format_chunk_result_untitled_without_page():
    """Test Untitled handling when page number is missing."""
    chunk = {
        "content": "Content.",
        "document_id": "BZ_VR1",
        "section_title": "",
        "hierarchical_path": "BZ_VR1 > Untitled",
        "chunk_id": "BZ_VR1_L3_400",
    }

    result = format_chunk_result(chunk, include_score=False)

    # Should remove " > Untitled" entirely
    assert result["breadcrumb"] == "BZ_VR1"
    # section_title should remain empty when no page available
    assert result["section_title"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
