"""
Unit tests for PhaseLoaders resume functionality.

Tests loading of saved phase outputs back into Python objects.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from src.phase_loaders import PhaseLoaders
from src.docling_extractor_v2 import ExtractedDocument, DocumentSection
from src.multi_layer_chunker import Chunk, ChunkMetadata


class TestPhaseLoaders:
    """Test phase loading for resume functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_phase1_json(self, temp_dir):
        """Create sample phase 1 JSON file."""
        data = {
            "document_id": "test_doc",
            "source_path": "/path/to/test.pdf",
            "sections": [
                {
                    "section_id": "test_doc_sec_1",
                    "title": "Introduction",
                    "level": 1,
                    "depth": 0,
                    "path": "Introduction",
                    "page_number": 1,
                    "content_length": 500,
                },
                {
                    "section_id": "test_doc_sec_2",
                    "title": "Methods",
                    "level": 1,
                    "depth": 0,
                    "path": "Methods",
                    "page_number": 2,
                    "content_length": 750,
                },
            ],
            "hierarchy_depth": 2,
            "num_roots": 2,
            "num_sections": 2,
            "num_tables": 0,
        }

        json_path = temp_dir / "phase1_extraction.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return json_path

    @pytest.fixture
    def sample_phase2_json(self, temp_dir):
        """Create sample phase 2 JSON file."""
        data = {
            "document_id": "test_doc",
            "document_summary": "This document discusses research methods and results.",
            "section_summaries": [
                {
                    "section_id": "test_doc_sec_1",
                    "summary": "Introduction to the research topic.",
                },
                {
                    "section_id": "test_doc_sec_2",
                    "summary": "Description of research methods used.",
                },
            ],
        }

        json_path = temp_dir / "phase2_summaries.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return json_path

    @pytest.fixture
    def sample_phase3_json(self, temp_dir):
        """Create sample phase 3 JSON file."""
        data = {
            "document_id": "test_doc",
            "chunking_stats": {
                "layer1_count": 1,
                "layer2_count": 2,
                "layer3_count": 4,
            },
            "layer1": [
                {
                    "chunk_id": "test_doc_L1_doc_chunk_0",
                    "content": "Document summary: Test content",
                    "raw_content": "Test content",
                    "metadata": {
                        "chunk_id": "test_doc_L1_doc_chunk_0",
                        "layer": 1,
                        "document_id": "test_doc",
                        "title": "test_doc",
                        "page_number": 1,
                        "char_start": 0,
                        "char_end": 100,
                    },
                }
            ],
            "layer2": [
                {
                    "chunk_id": "test_doc_L2_sec_1_chunk_0",
                    "content": "Section summary: Introduction content",
                    "raw_content": "Introduction content",
                    "metadata": {
                        "chunk_id": "test_doc_L2_sec_1_chunk_0",
                        "layer": 2,
                        "document_id": "test_doc",
                        "section_id": "test_doc_sec_1",
                        "title": "Introduction",
                        "page_number": 1,
                        "char_start": 0,
                        "char_end": 50,
                        "section_title": "Introduction",
                        "section_path": "Introduction",
                        "section_level": 1,
                        "section_depth": 0,
                    },
                },
                {
                    "chunk_id": "test_doc_L2_sec_2_chunk_0",
                    "content": "Section summary: Methods content",
                    "raw_content": "Methods content",
                    "metadata": {
                        "chunk_id": "test_doc_L2_sec_2_chunk_0",
                        "layer": 2,
                        "document_id": "test_doc",
                        "section_id": "test_doc_sec_2",
                        "title": "Methods",
                        "page_number": 2,
                        "char_start": 50,
                        "char_end": 100,
                        "section_title": "Methods",
                        "section_path": "Methods",
                        "section_level": 1,
                        "section_depth": 0,
                    },
                },
            ],
            "layer3": [
                {
                    "chunk_id": "test_doc_L3_sec_1_chunk_0",
                    "content": "Chunk with SAC context",
                    "raw_content": "Chunk without context",
                    "metadata": {
                        "chunk_id": "test_doc_L3_sec_1_chunk_0",
                        "layer": 3,
                        "document_id": "test_doc",
                        "section_id": "test_doc_sec_1",
                        "parent_chunk_id": "test_doc_L2_sec_1_chunk_0",
                        "page_number": 1,
                        "char_start": 0,
                        "char_end": 25,
                        "section_title": "Introduction",
                        "section_path": "Introduction",
                        "section_level": 1,
                        "section_depth": 0,
                    },
                },
                {
                    "chunk_id": "test_doc_L3_sec_1_chunk_1",
                    "content": "Another chunk with SAC",
                    "raw_content": "Another chunk",
                    "metadata": {
                        "chunk_id": "test_doc_L3_sec_1_chunk_1",
                        "layer": 3,
                        "document_id": "test_doc",
                        "section_id": "test_doc_sec_1",
                        "parent_chunk_id": "test_doc_L2_sec_1_chunk_0",
                        "page_number": 1,
                        "char_start": 25,
                        "char_end": 50,
                        "section_title": "Introduction",
                        "section_path": "Introduction",
                        "section_level": 1,
                        "section_depth": 0,
                    },
                },
            ],
        }

        json_path = temp_dir / "phase3_chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return json_path

    def test_load_phase1_success(self, sample_phase1_json):
        """Test successful loading of phase 1 extraction."""
        result = PhaseLoaders.load_phase1(sample_phase1_json)

        # Verify it's an ExtractedDocument
        assert isinstance(result, ExtractedDocument)

        # Verify basic fields
        assert result.document_id == "test_doc"
        assert result.source_path == "/path/to/test.pdf"
        assert result.hierarchy_depth == 2
        assert result.num_roots == 2
        assert result.num_sections == 2

        # Verify sections
        assert len(result.sections) == 2
        assert isinstance(result.sections[0], DocumentSection)
        assert result.sections[0].section_id == "test_doc_sec_1"
        assert result.sections[0].title == "Introduction"
        assert result.sections[1].section_id == "test_doc_sec_2"
        assert result.sections[1].title == "Methods"

        # Verify partial object (no summaries yet)
        assert result.document_summary is None
        assert result.sections[0].summary is None

    def test_load_phase2_success(self, sample_phase1_json, sample_phase2_json):
        """Test successful loading of phase 2 summaries."""
        # Load phase 1 first
        extraction_result = PhaseLoaders.load_phase1(sample_phase1_json)

        # Load phase 2 summaries
        result = PhaseLoaders.load_phase2(sample_phase2_json, extraction_result)

        # Verify document summary added
        assert result.document_summary == "This document discusses research methods and results."

        # Verify section summaries added
        assert result.sections[0].summary == "Introduction to the research topic."
        assert result.sections[1].summary == "Description of research methods used."

    def test_load_phase3_success(self, sample_phase3_json):
        """Test successful loading of phase 3 chunks."""
        chunks = PhaseLoaders.load_phase3(sample_phase3_json)

        # Verify structure
        assert isinstance(chunks, dict)
        assert "layer1" in chunks
        assert "layer2" in chunks
        assert "layer3" in chunks

        # Verify layer 1
        assert len(chunks["layer1"]) == 1
        l1_chunk = chunks["layer1"][0]
        assert isinstance(l1_chunk, Chunk)
        assert l1_chunk.chunk_id == "test_doc_L1_doc_chunk_0"
        assert l1_chunk.raw_content == "Test content"
        assert isinstance(l1_chunk.metadata, ChunkMetadata)
        assert l1_chunk.metadata.layer == 1

        # Verify layer 2
        assert len(chunks["layer2"]) == 2
        l2_chunk = chunks["layer2"][0]
        assert isinstance(l2_chunk, Chunk)
        assert l2_chunk.chunk_id == "test_doc_L2_sec_1_chunk_0"
        assert l2_chunk.metadata.layer == 2
        assert l2_chunk.metadata.section_title == "Introduction"

        # Verify layer 3
        assert len(chunks["layer3"]) == 2
        l3_chunk = chunks["layer3"][0]
        assert isinstance(l3_chunk, Chunk)
        assert l3_chunk.chunk_id == "test_doc_L3_sec_1_chunk_0"
        assert l3_chunk.metadata.layer == 3
        assert l3_chunk.metadata.parent_chunk_id == "test_doc_L2_sec_1_chunk_0"

    def test_load_phase1_missing_file(self, temp_dir):
        """Test loading phase 1 with missing file."""
        missing_path = temp_dir / "non_existent.json"

        with pytest.raises(FileNotFoundError):
            PhaseLoaders.load_phase1(missing_path)

    def test_load_phase1_invalid_json(self, temp_dir):
        """Test loading phase 1 with invalid JSON."""
        invalid_json = temp_dir / "invalid.json"
        with open(invalid_json, "w", encoding="utf-8") as f:
            f.write("{ invalid json ")

        with pytest.raises(ValueError, match="Phase 1 JSON corrupted"):
            PhaseLoaders.load_phase1(invalid_json)

    def test_load_phase2_missing_sections(self, sample_phase1_json, temp_dir):
        """Test loading phase 2 when section summaries don't match."""
        # Create phase 2 with different section IDs
        phase2_data = {
            "document_id": "test_doc",
            "document_summary": "Test summary",
            "section_summaries": [
                {"section_id": "non_existent_section", "summary": "Summary"},
            ],
        }

        phase2_path = temp_dir / "phase2_summaries.json"
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(phase2_data, f)

        # Load phase 1
        extraction_result = PhaseLoaders.load_phase1(sample_phase1_json)

        # Load phase 2 (should not crash, just skip missing sections)
        result = PhaseLoaders.load_phase2(phase2_path, extraction_result)

        # Document summary should be added
        assert result.document_summary == "Test summary"

        # Section summaries should be None (no match)
        assert result.sections[0].summary is None
        assert result.sections[1].summary is None

    def test_load_phase3_empty_layers(self, temp_dir):
        """Test loading phase 3 with empty layers."""
        phase3_data = {
            "document_id": "test_doc",
            "chunking_stats": {
                "layer1_count": 0,
                "layer2_count": 0,
                "layer3_count": 0,
            },
            "layer1": [],
            "layer2": [],
            "layer3": [],
        }

        phase3_path = temp_dir / "phase3_chunks.json"
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(phase3_data, f)

        chunks = PhaseLoaders.load_phase3(phase3_path)

        # Should load but with empty lists
        assert len(chunks["layer1"]) == 0
        assert len(chunks["layer2"]) == 0
        assert len(chunks["layer3"]) == 0

    def test_phase_loaders_end_to_end(
        self, sample_phase1_json, sample_phase2_json, sample_phase3_json
    ):
        """Test complete end-to-end phase loading workflow."""
        # Load phase 1
        extraction = PhaseLoaders.load_phase1(sample_phase1_json)
        assert extraction.document_id == "test_doc"
        assert len(extraction.sections) == 2

        # Load phase 2 (merges into extraction)
        extraction = PhaseLoaders.load_phase2(sample_phase2_json, extraction)
        assert extraction.document_summary is not None
        assert extraction.sections[0].summary is not None

        # Load phase 3 (independent)
        chunks = PhaseLoaders.load_phase3(sample_phase3_json)
        assert len(chunks["layer3"]) == 2

        # Verify all data is accessible
        assert extraction.sections[0].title == "Introduction"
        assert extraction.sections[0].summary == "Introduction to the research topic."
        assert chunks["layer3"][0].metadata.section_title == "Introduction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
