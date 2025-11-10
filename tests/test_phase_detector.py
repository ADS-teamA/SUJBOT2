"""
Unit tests for PhaseDetector resume functionality.

Tests phase detection logic, JSON validation, and sequence validation.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from src.phase_detector import PhaseDetector, PhaseStatus


class TestPhaseDetector:
    """Test phase detection for resume functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_phase1_data(self):
        """Create sample phase 1 JSON data."""
        return {
            "document_id": "test_doc",
            "source_path": "/path/to/test.pdf",
            "sections": [
                {
                    "section_id": "test_doc_sec_1",
                    "title": "Section 1",
                    "level": 1,
                    "depth": 0,
                    "path": "Section 1",
                    "page_number": 1,
                    "content_length": 500,
                }
            ],
            "hierarchy_depth": 2,
            "num_roots": 1,
            "num_sections": 5,
            "num_tables": 0,
        }

    @pytest.fixture
    def sample_phase2_data(self):
        """Create sample phase 2 JSON data."""
        return {
            "document_id": "test_doc",
            "document_summary": "Test document summary",
            "section_summaries": [
                {"section_id": "test_doc_sec_1", "summary": "Section 1 summary"}
            ],
        }

    @pytest.fixture
    def sample_phase3_data(self):
        """Create sample phase 3 JSON data."""
        return {
            "document_id": "test_doc",
            "chunking_stats": {
                "layer1_count": 1,
                "layer2_count": 5,
                "layer3_count": 20,
            },
            "layer1": [],
            "layer2": [],
            "layer3": [],
        }

    def test_detect_no_output_dir(self, temp_dir):
        """Test detection when output directory doesn't exist."""
        non_existent = temp_dir / "non_existent"
        status = PhaseDetector.detect(non_existent)

        assert status.completed_phase == 0
        assert status.output_dir == non_existent
        assert len(status.phase_files) == 0
        assert status.is_valid is True
        assert status.error is None

    def test_detect_empty_output_dir(self, temp_dir):
        """Test detection when output directory is empty."""
        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 0
        assert status.output_dir == temp_dir
        assert len(status.phase_files) == 0
        assert status.is_valid is True

    def test_detect_phase1_only(self, temp_dir, sample_phase1_data):
        """Test detection with only phase 1 completed."""
        phase1_path = temp_dir / "phase1_extraction.json"
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 1
        assert 1 in status.phase_files
        assert status.phase_files[1] == phase1_path
        assert status.is_valid is True

    def test_detect_phase1_and_2(
        self, temp_dir, sample_phase1_data, sample_phase2_data
    ):
        """Test detection with phases 1+2 completed."""
        phase1_path = temp_dir / "phase1_extraction.json"
        phase2_path = temp_dir / "phase2_summaries.json"

        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase2_data, f)

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 2
        assert 1 in status.phase_files
        assert 2 in status.phase_files
        assert status.is_valid is True

    def test_detect_all_phases(
        self, temp_dir, sample_phase1_data, sample_phase2_data, sample_phase3_data
    ):
        """Test detection with phases 1+2+3 completed."""
        phase1_path = temp_dir / "phase1_extraction.json"
        phase2_path = temp_dir / "phase2_summaries.json"
        phase3_path = temp_dir / "phase3_chunks.json"

        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase2_data, f)
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase3_data, f)

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 3
        assert 1 in status.phase_files
        assert 2 in status.phase_files
        assert 3 in status.phase_files
        assert status.is_valid is True

    def test_detect_phase4_vector_store(
        self, temp_dir, sample_phase1_data, sample_phase2_data, sample_phase3_data
    ):
        """Test detection with phase 4 vector store directory."""
        # Create phases 1-3
        phase1_path = temp_dir / "phase1_extraction.json"
        phase2_path = temp_dir / "phase2_summaries.json"
        phase3_path = temp_dir / "phase3_chunks.json"

        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase2_data, f)
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase3_data, f)

        # Create phase 4 directory
        phase4_dir = temp_dir / "phase4_vector_store"
        phase4_dir.mkdir()

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 4
        assert 4 in status.phase_files
        assert status.phase_files[4] == phase4_dir
        assert status.is_valid is True

    def test_detect_knowledge_graph(
        self, temp_dir, sample_phase1_data, sample_phase2_data, sample_phase3_data
    ):
        """Test detection with knowledge graph (phase 5)."""
        # Create phases 1-3
        phase1_path = temp_dir / "phase1_extraction.json"
        phase2_path = temp_dir / "phase2_summaries.json"
        phase3_path = temp_dir / "phase3_chunks.json"

        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase2_data, f)
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase3_data, f)

        # Create phase 4 directory
        phase4_dir = temp_dir / "phase4_vector_store"
        phase4_dir.mkdir()

        # Create KG file
        kg_path = temp_dir / "test_doc_kg.json"
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump({"entities": [], "relationships": []}, f)

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 5
        assert 5 in status.phase_files
        assert status.phase_files[5] == kg_path
        assert status.is_valid is True

    def test_detect_corrupted_json(self, temp_dir):
        """Test detection with corrupted JSON file."""
        phase1_path = temp_dir / "phase1_extraction.json"

        # Write invalid JSON
        with open(phase1_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json ")

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 0  # Stops before corrupted phase
        assert status.is_valid is False
        assert "corrupted" in status.error.lower()

    def test_detect_missing_required_fields(self, temp_dir):
        """Test detection with missing required fields in JSON."""
        phase1_path = temp_dir / "phase1_extraction.json"

        # Write JSON missing required field
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump({"document_id": "test"}, f)  # Missing 'sections'

        status = PhaseDetector.detect(temp_dir)

        assert status.completed_phase == 0
        assert status.is_valid is False

    def test_detect_gap_in_sequence(
        self, temp_dir, sample_phase1_data, sample_phase3_data
    ):
        """Test detection with gap in phase sequence (phase 2 missing)."""
        phase1_path = temp_dir / "phase1_extraction.json"
        phase3_path = temp_dir / "phase3_chunks.json"

        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)
        # Skip phase 2
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase3_data, f)

        status = PhaseDetector.detect(temp_dir)

        # Should stop at highest complete sequence (phase 1)
        assert status.completed_phase == 1
        assert status.is_valid is False
        assert "incomplete" in status.error.lower()

    def test_validate_json_file_success(self, temp_dir, sample_phase1_data):
        """Test JSON validation with valid file."""
        phase1_path = temp_dir / "phase1_extraction.json"
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(sample_phase1_data, f)

        is_valid = PhaseDetector._validate_json_file(phase1_path, 1)
        assert is_valid is True

    def test_validate_json_file_missing_fields(self, temp_dir):
        """Test JSON validation with missing required fields."""
        phase1_path = temp_dir / "phase1_extraction.json"
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump({"document_id": "test"}, f)

        is_valid = PhaseDetector._validate_json_file(phase1_path, 1)
        assert is_valid is False

    def test_validate_sequence_complete(self):
        """Test sequence validation with complete sequence."""
        phase_files = {1: Path("p1.json"), 2: Path("p2.json"), 3: Path("p3.json")}
        is_valid = PhaseDetector._validate_sequence(phase_files, 3)
        assert is_valid is True

    def test_validate_sequence_gap(self):
        """Test sequence validation with gap."""
        phase_files = {1: Path("p1.json"), 3: Path("p3.json")}  # Missing phase 2
        is_valid = PhaseDetector._validate_sequence(phase_files, 3)
        assert is_valid is False

    def test_validate_sequence_empty(self):
        """Test sequence validation with empty sequence."""
        phase_files = {}
        is_valid = PhaseDetector._validate_sequence(phase_files, 0)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
