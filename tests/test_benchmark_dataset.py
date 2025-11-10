"""
Tests for BenchmarkDataset module.

Critical tests for dataset loading, validation, and error handling.
"""

import json
import pytest
from pathlib import Path

from src.benchmark.dataset import BenchmarkDataset, QueryExample, GroundTruthSnippet


class TestDatasetLoading:
    """Test dataset loading from JSON files."""

    def test_from_json_valid_dataset(self, tmp_path):
        """Valid dataset should load successfully."""
        dataset_json = tmp_path / "valid_dataset.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "What is privacy?",
                            "snippets": [
                                {
                                    "file_path": "privacy.txt",
                                    "span": [0, 50],
                                    "answer": "Privacy is the right to control personal information",
                                }
                            ],
                        }
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))

        assert len(dataset.queries) == 1
        assert dataset.queries[0].query_id == 1
        assert dataset.queries[0].query == "What is privacy?"

    def test_from_json_file_not_found(self):
        """Missing JSON file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            BenchmarkDataset.from_json("nonexistent_dataset.json")

    def test_from_json_invalid_json(self, tmp_path):
        """Malformed JSON should raise ValueError."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json")

        with pytest.raises(ValueError, match="Invalid JSON file"):
            BenchmarkDataset.from_json(str(json_file))

    def test_from_json_missing_tests_key(self, tmp_path):
        """JSON without 'tests' key should raise ValueError."""
        json_file = tmp_path / "no_tests.json"
        json_file.write_text(json.dumps({"queries": []}))

        with pytest.raises(ValueError, match="Expected 'tests' key"):
            BenchmarkDataset.from_json(str(json_file))

    def test_from_json_all_invalid_queries(self, tmp_path):
        """Dataset with all invalid queries should raise ValueError."""
        json_file = tmp_path / "all_invalid.json"
        json_file.write_text(
            json.dumps(
                {
                    "tests": [
                        {"query": "", "snippets": []},  # Empty query
                        {"query": "test", "snippets": []},  # No snippets
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="No valid queries found"):
            BenchmarkDataset.from_json(str(json_file))

    def test_from_json_skips_invalid_queries(self, tmp_path, caplog):
        """Dataset should skip invalid queries with warning."""
        json_file = tmp_path / "mixed_validity.json"
        json_file.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "",  # Invalid: empty
                            "snippets": [
                                {"file_path": "test.txt", "span": [0, 10], "answer": "answer"}
                            ],
                        },
                        {
                            "query": "Valid query",  # Valid
                            "snippets": [
                                {"file_path": "test.txt", "span": [0, 10], "answer": "answer"}
                            ],
                        },
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(json_file))

        assert len(dataset.queries) == 1  # Only valid query loaded
        assert "Skipping invalid query" in caplog.text


class TestGroundTruthSnippet:
    """Test GroundTruthSnippet validation."""

    def test_snippet_valid(self):
        """Valid snippet should be created successfully."""
        snippet = GroundTruthSnippet(
            file_path="test.txt", span_start=0, span_end=50, answer="test answer"
        )

        assert snippet.file_path == "test.txt"
        assert snippet.span_start == 0
        assert snippet.span_end == 50
        assert snippet.answer == "test answer"

    def test_snippet_invalid_span_end_before_start(self):
        """Span with end < start should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid span"):
            GroundTruthSnippet(
                file_path="test.txt",
                span_start=100,
                span_end=50,  # Invalid: end < start
                answer="test",
            )

    def test_snippet_invalid_span_equal(self):
        """Span with end == start should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid span"):
            GroundTruthSnippet(
                file_path="test.txt",
                span_start=50,
                span_end=50,  # Invalid: end == start
                answer="test",
            )

    def test_snippet_empty_answer(self):
        """Empty answer should raise ValueError."""
        with pytest.raises(ValueError, match="Answer text cannot be empty"):
            GroundTruthSnippet(
                file_path="test.txt", span_start=0, span_end=50, answer=""  # Invalid: empty
            )


class TestDatasetStatistics:
    """Test dataset statistics and helper methods."""

    def test_get_statistics_single_query(self, tmp_path):
        """Statistics should be correct for single query."""
        dataset_json = tmp_path / "single_query.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "Test query",
                            "snippets": [
                                {"file_path": "doc1.txt", "span": [0, 50], "answer": "answer 1"},
                                {"file_path": "doc2.txt", "span": [100, 150], "answer": "answer 2"},
                            ],
                        }
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))
        stats = dataset.get_statistics()

        assert stats["total_queries"] == 1
        assert stats["total_snippets"] == 2
        assert stats["avg_snippets_per_query"] == 2.0
        assert stats["unique_source_documents"] == 2

    def test_get_expected_answers(self, tmp_path):
        """Expected answers should be extracted correctly."""
        dataset_json = tmp_path / "test.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "Test",
                            "snippets": [
                                {"file_path": "doc.txt", "span": [0, 10], "answer": "answer 1"},
                                {"file_path": "doc.txt", "span": [20, 30], "answer": "answer 2"},
                            ],
                        }
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))
        query = dataset.queries[0]
        answers = query.get_expected_answers()

        assert answers == ["answer 1", "answer 2"]

    def test_get_source_documents_unique(self, tmp_path):
        """Source documents should be deduplicated."""
        dataset_json = tmp_path / "test.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "Test",
                            "snippets": [
                                {"file_path": "doc1.txt", "span": [0, 10], "answer": "answer 1"},
                                {"file_path": "doc1.txt", "span": [20, 30], "answer": "answer 2"},
                                {"file_path": "doc2.txt", "span": [0, 10], "answer": "answer 3"},
                            ],
                        }
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))
        query = dataset.queries[0]
        docs = query.get_source_documents()

        assert len(docs) == 2  # Deduplicated
        assert "doc1.txt" in docs
        assert "doc2.txt" in docs

    def test_get_query_by_id_valid(self, tmp_path):
        """Get query by valid ID should return the query."""
        dataset_json = tmp_path / "test.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "Query 1",
                            "snippets": [{"file_path": "doc.txt", "span": [0, 10], "answer": "a"}],
                        },
                        {
                            "query": "Query 2",
                            "snippets": [{"file_path": "doc.txt", "span": [0, 10], "answer": "b"}],
                        },
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))
        query = dataset.get_query_by_id(2)

        assert query.query_id == 2
        assert query.query == "Query 2"

    def test_get_query_by_id_invalid(self, tmp_path):
        """Get query by invalid ID should raise ValueError."""
        dataset_json = tmp_path / "test.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "query": "Query 1",
                            "snippets": [{"file_path": "doc.txt", "span": [0, 10], "answer": "a"}],
                        }
                    ]
                }
            )
        )

        dataset = BenchmarkDataset.from_json(str(dataset_json))

        with pytest.raises(ValueError, match="Query ID 999 not found"):
            dataset.get_query_by_id(999)
