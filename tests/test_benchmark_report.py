"""
Tests for Benchmark Report Generation.

Critical tests for report edge cases, formatting, and error handling.
"""

import json
import pytest
from pathlib import Path

from src.benchmark.report import (
    save_markdown_report,
    save_json_report,
    generate_reports,
    BenchmarkResult,
    QueryResult,
)


@pytest.fixture
def sample_query_result():
    """Create a sample query result."""
    return QueryResult(
        query_id=1,
        query="What is privacy?",
        predicted_answer="Privacy is control of personal information",
        ground_truth_answers=["Privacy is control"],
        metrics={
            "exact_match": 0.0,
            "f1_score": 0.75,
            "precision": 0.75,
            "recall": 1.0,
            "embedding_similarity": 0.85,
            "combined_f1": 0.75,
        },
        retrieval_time_ms=250.5,
        cost_usd=0.002,
        rag_confidence=0.85,
    )


@pytest.fixture
def sample_benchmark_result(sample_query_result):
    """Create a sample benchmark result."""
    return BenchmarkResult(
        dataset_name="TestDataset",
        total_queries=1,
        aggregate_metrics={
            "exact_match": 0.0,
            "f1_score": 0.75,
            "precision": 0.75,
            "recall": 1.0,
            "embedding_similarity": 0.85,
            "combined_f1": 0.75,
        },
        query_results=[sample_query_result],
        total_time_seconds=1.5,
        total_cost_usd=0.002,
        config={"k": 5, "enable_reranking": True},
    )


class TestReportGeneration:
    """Test report generation functions."""

    def test_markdown_report_basic(self, tmp_path, sample_benchmark_result):
        """Basic markdown report should generate successfully."""
        report_path = save_markdown_report(sample_benchmark_result, str(tmp_path))

        assert report_path.exists()
        content = report_path.read_text()

        # Check key sections
        assert "# Benchmark Report: TestDataset" in content
        assert "Total Queries" in content
        assert "F1 Score" in content
        assert "0.75" in content

    def test_json_report_basic(self, tmp_path, sample_benchmark_result):
        """Basic JSON report should generate successfully."""
        report_path = save_json_report(sample_benchmark_result, str(tmp_path))

        assert report_path.exists()
        data = json.loads(report_path.read_text())

        assert data["dataset_name"] == "TestDataset"
        assert data["total_queries"] == 1
        assert "aggregate_metrics" in data

    def test_markdown_report_zero_queries(self, tmp_path):
        """Report with zero queries should not crash (division by zero)."""
        result = BenchmarkResult(
            dataset_name="EmptyDataset",
            total_queries=0,
            aggregate_metrics={},
            query_results=[],
            total_time_seconds=0.0,
            total_cost_usd=0.0,
            config={},
        )

        report_path = save_markdown_report(result, str(tmp_path))

        assert report_path.exists()
        content = report_path.read_text()

        # Should show 0ms and $0.00, not crash
        assert "0" in content or "N/A" in content

    def test_markdown_report_missing_rag_confidence(self, tmp_path):
        """Report should handle missing RAG confidence gracefully."""
        query_result = QueryResult(
            query_id=1,
            query="Test",
            predicted_answer="test",
            ground_truth_answers=["test"],
            metrics={"f1_score": 1.0},
            retrieval_time_ms=100.0,
            cost_usd=0.001,
            rag_confidence=None,  # Missing confidence
        )

        result = BenchmarkResult(
            dataset_name="Test",
            total_queries=1,
            aggregate_metrics={"f1_score": 1.0},
            query_results=[query_result],
            total_time_seconds=1.0,
            total_cost_usd=0.001,
            config={},
        )

        report_path = save_markdown_report(result, str(tmp_path))
        content = report_path.read_text()

        # Should show N/A for missing confidence
        assert "N/A" in content or "None" in content

    def test_markdown_report_small_result_set(self, tmp_path):
        """Report with <10 results should not crash (IndexError)."""
        query_results = [
            QueryResult(
                query_id=i,
                query=f"Query {i}",
                predicted_answer="answer",
                ground_truth_answers=["answer"],
                metrics={"f1_score": 0.8},
                retrieval_time_ms=100.0,
                cost_usd=0.001,
                rag_confidence=0.8,
            )
            for i in range(1, 4)  # Only 3 queries
        ]

        result = BenchmarkResult(
            dataset_name="SmallDataset",
            total_queries=3,
            aggregate_metrics={"f1_score": 0.8},
            query_results=query_results,
            total_time_seconds=1.0,
            total_cost_usd=0.003,
            config={},
        )

        report_path = save_markdown_report(result, str(tmp_path))
        content = report_path.read_text()

        # Should adapt table size, not crash
        assert "Query 1" in content
        assert "Query 3" in content

    def test_json_report_serialization(self, tmp_path, sample_benchmark_result):
        """JSON report should be valid and complete."""
        report_path = save_json_report(sample_benchmark_result, str(tmp_path))

        data = json.loads(report_path.read_text())

        # Check all required fields
        assert "dataset_name" in data
        assert "total_queries" in data
        assert "aggregate_metrics" in data
        assert "query_results" in data
        assert "total_time_seconds" in data
        assert "total_cost_usd" in data
        assert "config" in data

        # Check query result structure
        assert len(data["query_results"]) == 1
        query = data["query_results"][0]
        assert "query_id" in query
        assert "query" in query
        assert "predicted_answer" in query
        assert "metrics" in query

    def test_generate_reports_creates_both_files(self, tmp_path, sample_benchmark_result):
        """generate_reports should create both markdown and JSON files."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        generate_reports(sample_benchmark_result, str(output_dir))

        # Check both files exist
        md_files = list(output_dir.glob("*.md"))
        json_files = list(output_dir.glob("*.json"))

        assert len(md_files) == 1
        assert len(json_files) == 1

    def test_report_file_naming(self, tmp_path, sample_benchmark_result):
        """Report files should have proper naming convention."""
        report_path = save_markdown_report(sample_benchmark_result, str(tmp_path))

        # Check filename format: <dataset>_<timestamp>.md
        assert "TestDataset" in report_path.name or "benchmark_report" in report_path.name
        assert report_path.suffix == ".md"
