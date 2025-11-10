"""
Tests for BenchmarkRunner module.

Critical tests for benchmark orchestration, agent failure handling, and result aggregation.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.benchmark.runner import BenchmarkRunner
from src.benchmark.config import BenchmarkConfig
from src.benchmark.dataset import QueryExample, GroundTruthSnippet


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a minimal test dataset."""
    dataset_json = tmp_path / "test_dataset.json"
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
                                "answer": "Privacy is control of personal information",
                            }
                        ],
                    },
                    {
                        "query": "What is GDPR?",
                        "snippets": [
                            {
                                "file_path": "gdpr.txt",
                                "span": [0, 30],
                                "answer": "GDPR is EU privacy regulation",
                            }
                        ],
                    },
                ]
            }
        )
    )
    return str(dataset_json)


@pytest.fixture
def mock_vector_store(tmp_path):
    """Create a minimal mock vector store."""
    vector_store = tmp_path / "vector_store"
    vector_store.mkdir()
    (vector_store / "faiss_metadata.json").write_text("{}")
    return str(vector_store)


@pytest.fixture
def mock_config(mock_dataset, mock_vector_store, tmp_path):
    """Create a test configuration."""
    return BenchmarkConfig(
        dataset_path=mock_dataset,
        vector_store_path=mock_vector_store,
        output_dir=str(tmp_path / "output"),
        max_queries=2,
        fail_fast=False,
    )


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.process_message = Mock(return_value="<ANSWER>test answer</ANSWER>")
    agent.get_latest_rag_confidence = Mock(
        return_value={"overall_confidence": 0.85, "interpretation": "HIGH"}
    )
    agent.reset_conversation = Mock()
    return agent


class TestAgentFailureHandling:
    """Test agent failure scenarios."""

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_agent_failure_with_fail_fast_true(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Agent failure with fail_fast=True should raise RuntimeError."""
        mock_config.fail_fast = True
        mock_agent.process_message.side_effect = Exception("API Error")
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)

        with pytest.raises(RuntimeError, match="Agent failed for query"):
            runner.run()

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_agent_failure_with_fail_fast_false(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Agent failure with fail_fast=False should continue with empty answer."""
        mock_config.fail_fast = False
        mock_agent.process_message.side_effect = Exception("API Error")
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        # Should complete with empty answers
        assert result.total_queries == 2
        assert all(qr.predicted_answer == "" for qr in result.query_results)


class TestAnswerExtraction:
    """Test answer extraction from agent responses."""

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_extract_answer_with_tags(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Response with ANSWER tags should extract content."""
        mock_agent.process_message.return_value = (
            "Let me search...\n<ANSWER>Extracted answer</ANSWER>\nHope this helps!"
        )
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(qr.predicted_answer == "Extracted answer" for qr in result.query_results)

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_extract_answer_without_tags(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent, caplog
    ):
        """Response without ANSWER tags should use full response and warn."""
        mock_agent.process_message.return_value = "Full response without tags"
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(
            qr.predicted_answer == "Full response without tags" for qr in result.query_results
        )
        assert "No <ANSWER> tags found" in caplog.text

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_extract_answer_empty_tags(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Empty ANSWER tags should extract empty string."""
        mock_agent.process_message.return_value = "<ANSWER></ANSWER>"
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(qr.predicted_answer == "" for qr in result.query_results)


class TestRAGConfidenceExtraction:
    """Test RAG confidence extraction."""

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_rag_confidence_successful(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Successful confidence extraction should set value."""
        mock_agent.get_latest_rag_confidence.return_value = {"overall_confidence": 0.85}
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(qr.rag_confidence == 0.85 for qr in result.query_results)

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_rag_confidence_missing_method(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent, caplog
    ):
        """Missing get_latest_rag_confidence method should set None and warn."""
        delattr(mock_agent, "get_latest_rag_confidence")
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(qr.rag_confidence is None for qr in result.query_results)
        assert "missing" in caplog.text.lower() or "failed" in caplog.text.lower()

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_rag_confidence_invalid_structure(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent, caplog
    ):
        """Invalid confidence structure should set None and warn."""
        mock_agent.get_latest_rag_confidence.return_value = {"wrong_key": 0.85}
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert all(qr.rag_confidence is None for qr in result.query_results)


class TestResultAggregation:
    """Test result aggregation and statistics."""

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_empty_results_aggregation(self, mock_agent_class, mock_vector_store_class, tmp_path):
        """Empty results should aggregate without errors."""
        # Create config with max_queries=0
        dataset_json = tmp_path / "empty_dataset.json"
        dataset_json.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        config = BenchmarkConfig(
            dataset_path=str(dataset_json),
            vector_store_path=str(vector_store),
            output_dir=str(tmp_path / "output"),
            max_queries=0,
        )

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(config)

        # Should raise ValueError for no valid queries
        with pytest.raises(ValueError, match="No valid queries"):
            runner.run()

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_max_queries_limit(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """max_queries should limit the number of queries evaluated."""
        mock_config.max_queries = 1  # Only evaluate first query
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert len(result.query_results) == 1
        assert result.total_queries == 1

    @patch("src.benchmark.runner.HybridVectorStore")
    @patch("src.benchmark.runner.AgentCore")
    def test_cost_and_time_tracking(
        self, mock_agent_class, mock_vector_store_class, mock_config, mock_agent
    ):
        """Cost and time should be tracked correctly."""
        mock_agent_class.return_value = mock_agent

        mock_store = Mock()
        mock_store.get_all_documents.return_value = []
        mock_vector_store_class.load.return_value = mock_store

        runner = BenchmarkRunner(mock_config)
        result = runner.run()

        assert result.total_time_seconds > 0
        assert result.total_cost_usd >= 0
        assert all(qr.retrieval_time_ms > 0 for qr in result.query_results)
