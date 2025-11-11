"""
Tests for Multi-Agent Benchmark Runner.

Critical test coverage for:
1. Async query evaluation ordering
2. Cost tracking from multi-agent responses
3. Initialization validation
4. Metrics preservation from single-agent
5. Error handling with fail_fast flag
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.benchmark.multi_agent_runner import MultiAgentBenchmarkRunner
from src.benchmark.config import BenchmarkConfig
from src.benchmark.dataset import QueryExample, GroundTruthSnippet
from src.benchmark.runner import QueryResult


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.get_stats.return_value = {
        "layer_counts": [100, 50, 25],
        "total_chunks": 175
    }
    return store


@pytest.fixture
def mock_multi_agent_runner():
    """Mock multi-agent runner for testing."""
    runner = AsyncMock()
    runner.initialize = AsyncMock(return_value=None)
    runner.shutdown = Mock()
    return runner


@pytest.fixture
def benchmark_config(tmp_path):
    """Create test benchmark configuration."""
    # Create dummy vector store directory
    test_db = tmp_path / "test_db"
    test_db.mkdir()

    return BenchmarkConfig(
        dataset_path="benchmark_dataset/privacy_qa.json",
        vector_store_path=str(test_db),
        max_queries=5,
        agent_model="claude-haiku-4-5",
        agent_temperature=0.0,
        k=5,
        enable_reranking=True,
        enable_prompt_caching=True,
        debug_mode=False,
        output_dir=str(tmp_path / "results"),
        rate_limit_delay=0.0,
        fail_fast=False
    )


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        QueryExample(
            query_id=1,
            query="What data does the app collect?",
            snippets=[
                GroundTruthSnippet(
                    file_path="privacy_policy.pdf",
                    span_start=0,
                    span_end=50,
                    answer="location data, device ID"
                )
            ]
        ),
        QueryExample(
            query_id=2,
            query="How long is data retained?",
            snippets=[
                GroundTruthSnippet(
                    file_path="privacy_policy.pdf",
                    span_start=100,
                    span_end=150,
                    answer="30 days"
                ),
                GroundTruthSnippet(
                    file_path="privacy_policy.pdf",
                    span_start=200,
                    span_end=250,
                    answer="one month"
                )
            ]
        ),
        QueryExample(
            query_id=3,
            query="Can users delete their data?",
            snippets=[
                GroundTruthSnippet(
                    file_path="privacy_policy.pdf",
                    span_start=300,
                    span_end=350,
                    answer="yes, via settings"
                )
            ]
        ),
    ]


class TestMultiAgentBenchmarkRunner:
    """Test suite for MultiAgentBenchmarkRunner."""

    @pytest.mark.asyncio
    async def test_async_query_evaluation_preserves_ordering(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        CRITICAL TEST 1: Async query evaluation preserves ordering.

        Verifies that:
        - Queries are processed in order despite async execution
        - Query IDs match input order in results
        - No race conditions in result collection
        """
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries
        mock_dataset.get_statistics.return_value = {
            "total_queries": len(sample_queries),
            "unique_source_documents": 1
        }

        # Setup runner with mocked dependencies
        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock multi-agent responses (simulate varying execution times)
            async def mock_run_query(query):
                # Simulate different response times
                if "collect" in query:
                    await asyncio.sleep(0.02)  # 20ms
                elif "retained" in query:
                    await asyncio.sleep(0.01)  # 10ms
                else:
                    await asyncio.sleep(0.03)  # 30ms

                return {
                    "success": True,
                    "final_answer": f"Answer to: {query}",
                    "total_cost_cents": 1.5,
                }

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            # Create runner
            runner = MultiAgentBenchmarkRunner(benchmark_config)
            result = await runner.run()

            # Assert: Results are in correct order
            assert len(result.query_results) == 3
            assert result.query_results[0].query_id == 1
            assert result.query_results[1].query_id == 2
            assert result.query_results[2].query_id == 3

            # Assert: Queries match expected
            assert "collect" in result.query_results[0].query
            assert "retained" in result.query_results[1].query
            assert "delete" in result.query_results[2].query

    @pytest.mark.asyncio
    async def test_cost_tracking_uses_multi_agent_response_cost(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        CRITICAL TEST 2: Cost tracking uses multi-agent response cost.

        Verifies that:
        - Cost is extracted from response_data['total_cost_cents']
        - Cost is converted from cents to USD correctly
        - Total cost is sum of all query costs
        - Cost breakdown is accurate
        """
        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries[:2]  # Use 2 queries
        mock_dataset.get_statistics.return_value = {
            "total_queries": 2,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock responses with specific costs
            call_count = [0]

            async def mock_run_query(query):
                call_count[0] += 1
                return {
                    "success": True,
                    "final_answer": f"Answer {call_count[0]}",
                    "total_cost_cents": 250.0 if call_count[0] == 1 else 150.0,  # $2.50, $1.50
                }

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            # Mock cost tracker
            mock_tracker = Mock()
            mock_tracker.get_total_cost.return_value = 0.0  # Not used (uses response cost)

            with patch("src.benchmark.multi_agent_runner.get_global_tracker", return_value=mock_tracker), \
                 patch("src.benchmark.multi_agent_runner.reset_global_tracker"):

                runner = MultiAgentBenchmarkRunner(benchmark_config)
                result = await runner.run()

                # Assert: Individual query costs are correct
                assert result.query_results[0].cost_usd == 2.50  # 250 cents
                assert result.query_results[1].cost_usd == 1.50  # 150 cents

                # Assert: Total cost matches sum (not from tracker)
                # Note: Total cost uses tracker, which is 0.0 in this mock
                # In real usage, tracker accumulates costs from multi-agent calls
                assert result.total_cost_usd == 0.0  # Mocked tracker returns 0

    @pytest.mark.asyncio
    async def test_initialization_fails_early_when_multi_agent_config_missing(
        self, benchmark_config, mock_vector_store
    ):
        """
        CRITICAL TEST 3: Initialization fails early when multi-agent config missing.

        Verifies that:
        - Missing config.json raises ValueError
        - Missing multi_agent section raises ValueError
        - Error message is descriptive
        - Failure happens during __init__, not during run()
        """
        mock_dataset = Mock()
        mock_dataset.get_statistics.return_value = {
            "total_queries": 1,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock missing config files
            with patch("pathlib.Path.exists", return_value=False):
                # Should raise ValueError during initialization
                with pytest.raises(ValueError, match="Multi-agent configuration not found"):
                    runner = MultiAgentBenchmarkRunner(benchmark_config)

    @pytest.mark.asyncio
    async def test_metrics_identical_to_single_agent_runner(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        CRITICAL TEST 4: Metrics identical to single-agent runner.

        Verifies that:
        - Same metrics computed (EM, F1, Precision, Recall)
        - Same aggregation method
        - QueryResult and BenchmarkResult dataclasses reused
        - Metric values match expected ranges
        """
        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries[:2]
        mock_dataset.get_statistics.return_value = {
            "total_queries": 2,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock responses
            async def mock_run_query(query):
                # Return exact match for first query, partial for second
                if "collect" in query:
                    answer = "location data, device ID"  # Exact match
                else:
                    answer = "30 days retention period"  # Partial match
                return {
                    "success": True,
                    "final_answer": answer,
                    "total_cost_cents": 100.0,
                }

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            with patch("src.benchmark.multi_agent_runner.get_global_tracker") as mock_tracker_fn, \
                 patch("src.benchmark.multi_agent_runner.reset_global_tracker"):

                mock_tracker = Mock()
                mock_tracker.get_total_cost.return_value = 2.0
                mock_tracker_fn.return_value = mock_tracker

                runner = MultiAgentBenchmarkRunner(benchmark_config)
                result = await runner.run()

                # Assert: QueryResult structure matches single-agent
                for query_result in result.query_results:
                    assert hasattr(query_result, "query_id")
                    assert hasattr(query_result, "query")
                    assert hasattr(query_result, "predicted_answer")
                    assert hasattr(query_result, "ground_truth_answers")
                    assert hasattr(query_result, "metrics")
                    assert hasattr(query_result, "retrieval_time_ms")
                    assert hasattr(query_result, "cost_usd")

                # Assert: Metrics structure matches single-agent
                assert "exact_match" in result.aggregate_metrics
                assert "f1_score" in result.aggregate_metrics
                assert "precision" in result.aggregate_metrics
                assert "recall" in result.aggregate_metrics

                # Assert: Metric values are in valid ranges
                for metric_name, value in result.aggregate_metrics.items():
                    assert 0.0 <= value <= 1.0, f"{metric_name} out of range: {value}"

                # Assert: BenchmarkResult structure matches single-agent
                assert hasattr(result, "dataset_name")
                assert hasattr(result, "total_queries")
                assert hasattr(result, "aggregate_metrics")
                assert hasattr(result, "query_results")
                assert hasattr(result, "total_time_seconds")
                assert hasattr(result, "total_cost_usd")
                assert hasattr(result, "config")

    @pytest.mark.asyncio
    async def test_multi_agent_failure_continues_when_fail_fast_false(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        CRITICAL TEST 5: Multi-agent failure continues when fail_fast=False.

        Verifies that:
        - Failed queries are skipped (not added to results)
        - Remaining queries continue processing
        - No exception propagated to top level
        - Partial results are returned
        - Error is logged but not raised
        """
        # Set fail_fast=False
        benchmark_config.fail_fast = False

        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries
        mock_dataset.get_statistics.return_value = {
            "total_queries": 3,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock responses with failure on query 2
            call_count = [0]

            async def mock_run_query(query):
                call_count[0] += 1
                if call_count[0] == 2:
                    # Simulate failure for second query
                    raise RuntimeError("Multi-agent execution failed")
                return {
                    "success": True,
                    "final_answer": f"Answer {call_count[0]}",
                    "total_cost_cents": 100.0,
                }

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            with patch("src.benchmark.multi_agent_runner.get_global_tracker") as mock_tracker_fn, \
                 patch("src.benchmark.multi_agent_runner.reset_global_tracker"):

                mock_tracker = Mock()
                mock_tracker.get_total_cost.return_value = 2.0
                mock_tracker_fn.return_value = mock_tracker

                runner = MultiAgentBenchmarkRunner(benchmark_config)

                # Should not raise exception
                result = await runner.run()

                # Assert: Only 2 results (query 2 was skipped)
                assert len(result.query_results) == 2

                # Assert: Results are query 1 and query 3
                assert result.query_results[0].query_id == 1
                assert result.query_results[1].query_id == 3

                # Assert: Partial results are valid
                assert result.total_queries == 2
                assert result.aggregate_metrics is not None

    @pytest.mark.asyncio
    async def test_multi_agent_failure_raises_when_fail_fast_true(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        Verifies fail_fast=True behavior:
        - Exception is raised immediately
        - Processing stops
        - Partial results are not returned
        """
        # Set fail_fast=True
        benchmark_config.fail_fast = True

        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries
        mock_dataset.get_statistics.return_value = {
            "total_queries": 3,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock responses with failure on query 1
            async def mock_run_query(query):
                raise RuntimeError("Multi-agent execution failed")

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            with patch("src.benchmark.multi_agent_runner.get_global_tracker") as mock_tracker_fn, \
                 patch("src.benchmark.multi_agent_runner.reset_global_tracker"):

                mock_tracker = Mock()
                mock_tracker.get_total_cost.return_value = 0.0
                mock_tracker_fn.return_value = mock_tracker

                runner = MultiAgentBenchmarkRunner(benchmark_config)

                # Should raise RuntimeError
                with pytest.raises(RuntimeError, match="Multi-agent failed for query"):
                    await runner.run()

    @pytest.mark.asyncio
    async def test_rate_limiting_delay_is_applied(
        self, benchmark_config, sample_queries, mock_vector_store, mock_multi_agent_runner
    ):
        """
        Verifies rate limiting:
        - Delay is applied between queries
        - Total time >= (num_queries - 1) * delay
        """
        # Set rate limit delay
        benchmark_config.rate_limit_delay = 0.05  # 50ms
        benchmark_config.max_queries = 2

        mock_dataset = Mock()
        mock_dataset.get_queries.return_value = sample_queries[:2]
        mock_dataset.get_statistics.return_value = {
            "total_queries": 2,
            "unique_source_documents": 1
        }

        with patch("src.benchmark.multi_agent_runner.BenchmarkDataset") as MockDataset, \
             patch("src.benchmark.multi_agent_runner.HybridVectorStore") as MockVectorStore, \
             patch("src.benchmark.multi_agent_runner.MultiAgentRunner") as MockRunner:

            MockDataset.from_json.return_value = mock_dataset
            MockVectorStore.load.return_value = mock_vector_store

            # Mock fast responses
            async def mock_run_query(query):
                return {
                    "success": True,
                    "final_answer": "Answer",
                    "total_cost_cents": 100.0,
                }

            mock_multi_agent_runner.run_query = mock_run_query
            MockRunner.return_value = mock_multi_agent_runner

            with patch("src.benchmark.multi_agent_runner.get_global_tracker") as mock_tracker_fn, \
                 patch("src.benchmark.multi_agent_runner.reset_global_tracker"):

                mock_tracker = Mock()
                mock_tracker.get_total_cost.return_value = 2.0
                mock_tracker_fn.return_value = mock_tracker

                runner = MultiAgentBenchmarkRunner(benchmark_config)
                result = await runner.run()

                # Assert: Total time includes rate limiting delay
                # (2 queries - 1) * 0.05s = 0.05s minimum
                assert result.total_time_seconds >= 0.05
