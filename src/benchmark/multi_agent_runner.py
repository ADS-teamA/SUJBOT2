"""
Multi-Agent Benchmark Runner - Evaluates multi-agent system performance.

Migrated from single-agent runner.py with these changes:
- Uses MultiAgentRunner instead of AgentCore
- Async execution (run_query is async)
- Preserves all metrics and reporting
- Compatible with existing BenchmarkConfig and datasets
"""

import asyncio
import time
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from ..multi_agent.runner import MultiAgentRunner
from ..agent.config import AgentConfig
from ..hybrid_search import HybridVectorStore
from ..cost_tracker import get_global_tracker, reset_global_tracker

from .config import BenchmarkConfig
from .dataset import BenchmarkDataset, QueryExample
from .metrics import compute_all_metrics, aggregate_metrics, format_metrics
from .models import QueryResult, BenchmarkResult

logger = logging.getLogger(__name__)


class MultiAgentBenchmarkRunner:
    """
    Benchmark runner for multi-agent system.

    Evaluates multi-agent workflow on standardized QA datasets.
    Preserves all metrics and reporting from single-agent runner.

    Usage:
        config = BenchmarkConfig.from_env()
        runner = MultiAgentBenchmarkRunner(config)
        result = asyncio.run(runner.run())
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize multi-agent benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config

        # Load dataset
        logger.info(f"Loading dataset from {config.dataset_path}")
        self.dataset = BenchmarkDataset.from_json(config.dataset_path)

        # Print dataset statistics
        stats = self.dataset.get_statistics()
        logger.info(
            f"Dataset statistics: {stats['total_queries']} queries, "
            f"{stats['unique_source_documents']} source documents"
        )

        # Load vector store
        logger.info(f"Loading vector store from {config.vector_store_path}")
        self.vector_store = HybridVectorStore.load(config.vector_store_path)

        # Initialize multi-agent runner
        logger.info("Initializing multi-agent system")
        self.agent_runner = self._initialize_multi_agent()

        logger.info("Multi-agent benchmark runner initialized successfully")

    def _initialize_multi_agent(self) -> MultiAgentRunner:
        """
        Initialize multi-agent runner with benchmark-specific configuration.

        Returns:
            MultiAgentRunner instance configured for benchmark evaluation
        """
        # Load main config (needed for API keys)
        try:
            from ..config import load_config
            main_config = load_config()
        except Exception as e:
            logger.warning(f"Could not load main config: {e}, using AgentConfig.from_env()")
            agent_config = AgentConfig.from_env(
                vector_store_path=Path(self.config.vector_store_path)
            )
            main_config = {
                "api_keys": {
                    "anthropic_api_key": agent_config.anthropic_api_key,
                    "openai_api_key": agent_config.openai_api_key,
                    "google_api_key": agent_config.google_api_key,
                },
                "vector_store_path": str(self.config.vector_store_path),
            }

        # Load multi-agent configuration
        config_path = Path("config.json")
        multi_agent_config = {}

        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
                    multi_agent_config = full_config.get("multi_agent", {})
            except Exception as e:
                logger.warning(f"Could not load multi_agent config from config.json: {e}")

        # If no multi_agent config, load from extension file
        if not multi_agent_config:
            extension_path = Path("config_multi_agent_extension.json")
            if extension_path.exists():
                try:
                    with open(extension_path) as f:
                        extension_config = json.load(f)
                        multi_agent_config = extension_config.get("multi_agent", {})
                except Exception as e:
                    logger.error(f"Could not load config_multi_agent_extension.json: {e}")
                    raise

        if not multi_agent_config:
            raise ValueError(
                "Multi-agent configuration not found. "
                "Add 'multi_agent' section to config.json or ensure config_multi_agent_extension.json exists."
            )

        # Override model settings for benchmark (use benchmark config preferences)
        if "orchestrator" in multi_agent_config:
            multi_agent_config["orchestrator"]["model"] = self.config.agent_model
            multi_agent_config["orchestrator"]["temperature"] = self.config.agent_temperature
            multi_agent_config["orchestrator"]["enable_prompt_caching"] = self.config.enable_prompt_caching

        # Build runner configuration
        runner_config = {
            **main_config,
            "multi_agent": multi_agent_config,
        }

        # Initialize runner
        runner = MultiAgentRunner(runner_config)

        logger.info(
            f"Multi-agent runner initialized: "
            f"model={self.config.agent_model}, "
            f"temperature={self.config.agent_temperature}, "
            f"caching={'enabled' if self.config.enable_prompt_caching else 'disabled'}"
        )

        return runner

    async def _evaluate_query(self, query_example: QueryExample) -> QueryResult:
        """
        Evaluate single query with multi-agent system.

        Args:
            query_example: Query to evaluate

        Returns:
            QueryResult with metrics and timing

        Raises:
            RuntimeError: If agent fails (when fail_fast=True)
        """
        query_text = query_example.query
        expected_answers = query_example.get_expected_answers()

        # Track cost and time
        tracker = get_global_tracker()
        cost_before = tracker.get_total_cost()
        start_time = time.time()

        # Get multi-agent response
        response_data = await self._get_agent_response(query_example)

        # Calculate timing
        elapsed_ms = (time.time() - start_time) * 1000

        # Extract answer from multi-agent result
        extracted_answer = self._extract_answer_from_result(response_data, query_example.query_id)

        # Calculate query cost (from multi-agent workflow)
        query_cost = response_data.get("total_cost_cents", 0.0) / 100.0  # Convert cents to USD

        # Compute metrics
        metrics = compute_all_metrics(extracted_answer, expected_answers)

        # Extract RAG confidence (not yet implemented in multi-agent)
        rag_confidence = None  # TODO: Implement in multi-agent system

        return QueryResult(
            query_id=query_example.query_id,
            query=query_text,
            predicted_answer=extracted_answer,
            ground_truth_answers=expected_answers,
            metrics=metrics,
            retrieval_time_ms=elapsed_ms,
            cost_usd=query_cost,
            rag_confidence=rag_confidence,
        )

    async def _get_agent_response(self, query_example: QueryExample) -> Dict[str, Any]:
        """Get multi-agent response, handling errors based on fail_fast setting."""
        try:
            # Consume async generator to get final result
            result = None
            async for event in self.agent_runner.run_query(query_example.query):
                if event.get("type") == "final":
                    result = event
                    break

            if not result:
                raise RuntimeError("No final result returned from multi-agent system")

            return result
        except Exception as e:
            error_msg = f"Multi-agent failed for query {query_example.query_id}: {e}"
            if self.config.fail_fast:
                raise RuntimeError(error_msg) from e
            logger.error(error_msg)
            return {"success": False, "final_answer": "", "errors": [str(e)]}

    def _extract_answer_from_result(self, response_data: Dict[str, Any], query_id: int) -> str:
        """
        Extract answer from multi-agent result.

        Multi-agent returns structured dict with 'final_answer' field.
        We still try to extract from <ANSWER> tags for consistency.
        """
        if not response_data.get("success", False):
            logger.warning(f"Query {query_id} failed: {response_data.get('errors', [])}")
            return ""

        final_answer = response_data.get("final_answer", "")

        # Try to extract from <ANSWER> tags (if report generator used them)
        import re
        answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", final_answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            logger.debug(f"Extracted answer from tags: {answer[:100]}...")
            return answer

        # Fallback: use full response
        if not final_answer:
            logger.warning(f"No final_answer for query {query_id}, using empty string")
        return final_answer

    async def run(self) -> BenchmarkResult:
        """
        Run complete benchmark evaluation (async).

        Returns:
            BenchmarkResult with aggregated metrics and per-query results

        Raises:
            RuntimeError: If evaluation fails (when fail_fast=True)
        """
        logger.info("=" * 80)
        logger.info("STARTING MULTI-AGENT BENCHMARK EVALUATION")
        logger.info("=" * 80)

        # Initialize multi-agent system (async)
        logger.info("Initializing multi-agent system...")
        await self.agent_runner.initialize()
        logger.info("Multi-agent system initialized")

        # Reset cost tracker for this benchmark run
        reset_global_tracker()
        tracker = get_global_tracker()

        # Get queries to evaluate
        queries = self.dataset.get_queries(max_queries=self.config.max_queries)
        logger.info(f"Evaluating {len(queries)} queries")

        # Evaluation loop with progress bar
        start_time = time.time()
        query_results = []

        progress_bar = tqdm(
            queries,
            desc="Evaluating queries",
            unit="query",
            disable=self.config.debug_mode,  # Disable in debug mode
        )

        for query_example in progress_bar:
            try:
                result = await self._evaluate_query(query_example)
                query_results.append(result)

                # Update progress bar with metrics
                progress_bar.set_postfix(
                    {
                        "EM": f"{result.metrics.get('exact_match', 0):.2f}",
                        "F1": f"{result.metrics.get('f1_score', 0):.2f}",
                    }
                )

                # Log per-query results in debug mode
                if self.config.debug_mode:
                    logger.info(
                        f"[{result.query_id}/{len(queries)}] "
                        f"{format_metrics(result.metrics)} | "
                        f"Time: {result.retrieval_time_ms:.0f}ms | "
                        f"Cost: ${result.cost_usd:.6f}"
                    )

                # Rate limiting delay (for API quotas like Gemini Free Tier)
                if self.config.rate_limit_delay > 0:
                    await asyncio.sleep(self.config.rate_limit_delay)

            except Exception as e:
                # This should only happen if fail_fast=False
                logger.error(f"Skipping query {query_example.query_id}: {e}")
                continue

        # Calculate total time and cost
        total_time = time.time() - start_time
        total_cost = tracker.get_total_cost()

        # Aggregate metrics
        all_metrics = [qr.metrics for qr in query_results]
        aggregated = aggregate_metrics(all_metrics)

        # Aggregate RAG confidence (if available in future)
        confidence_scores = [
            qr.rag_confidence for qr in query_results if qr.rag_confidence is not None
        ]
        if confidence_scores:
            aggregated["rag_confidence"] = sum(confidence_scores) / len(confidence_scores)
            logger.info(
                f"Average RAG Confidence: {aggregated['rag_confidence']:.4f} "
                f"({len(confidence_scores)}/{len(query_results)} queries)"
            )

        # Create result object
        result = BenchmarkResult(
            dataset_name="PrivacyQA (Multi-Agent)",
            total_queries=len(query_results),
            aggregate_metrics=aggregated,
            query_results=query_results,
            total_time_seconds=total_time,
            total_cost_usd=total_cost,
            config=self.config.to_dict(),
        )

        logger.info("=" * 80)
        logger.info("MULTI-AGENT BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Queries evaluated: {result.total_queries}")
        logger.info(f"Total time: {result.total_time_seconds:.1f}s")
        logger.info(f"Total cost: ${result.total_cost_usd:.4f}")
        logger.info(f"Aggregate metrics: {format_metrics(aggregated, precision=3)}")

        # Shutdown multi-agent system
        self.agent_runner.shutdown()

        return result
