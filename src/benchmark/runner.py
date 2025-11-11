"""
Benchmark runner - Legacy single-agent runner (DEPRECATED).

DEPRECATED: Use multi_agent_runner.py instead.
This file is kept for backward compatibility with dataclass definitions only.

Coordinates:
1. Dataset loading
2. Agent initialization
3. Query evaluation loop
4. Metric computation
5. Cost tracking
6. Progress reporting
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Legacy imports removed (use multi_agent_runner.py)
# from ..agent.agent_core import AgentCore  # DELETED
from ..agent.config import AgentConfig
# from ..agent.prompt_loader import load_prompt  # Not used in multi-agent
from ..hybrid_search import HybridVectorStore
from ..cost_tracker import get_global_tracker, reset_global_tracker

from .config import BenchmarkConfig
from .dataset import BenchmarkDataset, QueryExample
from .metrics import compute_all_metrics, aggregate_metrics, format_metrics

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """
    Result for a single query evaluation.

    Attributes:
        query_id: Unique query identifier
        query: Question text
        predicted_answer: Model's answer
        ground_truth_answers: List of expected answers
        metrics: Dict of metric scores
        retrieval_time_ms: Time to retrieve + generate answer
        cost_usd: API cost for this query
        rag_confidence: RAG confidence score (0-1) or None if unavailable
    """

    query_id: int
    query: str
    predicted_answer: str
    ground_truth_answers: List[str]
    metrics: Dict[str, float]
    retrieval_time_ms: float
    cost_usd: float
    rag_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        result = {
            "query_id": self.query_id,
            "query": self.query,
            "predicted_answer": self.predicted_answer,
            "ground_truth_answers": self.ground_truth_answers,
            "metrics": self.metrics,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }

        # Add RAG confidence if available
        if self.rag_confidence is not None:
            result["rag_confidence"] = round(self.rag_confidence, 4)

        return result


@dataclass
class BenchmarkResult:
    """
    Complete benchmark evaluation results.

    Attributes:
        dataset_name: Name of dataset evaluated
        total_queries: Number of queries evaluated
        aggregate_metrics: Mean scores per metric
        query_results: Per-query detailed results
        total_time_seconds: Total evaluation time
        total_cost_usd: Total API cost
        config: Benchmark configuration
        timestamp: When evaluation was run
    """

    dataset_name: str
    total_queries: int
    aggregate_metrics: Dict[str, float]
    query_results: List[QueryResult]
    total_time_seconds: float
    total_cost_usd: float
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "dataset_name": self.dataset_name,
            "total_queries": self.total_queries,
            "aggregate_metrics": self.aggregate_metrics,
            "query_results": [qr.to_dict() for qr in self.query_results],
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_time_per_query_ms": (
                round((self.total_time_seconds * 1000) / self.total_queries, 2)
                if self.total_queries > 0
                else 0
            ),
            "cost_per_query_usd": (
                round(self.total_cost_usd / self.total_queries, 6) if self.total_queries > 0 else 0
            ),
            "config": self.config,
            "timestamp": self.timestamp,
        }


class BenchmarkRunner:
    """
    DEPRECATED: Legacy single-agent benchmark runner.

    Use MultiAgentBenchmarkRunner from multi_agent_runner.py instead.

    This class is deprecated and will raise an error if instantiated.
    Kept for backward compatibility with import statements only.

    For new code, use:
        from src.benchmark.multi_agent_runner import MultiAgentBenchmarkRunner
        runner = MultiAgentBenchmarkRunner(config)
        result = await runner.run()
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner (DEPRECATED).

        Args:
            config: Benchmark configuration

        Raises:
            RuntimeError: Always raises - use MultiAgentBenchmarkRunner instead
        """
        raise RuntimeError(
            "BenchmarkRunner is deprecated. "
            "Use MultiAgentBenchmarkRunner from src.benchmark.multi_agent_runner instead. "
            "Example: runner = MultiAgentBenchmarkRunner(config); await runner.run()"
        )

        # Legacy code below (unreachable) - kept for reference
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

        # Initialize agent
        logger.info("Initializing RAG agent")
        self.agent = self._initialize_agent()

        logger.info("Benchmark runner initialized successfully")

    def _initialize_agent(self) -> Any:  # AgentCore (deleted)
        """
        Initialize agent with benchmark-specific configuration.

        Returns:
            AgentCore instance configured for benchmark evaluation
        """
        # Create agent config with benchmark settings
        agent_config = AgentConfig.from_env(
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            enable_prompt_caching=self.config.enable_prompt_caching,
            vector_store_path=Path(self.config.vector_store_path),
            debug_mode=self.config.debug_mode,
        )

        # Override tool config for retrieval parameters
        agent_config.tool_config.default_k = self.config.k
        agent_config.tool_config.enable_reranking = self.config.enable_reranking
        agent_config.tool_config.enable_graph_boost = self.config.enable_graph_boost

        # Initialize pipeline components
        components = self._initialize_pipeline_components(agent_config)

        # Initialize tool registry
        self._initialize_tool_registry(components, agent_config.tool_config)

        # Create and configure agent
        agent = AgentCore(agent_config)

        # Append benchmark-specific prompt
        benchmark_prompt = load_prompt("agent_benchmark_prompt")
        agent.config.system_prompt = f"{agent.config.system_prompt}\n\n---\n\n{benchmark_prompt}"
        logger.info("Benchmark system prompt appended to agent prompt")

        # Initialize with documents
        agent.initialize_with_documents()

        logger.info(
            f"Agent initialized: {agent_config.model}, k={self.config.k}, "
            f"reranking={'enabled' if self.config.enable_reranking else 'disabled'}"
        )

        return agent

    def _initialize_pipeline_components(self, agent_config: AgentConfig) -> Dict[str, Any]:
        """Initialize embedder, reranker, and context assembler."""
        from ..embedding_generator import EmbeddingGenerator, EmbeddingConfig
        from ..reranker import CrossEncoderReranker
        from ..context_assembly import ContextAssembler, CitationFormat

        components = {}

        # Create embedder
        logger.info(f"Initializing embedder: {agent_config.embedding_model}")
        components["embedder"] = EmbeddingGenerator(
            EmbeddingConfig(model=agent_config.embedding_model, batch_size=100, normalize=True)
        )

        # Create reranker if enabled
        components["reranker"] = None
        if self.config.enable_reranking:
            logger.info(f"Initializing reranker: {agent_config.tool_config.reranker_model}")
            try:
                components["reranker"] = CrossEncoderReranker(
                    model_name=agent_config.tool_config.reranker_model
                )
                logger.info(f"✓ Reranker loaded: {agent_config.tool_config.reranker_model}")
            except Exception as e:
                import time

                error_id = f"ERR_RERANKER_LOAD_{int(time.time())}"
                logger.error(
                    f"[{error_id}] Failed to load reranker model "
                    f"'{agent_config.tool_config.reranker_model}': {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"[{error_id}] Reranking is enabled but model failed to load. "
                    f"Either fix the model or set enable_reranking=False in config. "
                    f"Cannot proceed with invalid benchmark configuration that would "
                    f"produce incomparable results."
                ) from e

        # Create context assembler
        components["context_assembler"] = ContextAssembler(citation_format=CitationFormat.INLINE)

        return components

    def _initialize_tool_registry(self, components: Dict[str, Any], tool_config: Any) -> None:
        """Initialize tool registry with pipeline components."""
        from ..agent.tools.registry import get_registry
        from ..graph.models import KnowledgeGraph
        from ..agent.graph_adapter import SimpleGraphAdapter

        logger.info("Initializing tool registry...")

        # Try to load knowledge graph if available
        knowledge_graph = None
        kg_path = Path(self.config.vector_store_path).parent / "unified_kg.json"
        if kg_path.exists():
            try:
                logger.info(f"Loading knowledge graph from {kg_path}")
                kg = KnowledgeGraph.load_json(str(kg_path))
                # Wrap in SimpleGraphAdapter for tools compatibility
                knowledge_graph = SimpleGraphAdapter(kg)
                logger.info(
                    f"✓ KG loaded: {len(kg.entities)} entities, "
                    f"{len(kg.relationships)} relationships"
                )
            except Exception as e:
                logger.warning(f"Failed to load knowledge graph: {e}")
                knowledge_graph = None
        else:
            logger.info(f"No knowledge graph found at {kg_path} (optional)")

        registry = get_registry()
        registry.initialize_tools(
            vector_store=self.vector_store,
            embedder=components["embedder"],
            reranker=components["reranker"],
            graph_retriever=None,  # No graph retriever for benchmark
            knowledge_graph=knowledge_graph,
            context_assembler=components["context_assembler"],
            config=tool_config,
        )

        # Count available vs unavailable tools
        total_tools = len(registry)
        unavailable = registry.get_unavailable_tools() if hasattr(registry, 'get_unavailable_tools') else []
        available = total_tools - len(unavailable)

        if unavailable:
            logger.info(
                f"Tool registry initialized: {available}/{total_tools} tools available"
            )
            logger.info(f"Unavailable tools: {unavailable}")
        else:
            logger.info(f"Tool registry initialized: {total_tools} tools available")

    def _evaluate_query(self, query_example: QueryExample) -> QueryResult:
        """
        Evaluate single query.

        Args:
            query_example: Query to evaluate

        Returns:
            QueryResult with metrics and timing

        Raises:
            RuntimeError: If agent fails (when fail_fast=True)
        """
        query_text = query_example.query
        expected_answers = query_example.get_expected_answers()

        # Reset conversation for independent query
        self.agent.reset_conversation()

        # Track cost and time
        tracker = get_global_tracker()
        cost_before = tracker.get_total_cost()
        start_time = time.time()

        # Get agent response
        response_text = self._get_agent_response(query_example)

        # Calculate metrics
        elapsed_ms = (time.time() - start_time) * 1000
        query_cost = tracker.get_total_cost() - cost_before

        # Extract answer and compute metrics
        extracted_answer = self._extract_answer(response_text, query_example.query_id)
        metrics = compute_all_metrics(extracted_answer, expected_answers)

        # Extract RAG confidence if available
        rag_confidence = self._extract_rag_confidence()

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

    def _get_agent_response(self, query_example: QueryExample) -> str:
        """Get agent response, handling errors based on fail_fast setting."""
        try:
            return self.agent.process_message(query_example.query, stream=False)
        except Exception as e:
            error_msg = f"Agent failed for query {query_example.query_id}: {e}"
            if self.config.fail_fast:
                raise RuntimeError(error_msg) from e
            logger.error(error_msg)
            return ""

    def _extract_answer(self, response_text: str, query_id: int) -> str:
        """Extract answer from response tags or use full response as fallback."""
        import re

        answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", response_text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            logger.debug(f"Extracted answer from tags: {answer[:100]}...")
            return answer

        # Fallback: use full response if tags not found
        logger.warning(f"No <ANSWER> tags found for query {query_id}, using full response")
        return response_text

    def _extract_rag_confidence(self) -> Optional[float]:
        """Extract RAG confidence score from agent if available."""
        try:
            confidence_data = self.agent.get_latest_rag_confidence()
            if not confidence_data:
                return None

            rag_confidence = confidence_data.get("overall_confidence")
            if self.config.debug_mode and rag_confidence is not None:
                logger.debug(
                    f"RAG Confidence: {rag_confidence:.3f} "
                    f"({confidence_data.get('interpretation', 'Unknown')})"
                )
            return rag_confidence
        except AttributeError as e:
            # Agent doesn't have get_latest_rag_confidence method
            import time

            error_id = f"ERR_RAG_CONF_ATTR_{int(time.time())}"
            logger.warning(f"[{error_id}] Agent missing get_latest_rag_confidence() method: {e}")
            return None
        except (KeyError, TypeError, ValueError) as e:
            # Confidence dict has unexpected structure
            import time

            error_id = f"ERR_RAG_CONF_PARSE_{int(time.time())}"
            logger.warning(
                f"[{error_id}] Failed to parse RAG confidence: {e}. "
                f"Expected dict with 'overall_confidence' key."
            )
            return None
        except Exception as e:
            # Unexpected error - this SHOULD be investigated
            import time

            error_id = f"ERR_RAG_CONF_UNKNOWN_{int(time.time())}"
            logger.error(
                f"[{error_id}] Unexpected error in RAG confidence extraction: {e}", exc_info=True
            )
            return None

    def run(self) -> BenchmarkResult:
        """
        Run complete benchmark evaluation.

        Returns:
            BenchmarkResult with aggregated metrics and per-query results

        Raises:
            RuntimeError: If evaluation fails (when fail_fast=True)
        """
        logger.info("=" * 80)
        logger.info("STARTING BENCHMARK EVALUATION")
        logger.info("=" * 80)

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
                result = self._evaluate_query(query_example)
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
                    time.sleep(self.config.rate_limit_delay)

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

        # Aggregate RAG confidence (if available)
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
            dataset_name="PrivacyQA",
            total_queries=len(query_results),
            aggregate_metrics=aggregated,
            query_results=query_results,
            total_time_seconds=total_time,
            total_cost_usd=total_cost,
            config=self.config.to_dict(),
        )

        logger.info("=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Queries evaluated: {result.total_queries}")
        logger.info(f"Total time: {result.total_time_seconds:.1f}s")
        logger.info(f"Total cost: ${result.total_cost_usd:.4f}")
        logger.info(f"Aggregate metrics: {format_metrics(aggregated, precision=3)}")

        return result
