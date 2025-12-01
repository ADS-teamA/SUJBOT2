"""
Common evaluation runner utilities for benchmark scripts.

This module provides reusable components for evaluation workflows,
following the DRY principle to eliminate code duplication.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

from src.multi_agent.runner import MultiAgentRunner
from src.agent.tools import get_registry
from src.utils.eval_metrics import calculate_all_metrics


class EvaluationConfig:
    """Configuration for evaluation runs with sensible defaults."""

    def __init__(
        self,
        dataset_path: str = None,
        output_path: str = None,
        k: int = 100,
        use_hyde: bool = False,
        num_expands: int = 0,
        enable_graph_boost: bool = False,
        search_method: str = "hybrid",
        experiment_name: str = "evaluation"
    ):
        """
        Initialize evaluation configuration.

        Args:
            dataset_path: Path to retrieval dataset JSON
            output_path: Path for results output
            k: Number of top results to retrieve
            use_hyde: Whether to use HyDE query expansion
            num_expands: Number of query expansions (0-2)
            enable_graph_boost: Whether to enable knowledge graph boosting
            search_method: Search method ("hybrid", "bm25", "dense")
            experiment_name: Name for this experiment run
        """
        self.dataset_path = dataset_path or "/app/benchmark_dataset/retrieval.json"
        self.output_path = output_path
        self.k = k
        self.use_hyde = use_hyde
        self.num_expands = num_expands
        self.enable_graph_boost = enable_graph_boost
        self.search_method = search_method
        self.experiment_name = experiment_name

        # Auto-generate output path if not provided
        if not self.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"/app/results/{experiment_name}_{timestamp}_k{k}.json"


class EvaluationRunner:
    """Reusable evaluation runner for benchmark experiments."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation runner.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.runner: Optional[MultiAgentRunner] = None
        self.registry = None
        self.results: List[Dict] = []

    async def initialize(self) -> bool:
        """
        Initialize the MultiAgentRunner and tool registry.

        Returns:
            True if initialization successful, False otherwise
        """
        print("Loading configuration...")
        # Auto-detect paths (Docker vs local)
        if Path("/app/config.json").exists():
            config_path = Path("/app/config.json")
            vector_store_path = "/app/vector_db"
        else:
            config_path = Path("config.json")
            vector_store_path = "vector_db"

        try:
            with open(config_path) as f:
                full_config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Configuration file not found: {config_path}\n"
                "Please create config.json with evaluation settings.\n"
                "See config.json.example for template."
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON in configuration file {config_path}: {e}\n"
                "Fix JSON syntax errors and try again."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load configuration from {config_path}: {e}"
            ) from e

        # Build runner configuration
        # Use PostgreSQL backend (has both dense vectors + BM25)
        storage_config = full_config.get("storage", {}).copy()

        runner_config = {
            "api_keys": {
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
            },
            "vector_store_path": vector_store_path,
            "models": full_config.get("models", {}),
            "storage": storage_config,
            "agent_tools": full_config.get("agent_tools", {}),
            "knowledge_graph": full_config.get("knowledge_graph", {}),
            "neo4j": full_config.get("neo4j", {}),
            "multi_agent": full_config.get("multi_agent", {}),
        }

        backend = storage_config.get("backend", "postgresql")
        print(f"Evaluation using storage backend: {backend}")

        print("Initializing MultiAgentRunner...")
        self.runner = MultiAgentRunner(runner_config)
        success = await self.runner.initialize()

        if not success:
            print("ERROR: Runner initialization failed!")
            return False

        self.registry = get_registry()
        print(f"Tool registry has {len(self.registry._tools)} tools")
        return True

    def _save_training_data(self, chunks: List[Dict], relevant_ids: List[str], metrics: Dict) -> None:
        """
        Save training data for score distribution analysis.

        NEW FORMAT:
        - Only for dense_only and bm25_only (skip hybrid)
        - Format: recall@100, score1, score2, ..., score100 (101 columns)
        - Scores sorted descending (highest score = most relevant = rank 1)
        - One row per query

        Args:
            chunks: Retrieved chunks with scores (k=100)
            relevant_ids: Ground truth relevant chunk IDs
            metrics: Calculated metrics including recall@100
        """
        import csv
        from pathlib import Path

        method = self.config.search_method

        # Only save for dense_only and bm25_only (skip hybrid)
        if method not in ["dense_only", "bm25_only"]:
            return

        # Determine output directory and file
        output_dir = Path(self.config.output_path).parent if self.config.output_path else Path("results/grid_search_k100")

        # Create output directory (including all parent directories)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # If mkdir fails due to permissions or other OS issues, log error
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            # Don't raise - allow evaluation to continue without saving training data
            # (better than failing entire evaluation)
            return

        if method == "dense_only":
            filepath = output_dir / "dense_training_data.csv"
        else:  # bm25_only
            filepath = output_dir / "bm25_training_data.csv"

        # Create file with headers if doesn't exist
        if not filepath.exists():
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: recall@100, score1, score2, ..., score100
                headers = ["recall@100"] + [f"score{i+1}" for i in range(100)]
                writer.writerow(headers)

        # Extract scores from chunks (already sorted by search method)
        scores = []
        for chunk in chunks[:100]:  # Ensure exactly 100 scores
            score = chunk.get("score", 0.0)
            scores.append(score)

        # Pad with zeros if less than 100 results
        while len(scores) < 100:
            scores.append(0.0)

        # Get recall@100 from metrics
        recall_key = f"recall@{self.config.k}"
        recall = metrics.get(recall_key, 0.0)

        # Write one row: recall@100, score1, score2, ..., score100
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [f"{recall:.6f}"] + [f"{s:.6f}" for s in scores]
            writer.writerow(row)

    async def run_single_query(self, query: str, relevant_ids: List[str]) -> Dict[str, Any]:
        """
        Run evaluation for a single query.

        Args:
            query: The search query
            relevant_ids: List of ground truth relevant chunk IDs

        Returns:
            Dictionary with query results and metrics
        """
        try:
            # Execute search
            result = self.registry.execute_tool(
                "search",
                query=query,
                k=self.config.k,
                use_hyde=self.config.use_hyde,
                num_expands=self.config.num_expands,
                enable_graph_boost=self.config.enable_graph_boost,
                search_method=self.config.search_method,
            )

            if not result.success:
                return {
                    "query": query,
                    "success": False,
                    "error": result.error
                }

            # Extract retrieved chunk IDs and scores
            retrieved_ids = [c.get("chunk_id") or c.get("id") for c in result.data]
            retrieved_chunks = result.data  # Full chunk data with scores

            # Calculate metrics
            metrics = calculate_all_metrics(retrieved_ids, relevant_ids, self.config.k)

            # Save training data (scores + recall@100)
            self._save_training_data(retrieved_chunks, relevant_ids, metrics)

            return {
                "query": query,
                "success": True,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids[:self.config.k],
                "metrics": metrics,
            }

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            return {
                "query": query,
                "success": False,
                "error": str(e)
            }

    async def run_evaluation(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        Run evaluation on entire dataset.

        Args:
            dataset: List of query items with relevant chunk IDs

        Returns:
            Dictionary with all results and aggregate metrics
        """
        print(f"\nStarting evaluation: {self.config.experiment_name}")
        print(f"Configuration: k={self.config.k}, hyde={self.config.use_hyde}, "
              f"expands={self.config.num_expands}, graph={self.config.enable_graph_boost}, "
              f"method={self.config.search_method}\n")

        results = []
        successful = 0

        for i, item in enumerate(dataset, 1):
            query = item["query"]
            relevant_ids = item["relevant_chunk_ids"]

            print(f"[{i}/{len(dataset)}] {query[:60]}...")

            result = await self.run_single_query(query, relevant_ids)
            results.append(result)

            if result["success"]:
                successful += 1
                metrics = result["metrics"]
                print(f"  NDCG@{self.config.k}={metrics[f'ndcg@{self.config.k}']:.3f}, "
                      f"RR={metrics['reciprocal_rank']:.3f}, "
                      f"P@{self.config.k}={metrics[f'precision@{self.config.k}']:.3f}, "
                      f"R@{self.config.k}={metrics[f'recall@{self.config.k}']:.3f}")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown error')}")

        # Calculate aggregate metrics
        successful_results = [r for r in results if r.get("success")]

        if not successful_results:
            print("\nNo successful queries!")
            aggregate_metrics = {}
        else:
            aggregate_metrics = self._calculate_aggregate_metrics(successful_results)

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
                "successful_queries": successful,
                "failed_queries": len(dataset) - successful,
                "config": {
                    "k": self.config.k,
                    "use_hyde": self.config.use_hyde,
                    "num_expands": self.config.num_expands,
                    "enable_graph_boost": self.config.enable_graph_boost,
                    "search_method": self.config.search_method,
                    "experiment_name": self.config.experiment_name,
                }
            },
            "aggregate_metrics": aggregate_metrics,
            "results": results
        }

    def _calculate_aggregate_metrics(self, successful_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from successful results."""
        import numpy as np

        k = self.config.k
        metrics_keys = [
            f"ndcg@{k}",
            "reciprocal_rank",
            f"precision@{k}",
            f"recall@{k}",
            f"f1@{k}"
        ]

        aggregate = {}
        for key in metrics_keys:
            values = [r["metrics"][key] for r in successful_results if key in r["metrics"]]
            if values:
                aggregate[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        # Add MRR (Mean Reciprocal Rank) as average of reciprocal ranks
        rr_values = [r["metrics"]["reciprocal_rank"] for r in successful_results]
        if rr_values:
            aggregate["mrr"] = float(np.mean(rr_values))

        return aggregate

    def save_results(self, results: Dict[str, Any]):
        """
        Save evaluation results to JSON file.

        Args:
            results: Dictionary with all results and metrics
        """
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, aggregate_metrics: Dict[str, Any]):
        """
        Print evaluation summary.

        Args:
            aggregate_metrics: Dictionary with aggregate metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        if not aggregate_metrics:
            print("No metrics to display (all queries failed)")
            return

        k = self.config.k
        main_metrics = [
            (f"NDCG@{k}", f"ndcg@{k}"),
            ("MRR", "mrr"),
            (f"Precision@{k}", f"precision@{k}"),
            (f"Recall@{k}", f"recall@{k}"),
            (f"F1@{k}", f"f1@{k}"),
        ]

        for display_name, key in main_metrics:
            if key == "mrr":
                if key in aggregate_metrics:
                    print(f"{display_name:15} = {aggregate_metrics[key]:.4f}")
            elif key in aggregate_metrics:
                stats = aggregate_metrics[key]
                print(f"{display_name:15} = {stats['mean']:.4f} "
                      f"(±{stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f})")

        print("=" * 60)


async def run_standard_evaluation(
    experiment_name: str,
    k: int = 100,
    use_hyde: bool = False,
    num_expands: int = 0,
    enable_graph_boost: bool = False,
    search_method: str = "hybrid",
    dataset_path: str = None,
    output_path: str = None
) -> None:
    """
    Standard evaluation runner with common workflow.

    Args:
        experiment_name: Name for this experiment
        k: Number of top results to retrieve
        use_hyde: Whether to use HyDE
        num_expands: Number of query expansions
        enable_graph_boost: Whether to enable graph boosting
        search_method: Search method to use
        dataset_path: Optional custom dataset path
        output_path: Optional custom output path
    """
    # Create configuration
    config = EvaluationConfig(
        dataset_path=dataset_path,
        output_path=output_path,
        k=k,
        use_hyde=use_hyde,
        num_expands=num_expands,
        enable_graph_boost=enable_graph_boost,
        search_method=search_method,
        experiment_name=experiment_name
    )

    # Load dataset
    print(f"Loading dataset from {config.dataset_path}...")
    with open(config.dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} queries")

    # Create and initialize runner
    runner = EvaluationRunner(config)
    if not await runner.initialize():
        sys.exit(1)

    # Run evaluation
    results = await runner.run_evaluation(dataset)

    # Save results
    runner.save_results(results)

    # Print summary
    runner.print_summary(results["aggregate_metrics"])