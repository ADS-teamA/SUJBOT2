"""
Run benchmark evaluation.

Evaluates RAG pipeline on PrivacyQA benchmark (194 QA pairs).

Usage:
    # Full benchmark (all queries)
    uv run python scripts/run_benchmark.py

    # Quick test (first 5 queries)
    uv run python scripts/run_benchmark.py --max-queries 5

    # Debug mode (verbose output + per-query JSON)
    uv run python scripts/run_benchmark.py --debug
"""

import argparse
import logging
import sys
from pathlib import Path

from src.benchmark import BenchmarkConfig, BenchmarkRunner
from src.benchmark.report import generate_reports
from src.benchmark.metrics import METRIC_ABBREVIATIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_summary(result, config: BenchmarkConfig) -> None:
    """Print benchmark result summary."""
    # Define metric display labels
    metric_labels = {
        "exact_match": "Exact Match (EM)",
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "embedding_similarity": "Embedding Similarity",
        "combined_f1": "Combined F1",
        "rag_confidence": "RAG Confidence",
    }

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Dataset:       {result.dataset_name}")
    print(f"Total Queries: {result.total_queries}")
    print(f"Total Time:    {result.total_time_seconds:.1f}s")
    print(f"Total Cost:    ${result.total_cost_usd:.4f}")
    print("\nAggregate Metrics:")

    for metric_name, score in sorted(result.aggregate_metrics.items()):
        label = metric_labels.get(metric_name, metric_name.replace("_", " ").title())
        print(f"  {label:25s}: {score:.4f}")

    print("\nReports saved to:")
    print(f"  {Path(config.output_dir).absolute()}")
    print("=" * 80 + "\n")


def print_configuration(config: BenchmarkConfig) -> None:
    """Print benchmark configuration in a clean format."""
    config_items = [
        ("Dataset", config.dataset_path),
        ("Vector Store", config.vector_store_path),
        ("Max Queries", config.max_queries if config.max_queries else "all (194)"),
        ("Retrieval k", config.k),
        ("Reranking", "enabled" if config.enable_reranking else "disabled"),
        ("Graph Boost", "enabled" if config.enable_graph_boost else "disabled"),
        ("Agent Model", config.agent_model),
        ("Output Dir", config.output_dir),
        ("Debug Mode", config.debug_mode),
        ("Fail Fast", config.fail_fast),
    ]

    logger.info("\nConfiguration:")
    for label, value in config_items:
        logger.info(f"  {label}: {value}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark evaluation on PrivacyQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark (all 194 queries)
  uv run python scripts/run_benchmark.py

  # Quick test (first 5 queries)
  uv run python scripts/run_benchmark.py --max-queries 5

  # Debug mode (verbose + per-query JSONs)
  uv run python scripts/run_benchmark.py --debug

  # Custom vector store path
  uv run python scripts/run_benchmark.py --vector-store my_benchmark_db
        """,
    )

    parser.add_argument(
        "--dataset",
        default="benchmark_dataset/privacy_qa.json",
        help="Path to QA JSON file (default: benchmark_dataset/privacy_qa.json)",
    )

    parser.add_argument(
        "--vector-store",
        default="benchmark_db",
        help="Path to indexed vector store (default: benchmark_db)",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        help="Limit number of queries for testing (default: all queries)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )

    parser.add_argument(
        "--no-reranking", action="store_true", help="Disable cross-encoder reranking"
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for reports (default: benchmark_results)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (verbose + per-query JSON)"
    )

    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue on agent errors (default: fail on first error)",
    )

    parser.add_argument(
        "--agent-model",
        default="claude-haiku-4-5",
        help="Agent model to use (default: claude-haiku-4-5). "
        "Supported: claude-haiku-4-5, claude-sonnet-4-5, gpt-5-mini, gpt-5-nano, "
        "gemini-2.5-flash, gemini-2.5-pro",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("BENCHMARK EVALUATION - PRIVACY QA")
    logger.info("=" * 80)

    # Create config
    try:
        config = BenchmarkConfig.from_env(
            dataset_path=args.dataset,
            vector_store_path=args.vector_store,
            max_queries=args.max_queries,
            k=args.k,
            enable_reranking=not args.no_reranking,
            output_dir=args.output_dir,
            debug_mode=args.debug,
            fail_fast=not args.no_fail_fast,
            agent_model=args.agent_model,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Print configuration
    print_configuration(config)

    # Initialize runner
    try:
        logger.info("\nInitializing benchmark runner...")
        runner = BenchmarkRunner(config)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        logger.info("\nStarting evaluation...\n")
        result = runner.run()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)

    # Generate reports
    try:
        logger.info("\nGenerating reports...")
        report_paths = generate_reports(
            result,
            output_dir=config.output_dir,
            save_markdown=config.save_markdown,
            save_json=config.save_json,
            save_per_query=config.save_per_query,
        )

        logger.info("\nReports generated:")
        for report_type, path in report_paths.items():
            logger.info(f"  {report_type}: {path}")

    except Exception as e:
        logger.error(f"Failed to generate reports: {e}")
        sys.exit(1)

    # Print summary
    print_summary(result, config)

    logger.info("Benchmark evaluation complete!")


if __name__ == "__main__":
    main()
