#!/usr/bin/env python3
"""
Multi-Agent Benchmark CLI - Evaluates multi-agent system on QA datasets.

Usage:
    # Run full benchmark
    uv run python run_benchmark.py

    # Quick test (10 queries)
    uv run python run_benchmark.py --max-queries 10

    # With custom model
    uv run python run_benchmark.py --model claude-sonnet-4-5

    # Debug mode
    uv run python run_benchmark.py --debug --max-queries 3

Outputs:
    - benchmark_results/MULTI-AGENT_YYYYMMDD_HHMMSS.json
    - benchmark_results/MULTI-AGENT_YYYYMMDD_HHMMSS.md
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  uv run python run_benchmark.py

  # Quick test with 10 queries
  uv run python run_benchmark.py --max-queries 10

  # Debug mode with verbose logging
  uv run python run_benchmark.py --debug --max-queries 3

  # Use Sonnet model instead of Haiku
  uv run python run_benchmark.py --model claude-sonnet-4-5

  # Custom dataset and vector store
  uv run python run_benchmark.py \\
    --dataset my_dataset/qa.json \\
    --vector-store my_dataset_db
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_dataset/privacy_qa.json",
        help="Path to QA dataset JSON (default: benchmark_dataset/privacy_qa.json)",
    )

    parser.add_argument(
        "--vector-store",
        type=str,
        default="benchmark_db",
        help="Path to indexed vector store (default: benchmark_db)",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (default: all queries in dataset)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Model to use for evaluation (default: claude-haiku-4-5)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic evaluation)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )

    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking (default: enabled)",
    )

    parser.add_argument(
        "--no-caching",
        action="store_true",
        help="Disable prompt caching (default: enabled)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save per-query JSONs, verbose logging)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )

    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.0,
        help="Delay between queries in seconds (for API rate limits, default: 0.0)",
    )

    return parser.parse_args()


async def main():
    """Main benchmark execution."""
    args = parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        # Import here to show clearer error if dependencies missing
        from src.benchmark.config import BenchmarkConfig
        from src.benchmark.multi_agent_runner import MultiAgentBenchmarkRunner
        from src.benchmark.report import save_benchmark_report

        logger.info("=" * 80)
        logger.info("MULTI-AGENT BENCHMARK EVALUATION")
        logger.info("=" * 80)

        # Create configuration
        config = BenchmarkConfig(
            dataset_path=args.dataset,
            vector_store_path=args.vector_store,
            max_queries=args.max_queries,
            agent_model=args.model,
            agent_temperature=args.temperature,
            k=args.k,
            enable_reranking=not args.no_reranking,
            enable_prompt_caching=not args.no_caching,
            debug_mode=args.debug,
            output_dir=args.output_dir,
            rate_limit_delay=args.rate_limit_delay,
        )

        logger.info(f"Configuration:")
        logger.info(f"  Dataset: {config.dataset_path}")
        logger.info(f"  Vector Store: {config.vector_store_path}")
        logger.info(f"  Model: {config.agent_model}")
        logger.info(f"  Temperature: {config.agent_temperature}")
        logger.info(f"  Retrieval k: {config.k}")
        logger.info(f"  Reranking: {'enabled' if config.enable_reranking else 'disabled'}")
        logger.info(f"  Prompt Caching: {'enabled' if config.enable_prompt_caching else 'disabled'}")
        logger.info(f"  Max Queries: {config.max_queries if config.max_queries else 'all'}")
        logger.info("")

        # Create runner
        runner = MultiAgentBenchmarkRunner(config)

        # Run evaluation
        result = await runner.run()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.upper().replace("-", "_").replace(".", "_")
        output_prefix = f"MULTI-AGENT_{model_name}_{timestamp}"

        save_benchmark_report(
            result=result,
            output_dir=Path(args.output_dir),
            output_prefix=output_prefix,
            save_markdown=True,
            save_json=True,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved:")
        logger.info(f"  - {args.output_dir}/{output_prefix}.json")
        logger.info(f"  - {args.output_dir}/{output_prefix}.md")
        logger.info("")
        logger.info(f"Final Metrics:")
        logger.info(f"  Exact Match: {result.aggregate_metrics.get('exact_match', 0):.3f}")
        logger.info(f"  F1 Score: {result.aggregate_metrics.get('f1_score', 0):.3f}")
        logger.info(f"  Precision: {result.aggregate_metrics.get('precision', 0):.3f}")
        logger.info(f"  Recall: {result.aggregate_metrics.get('recall', 0):.3f}")
        logger.info("")
        logger.info(f"Performance:")
        logger.info(f"  Total Time: {result.total_time_seconds:.1f}s")
        logger.info(f"  Avg Time/Query: {(result.total_time_seconds * 1000 / result.total_queries):.0f}ms")
        logger.info(f"  Total Cost: ${result.total_cost_usd:.4f}")
        logger.info(f"  Avg Cost/Query: ${(result.total_cost_usd / result.total_queries):.6f}")

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        sys.exit(130)

    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.error("\nPlease ensure:")
        logger.error("  1. Dataset exists (or use --dataset flag)")
        logger.error("  2. Vector store is indexed (or use --vector-store flag)")
        logger.error("\nRun 'uv run python scripts/index_benchmark_docs.py' to index documents first.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
