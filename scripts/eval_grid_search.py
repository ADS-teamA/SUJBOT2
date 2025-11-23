#!/usr/bin/env python3
"""
Grid Search Evaluation - Systematic Testing of RAG Parameters

Tests all combinations of:
- hyde: [True, False]
- num_expands: [0, 1, 2]
- search_method: ['hybrid', 'dense_only', 'bm25_only']
- k: 100 (fixed)
- multi_layer: True (fixed)

Total: 2 × 3 × 3 = 18 configurations
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, '/app')
from src.utils.eval_runner import EvaluationRunner, EvaluationConfig
from src.utils.eval_metrics import calculate_all_metrics


# Grid search parameters
HYDE_VALUES = [True, False]
NUM_EXPANDS_VALUES = [0, 1, 2]
SEARCH_METHOD_VALUES = ['hybrid', 'dense_only', 'bm25_only']
K = 100
DATASET_PATH = "/app/benchmark_dataset/retrieval.json"
OUTPUT_DIR = "/app/results/grid_search_k100"


def get_config_name(use_hyde, num_expands, search_method):
    """Generate descriptive config name."""
    hyde_str = "hyde" if use_hyde else "nohyde"
    exp_str = f"exp{num_expands}"
    return f"{search_method}_{hyde_str}_{exp_str}"


async def main():
    """Run grid search evaluation."""
    print("=" * 80)
    print("GRID SEARCH EVALUATION")
    print("=" * 80)

    # Calculate total configurations
    total_configs = len(HYDE_VALUES) * len(NUM_EXPANDS_VALUES) * len(SEARCH_METHOD_VALUES)
    print(f"Total configurations to test: {total_configs}")
    print(f"Parameters:")
    print(f"  hyde: {HYDE_VALUES}")
    print(f"  num_expands: {NUM_EXPANDS_VALUES}")
    print(f"  search_method: {SEARCH_METHOD_VALUES}")
    print(f"  k: {K}")
    print("=" * 80)

    # Load dataset once
    print(f"\nLoading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} queries")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary results for all configurations
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "k": K,
        "configurations": []
    }

    config_num = 0

    # Test each configuration
    for use_hyde, num_expands, search_method in product(
        HYDE_VALUES, NUM_EXPANDS_VALUES, SEARCH_METHOD_VALUES
    ):
        config_num += 1
        config_name = get_config_name(use_hyde, num_expands, search_method)

        print(f"\n{'='*80}")
        print(f"[{config_num}/{total_configs}] CONFIG: {config_name}")
        print(f"  use_hyde={use_hyde}, num_expands={num_expands}, search_method={search_method}")
        print(f"{'='*80}\n")

        # Create configuration
        config = EvaluationConfig(
            dataset_path=DATASET_PATH,
            output_path=str(output_dir / f"{config_name}_k{K}.json"),
            k=K,
            use_hyde=use_hyde,
            num_expands=num_expands,
            enable_graph_boost=False,  # Disable for clean comparison
            search_method=search_method,
            experiment_name=config_name
        )

        # Create and initialize runner for this configuration
        runner = EvaluationRunner(config)

        # Initialize only once for first config, reuse for others
        if config_num == 1:
            if not await runner.initialize():
                print("ERROR: Runner initialization failed!")
                sys.exit(1)
            # Save registry for reuse
            registry = runner.registry
            multi_runner = runner.runner
        else:
            # Reuse existing runner and registry
            runner.runner = multi_runner
            runner.registry = registry

        # Run evaluation
        results = await runner.run_evaluation(dataset)

        # Save individual results
        runner.save_results(results)

        # Add to summary
        config_summary = {
            "name": config_name,
            "use_hyde": use_hyde,
            "num_expands": num_expands,
            "search_method": search_method,
            "success_rate": results["metadata"]["successful_queries"] / len(dataset),
            "metrics": results["aggregate_metrics"]
        }
        summary["configurations"].append(config_summary)

        # Print config summary
        runner.print_summary(results["aggregate_metrics"])

    # Save summary file
    summary_path = output_dir / "grid_search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")

    # Print top configurations
    print_top_configs(summary)


def print_top_configs(summary, n=5):
    """Print top N configurations by NDCG@100."""
    configs = summary["configurations"]
    k = summary["k"]

    # Sort by NDCG@100
    configs_with_ndcg = []
    for cfg in configs:
        if cfg["metrics"] and f"ndcg@{k}" in cfg["metrics"]:
            ndcg = cfg["metrics"][f"ndcg@{k}"]["mean"]
            configs_with_ndcg.append((cfg, ndcg))

    configs_with_ndcg.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(n, len(configs_with_ndcg))} configurations by NDCG@{k}:")
    print("-" * 70)

    for i, (cfg, ndcg) in enumerate(configs_with_ndcg[:n], 1):
        print(f"{i}. {cfg['name']:30} NDCG@{k}={ndcg:.4f}")
        print(f"   hyde={cfg['use_hyde']}, expands={cfg['num_expands']}, method={cfg['search_method']}")


if __name__ == "__main__":
    asyncio.run(main())