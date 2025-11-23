#!/usr/bin/env python3
"""
Analyze and compare evaluation results from multiple experiments.

Usage:
    python analyze_eval_results.py [file1.json] [file2.json] ...

    If no files specified, analyzes all results in the results directory.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, '/app')
from src.utils.eval_analysis import (
    load_evaluation_results,
    compare_experiments,
    extract_metrics_dataframe
)
from src.utils.eval_config import EvalPaths
import pandas as pd


def analyze_single_file(filepath: Path):
    """
    Analyze a single evaluation result file.

    Args:
        filepath: Path to the evaluation result file
    """
    print(f"\n{'='*80}")
    print(f"File: {filepath.name}")
    print(f"{'='*80}")

    try:
        data = load_evaluation_results(filepath)

        # Extract metadata
        metadata = data.get('metadata', data.get('setup', {}))
        aggregate = data.get('aggregate_metrics', {})

        print("\nConfiguration:")
        if 'config' in metadata:
            config = metadata['config']
            print(f"  Experiment: {config.get('experiment_name', 'Unknown')}")
            print(f"  k: {config.get('k', 'N/A')}")
            print(f"  HyDE: {config.get('use_hyde', False)}")
            print(f"  Expansions: {config.get('num_expands', 0)}")
            print(f"  Graph Boost: {config.get('enable_graph_boost', False)}")
            print(f"  Search Method: {config.get('search_method', 'hybrid')}")
        else:
            # Legacy format
            print(f"  k: {metadata.get('k', 'N/A')}")
            print(f"  HyDE: {metadata.get('use_hyde', 'N/A')}")
            print(f"  Expansions: {metadata.get('num_expands', 'N/A')}")
            print(f"  Search Method: {metadata.get('search_method', 'N/A')}")

        print(f"\nDataset:")
        print(f"  Total queries: {metadata.get('dataset_size', data.get('num_queries', 'N/A'))}")
        print(f"  Successful: {metadata.get('successful_queries', data.get('num_successful', 'N/A'))}")
        print(f"  Failed: {metadata.get('failed_queries', len(data.get('results', [])) - metadata.get('successful_queries', 0))}")

        if aggregate:
            print("\nMetrics:")
            for metric_name, metric_value in aggregate.items():
                if isinstance(metric_value, dict):
                    mean = metric_value.get('mean', 0)
                    std = metric_value.get('std', 0)
                    print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")
                elif metric_name == 'mrr' or isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.4f}")

        # Success rate
        success_rate = data.get('success_rate')
        if success_rate is not None:
            print(f"\nSuccess Rate: {success_rate:.2%}")

        return data

    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None


def find_result_files() -> List[Path]:
    """Find all evaluation result JSON files in the results directory."""
    results_dir = EvalPaths.RESULTS_DIR
    if not results_dir.exists():
        return []

    # Find all JSON files that look like evaluation results
    json_files = list(results_dir.glob("eval_*.json"))
    json_files.extend(results_dir.glob("**/eval_*.json"))

    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return json_files


def main():
    """Main analysis function."""
    print("=" * 80)
    print("EVALUATION RESULTS ANALYSIS")
    print("=" * 80)

    # Determine which files to analyze
    if len(sys.argv) > 1:
        # Use provided files
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # Find all evaluation files
        files = find_result_files()[:10]  # Limit to 10 most recent

    if not files:
        print("\nNo evaluation result files found.")
        print(f"Expected location: {EvalPaths.RESULTS_DIR}/eval_*.json")
        sys.exit(1)

    print(f"\nAnalyzing {len(files)} result file(s)...")

    # Analyze each file and collect for comparison
    experiments = []
    for filepath in files:
        if not filepath.exists():
            print(f"\nWARNING: File not found: {filepath}")
            continue

        data = analyze_single_file(filepath)
        if data:
            # Extract experiment name
            metadata = data.get('metadata', data.get('setup', {}))
            if 'config' in metadata:
                exp_name = metadata['config'].get('experiment_name', filepath.stem)
            else:
                exp_name = filepath.stem

            experiments.append((exp_name, filepath))

    # Compare experiments if multiple files
    if len(experiments) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON ACROSS EXPERIMENTS")
        print(f"{'='*80}\n")

        # Compare key metrics
        for metric in ['ndcg@100', 'mrr', 'precision@100', 'recall@100']:
            print(f"\n--- {metric.upper()} Comparison ---")
            comparison_df = compare_experiments(experiments, metric)
            if not comparison_df.empty:
                print(comparison_df.to_string(index=False))

        # Rank experiments by NDCG@100
        print(f"\n{'='*80}")
        print("EXPERIMENT RANKING (by NDCG@100)")
        print(f"{'='*80}")

        ranking_df = compare_experiments(experiments, 'ndcg@100')
        if not ranking_df.empty:
            for i, row in ranking_df.iterrows():
                print(f"\n{i+1}. {row['experiment']}")
                print(f"   NDCG@100: {row['ndcg@100_mean']:.4f} ± {row['ndcg@100_std']:.4f}")
                print(f"   Success Rate: {row['success_rate']:.2%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()