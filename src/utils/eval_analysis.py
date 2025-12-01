"""
Analysis utilities for evaluation results.

This module provides reusable components for analyzing and visualizing
evaluation results from benchmark experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


def load_evaluation_results(filepath: Path) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.

    Args:
        filepath: Path to the results file

    Returns:
        Dictionary containing evaluation results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_grid_search_summary(summary_path: Path) -> Dict[str, Any]:
    """
    Load grid search summary file.

    Args:
        summary_path: Path to grid search summary JSON

    Returns:
        Dictionary containing grid search summary
    """
    return load_evaluation_results(summary_path)


def extract_metrics_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Extract metrics from results into a pandas DataFrame.

    Args:
        results: List of configuration results

    Returns:
        DataFrame with metrics for each configuration
    """
    rows = []
    for config in results:
        row = {
            "name": config["name"],
            "use_hyde": config.get("use_hyde", False),
            "num_expands": config.get("num_expands", 0),
            "search_method": config.get("search_method", "hybrid"),
            "success_rate": config.get("success_rate", 0.0)
        }

        # Extract metrics if available
        metrics = config.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                # Handle structured metrics with mean, std, min, max
                row[f"{metric_name}_mean"] = metric_data.get("mean", 0.0)
                row[f"{metric_name}_std"] = metric_data.get("std", 0.0)
            else:
                # Handle simple numeric metrics
                row[metric_name] = metric_data

        rows.append(row)

    return pd.DataFrame(rows)


def get_top_configurations(
    df: pd.DataFrame,
    metric: str = "ndcg@100_mean",
    n: int = 5
) -> pd.DataFrame:
    """
    Get top N configurations by a specific metric.

    Args:
        df: DataFrame with configuration results
        metric: Metric column to sort by
        n: Number of top configurations to return

    Returns:
        DataFrame with top N configurations
    """
    if metric not in df.columns:
        available = [col for col in df.columns if '_mean' in col or col == 'mrr']
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    return df.nlargest(n, metric)[
        ["name", metric, "use_hyde", "num_expands", "search_method", "success_rate"]
    ]


def analyze_by_parameter(
    df: pd.DataFrame,
    parameter: str,
    metric: str = "ndcg@100_mean"
) -> pd.DataFrame:
    """
    Analyze results grouped by a specific parameter.

    Args:
        df: DataFrame with configuration results
        parameter: Parameter to group by (e.g., 'use_hyde', 'num_expands', 'search_method')
        metric: Metric to analyze

    Returns:
        DataFrame with aggregated results by parameter
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter '{parameter}' not found in results")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")

    return df.groupby(parameter)[metric].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).round(4)


def find_best_configuration_per_method(
    df: pd.DataFrame,
    metric: str = "ndcg@100_mean"
) -> pd.DataFrame:
    """
    Find the best configuration for each search method.

    Args:
        df: DataFrame with configuration results
        metric: Metric to optimize for

    Returns:
        DataFrame with best configuration per search method
    """
    best_configs = []

    for method in df['search_method'].unique():
        method_df = df[df['search_method'] == method]
        best_idx = method_df[metric].idxmax()
        best_configs.append(method_df.loc[best_idx])

    result_df = pd.DataFrame(best_configs)
    return result_df.sort_values(metric, ascending=False)


def calculate_improvement_matrix(
    df: pd.DataFrame,
    baseline_config: Dict[str, Any],
    metric: str = "ndcg@100_mean"
) -> pd.DataFrame:
    """
    Calculate improvement matrix relative to a baseline configuration.

    Args:
        df: DataFrame with configuration results
        baseline_config: Dictionary specifying baseline configuration
        metric: Metric to calculate improvements for

    Returns:
        DataFrame with percentage improvements over baseline
    """
    # Find baseline
    baseline_df = df
    for key, value in baseline_config.items():
        baseline_df = baseline_df[baseline_df[key] == value]

    if baseline_df.empty:
        raise ValueError(f"Baseline configuration not found: {baseline_config}")

    baseline_value = baseline_df[metric].iloc[0]

    # Calculate improvements
    df['improvement_%'] = ((df[metric] - baseline_value) / baseline_value * 100).round(2)
    df['improvement_abs'] = (df[metric] - baseline_value).round(4)

    return df[["name", metric, "improvement_%", "improvement_abs"]].sort_values(
        "improvement_%", ascending=False
    )


def print_analysis_summary(df: pd.DataFrame, k: int = 100):
    """
    Print a comprehensive analysis summary.

    Args:
        df: DataFrame with configuration results
        k: Value of k parameter for metrics
    """
    print("=" * 80)
    print("EVALUATION ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\nDataset: {len(df)} configurations evaluated")
    print(f"Average success rate: {df['success_rate'].mean():.2%}")

    # Top configurations
    print(f"\n{'='*80}")
    print(f"TOP 5 CONFIGURATIONS BY NDCG@{k}")
    print(f"{'='*80}")
    top_configs = get_top_configurations(df, f"ndcg@{k}_mean", 5)
    for i, row in top_configs.iterrows():
        print(f"{row['name']:30} NDCG@{k}={row[f'ndcg@{k}_mean']:.4f}")
        print(f"  hyde={row['use_hyde']}, expands={row['num_expands']}, method={row['search_method']}")

    # Analysis by parameter
    parameters = ['use_hyde', 'num_expands', 'search_method']
    for param in parameters:
        print(f"\n{'='*80}")
        print(f"ANALYSIS BY {param.upper()}")
        print(f"{'='*80}")
        analysis = analyze_by_parameter(df, param, f"ndcg@{k}_mean")
        print(analysis)

    # Best per search method
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION PER SEARCH METHOD")
    print(f"{'='*80}")
    best_per_method = find_best_configuration_per_method(df, f"ndcg@{k}_mean")
    for i, row in best_per_method.iterrows():
        print(f"{row['search_method']:15} -> {row['name']:30} NDCG@{k}={row[f'ndcg@{k}_mean']:.4f}")


def compare_experiments(
    experiment_files: List[Tuple[str, Path]],
    metric: str = "ndcg@100"
) -> pd.DataFrame:
    """
    Compare results across multiple experiments.

    Args:
        experiment_files: List of (name, filepath) tuples
        metric: Metric to compare

    Returns:
        DataFrame comparing experiments
    """
    comparisons = []

    for exp_name, filepath in experiment_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        aggregate = data.get("aggregate_metrics", {})
        if metric in aggregate:
            if isinstance(aggregate[metric], dict):
                value = aggregate[metric].get("mean", 0.0)
                std = aggregate[metric].get("std", 0.0)
            else:
                value = aggregate[metric]
                std = 0.0
        else:
            value = 0.0
            std = 0.0

        comparisons.append({
            "experiment": exp_name,
            "file": filepath.name,
            f"{metric}_mean": value,
            f"{metric}_std": std,
            "success_rate": data.get("success_rate", 0.0),
            "num_queries": data.get("num_queries", 0)
        })

    return pd.DataFrame(comparisons).sort_values(f"{metric}_mean", ascending=False)