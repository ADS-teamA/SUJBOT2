#!/usr/bin/env python3
"""
Grid Search Results Analyzer

Analyzes grid search results and creates comprehensive reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/app')
from src.utils.eval_analysis import (
    load_grid_search_summary,
    extract_metrics_dataframe,
    print_analysis_summary,
    get_top_configurations,
    analyze_by_parameter,
    find_best_configuration_per_method
)
from src.utils.eval_config import EvalPaths


def main():
    """Analyze grid search results."""
    # Determine summary path
    if len(sys.argv) < 2:
        summary_path = EvalPaths.GRID_SEARCH_RESULTS / "grid_search_summary.json"
    else:
        summary_path = Path(sys.argv[1])

    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        print("Usage: python analyze_grid_search.py [path/to/summary.json]")
        sys.exit(1)

    print("=" * 80)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("=" * 80)

    # Load summary
    summary = load_grid_search_summary(summary_path)
    print(f"\nSummary file: {summary_path}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Dataset size: {summary['dataset_size']}")
    print(f"Total configurations: {len(summary['configurations'])}")

    # Extract metrics to DataFrame
    df = extract_metrics_dataframe(summary['configurations'])

    # Print comprehensive analysis
    print_analysis_summary(df, k=summary['k'])

    # Additional analyses
    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    # Effect of HyDE
    print("\n--- HyDE Effect ---")
    hyde_analysis = analyze_by_parameter(df, 'use_hyde', f"ndcg@{summary['k']}_mean")
    print(hyde_analysis)

    # Effect of expansions
    print("\n--- Query Expansion Effect ---")
    expand_analysis = analyze_by_parameter(df, 'num_expands', f"ndcg@{summary['k']}_mean")
    print(expand_analysis)

    # Effect of search method
    print("\n--- Search Method Effect ---")
    method_analysis = analyze_by_parameter(df, 'search_method', f"ndcg@{summary['k']}_mean")
    print(method_analysis)

    # Find best configuration per search method
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATIONS PER SEARCH METHOD")
    print("=" * 80)
    best_per_method = find_best_configuration_per_method(df, f"ndcg@{summary['k']}_mean")
    for _, row in best_per_method.iterrows():
        print(f"\n{row['search_method'].upper()}:")
        print(f"  Configuration: {row['name']}")
        print(f"  HyDE: {row['use_hyde']}, Expands: {row['num_expands']}")
        print(f"  NDCG@{summary['k']}: {row[f'ndcg@{summary["k"]}_mean']:.4f}")

    # Save detailed CSV
    csv_path = summary_path.parent / "grid_search_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed analysis saved to: {csv_path}")

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    top_config = df.nlargest(1, f"ndcg@{summary['k']}_mean").iloc[0]
    print(f"\n1. Best overall configuration: {top_config['name']}")
    print(f"   - HyDE: {'Enabled' if top_config['use_hyde'] else 'Disabled'}")
    print(f"   - Query Expansions: {top_config['num_expands']}")
    print(f"   - Search Method: {top_config['search_method']}")
    print(f"   - NDCG@{summary['k']}: {top_config[f'ndcg@{summary["k"]}_mean']:.4f}")

    # Analyze if HyDE helps
    hyde_grouped = df.groupby('use_hyde')[f"ndcg@{summary['k']}_mean"].mean()
    if hyde_grouped[True] > hyde_grouped[False]:
        improvement = (hyde_grouped[True] - hyde_grouped[False]) / hyde_grouped[False] * 100
        print(f"\n2. HyDE improves NDCG by {improvement:.1f}% on average - RECOMMEND ENABLING")
    else:
        decline = (hyde_grouped[False] - hyde_grouped[True]) / hyde_grouped[True] * 100
        print(f"\n2. HyDE reduces NDCG by {decline:.1f}% on average - RECOMMEND DISABLING")

    # Optimal number of expansions
    expand_grouped = df.groupby('num_expands')[f"ndcg@{summary['k']}_mean"].mean()
    best_expands = expand_grouped.idxmax()
    print(f"\n3. Optimal number of query expansions: {best_expands}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()