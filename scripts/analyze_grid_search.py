#!/usr/bin/env python3
"""
Grid Search Results Analyzer

Analyzuje výsledky grid search a vytváří přehledné reporty.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_summary(summary_path: str) -> Dict:
    """Load grid search summary."""
    with open(summary_path) as f:
        return json.load(f)

def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create comparison table from results."""
    rows = []

    for r in results:
        if not r.get("aggregate_metrics"):
            continue

        metrics = r["aggregate_metrics"]
        rows.append({
            "Config": r["config_name"],
            "HyDE": "✓" if r["use_hyde"] else "✗",
            "Expands": r["num_expands"],
            "Method": r["search_method"],
            "NDCG@100": metrics.get("mean_ndcg@100", 0),
            "MRR": metrics.get("mean_reciprocal_rank", 0),
            "Precision@100": metrics.get("mean_precision@100", 0),
            "Recall@100": metrics.get("mean_recall@100", 0),
            "Success Rate": r["success_rate"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("NDCG@100", ascending=False)

def print_top_configs(df: pd.DataFrame, n: int = 5):
    """Print top N configurations."""
    print(f"\n{'='*80}")
    print(f"TOP {n} CONFIGURATIONS BY NDCG@100")
    print(f"{'='*80}\n")

    for i, row in df.head(n).iterrows():
        print(f"{i+1}. {row['Config']}")
        print(f"   HyDE={row['HyDE']}, Expands={row['Expands']}, Method={row['Method']}")
        print(f"   NDCG@100={row['NDCG@100']:.4f}, MRR={row['MRR']:.4f}, "
              f"P@100={row['Precision@100']:.4f}, R@100={row['Recall@100']:.4f}")
        print()

def analyze_by_dimension(df: pd.DataFrame, dimension: str, metric: str = "NDCG@100"):
    """Analyze results by specific dimension."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS BY {dimension.upper()}")
    print(f"{'='*80}\n")

    grouped = df.groupby(dimension)[metric].agg(['mean', 'std', 'min', 'max'])
    print(grouped.to_string())
    print()

def main():
    if len(sys.argv) < 2:
        summary_path = "results/grid_search_k100/grid_search_summary_k100.json"
    else:
        summary_path = sys.argv[1]

    if not Path(summary_path).exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        print("Usage: python analyze_grid_search.py [path/to/summary.json]")
        sys.exit(1)

    print("="*80)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("="*80)

    # Load data
    summary = load_summary(summary_path)
    print(f"\nSummary file: {summary_path}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total configurations: {summary['total_configurations']}")

    # Create comparison table
    df = create_comparison_table(summary['results'])

    # Print top configs
    print_top_configs(df, n=5)

    # Analyze by dimensions
    analyze_by_dimension(df, "HyDE", "NDCG@100")
    analyze_by_dimension(df, "Expands", "NDCG@100")
    analyze_by_dimension(df, "Method", "NDCG@100")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")
    print(f"Mean NDCG@100: {df['NDCG@100'].mean():.4f} ± {df['NDCG@100'].std():.4f}")
    print(f"Mean MRR: {df['MRR'].mean():.4f} ± {df['MRR'].std():.4f}")
    print(f"Mean Recall@100: {df['Recall@100'].mean():.4f} ± {df['Recall@100'].std():.4f}")
    print(f"\nBest NDCG@100: {df['NDCG@100'].max():.4f}")
    print(f"Worst NDCG@100: {df['NDCG@100'].min():.4f}")
    print(f"Range: {df['NDCG@100'].max() - df['NDCG@100'].min():.4f}")

    # Save detailed CSV
    csv_path = Path(summary_path).parent / "grid_search_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed comparison saved to: {csv_path}")

    print("="*80)

if __name__ == "__main__":
    main()
