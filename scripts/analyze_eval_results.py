#!/usr/bin/env python3
"""Analyze evaluation results from JSON files."""

import json
import sys
from pathlib import Path

def analyze_file(filepath):
    """Analyze a single evaluation file."""
    with open(filepath) as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"File: {filepath}")
    print(f"{'='*80}")

    # Top-level keys
    print(f"\nTop-level keys: {list(data.keys())}")

    # Results count
    results = data.get('results', [])
    print(f"Total queries: {len(results)}")

    if results:
        # Sample result structure
        print(f"\nFirst result keys: {list(results[0].keys())}")
        print(f"\nSample result:")
        print(json.dumps(results[0], indent=2)[:800])

        # Calculate metrics if possible
        if 'recall' in results[0]:
            recalls = [r.get('recall', 0) for r in results]
            precisions = [r.get('precision', 0) for r in results]
            mrrs = [r.get('mrr', 0) for r in results]

            print(f"\n--- Aggregate Metrics ---")
            print(f"Recall@10: {sum(recalls)/len(recalls):.4f}")
            print(f"Precision@10: {sum(precisions)/len(precisions):.4f}")
            print(f"MRR: {sum(mrrs)/len(mrrs):.4f}")

if __name__ == '__main__':
    files = [
        'results_all/eval_hyde_expand2_k10_FIXED.json',
        'results_all/eval_hyde_expand1_k10.json',
        'results_all/eval_expansion_only_k10.json',
    ]

    for f in files:
        if Path(f).exists():
            analyze_file(f)
        else:
            print(f"\nFile not found: {f}")
