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
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from itertools import product

sys.path.insert(0, '/app')
from src.multi_agent.runner import MultiAgentRunner
from src.agent.tools import get_registry

# Configuration
DATASET_PATH = "/app/benchmark_dataset/retrieval.json"
OUTPUT_DIR = "/app/results/grid_search_k100"
K = 100
ENABLE_GRAPH_BOOST = False  # Disable for clean comparison

# Grid search parameters
HYDE_VALUES = [True, False]
NUM_EXPANDS_VALUES = [0, 1, 2]
SEARCH_METHOD_VALUES = ['hybrid', 'dense_only', 'bm25_only']


def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevances = [1 if cid in relevant_ids else 0 for cid in retrieved_ids[:k]]
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k([1] * min(len(relevant_ids), k), k)
    return dcg / ideal if ideal > 0 else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    if len(relevant_ids) == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def get_config_name(use_hyde, num_expands, search_method):
    """Generate descriptive config name."""
    hyde_str = "hyde" if use_hyde else "nohyde"
    exp_str = f"exp{num_expands}"
    return f"{search_method}_{hyde_str}_{exp_str}"


async def evaluate_config(runner, registry, dataset, use_hyde, num_expands, search_method):
    """Evaluate single configuration."""
    config_name = get_config_name(use_hyde, num_expands, search_method)
    print(f"\n{'='*80}")
    print(f"CONFIG: {config_name}")
    print(f"  use_hyde={use_hyde}, num_expands={num_expands}, search_method={search_method}")
    print(f"{'='*80}\n")

    results = []
    successful = 0

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        print(f"[{i}/{len(dataset)}] {query[:60]}...")

        try:
            result = registry.execute_tool(
                "search",
                query=query,
                k=K,
                use_hyde=use_hyde,
                num_expands=num_expands,
                enable_graph_boost=ENABLE_GRAPH_BOOST,
                search_method=search_method,
            )

            if not result.success:
                print(f"  FAILED: {result.error}")
                results.append({"query": query, "success": False, "error": result.error})
                continue

            retrieved_ids = [c.get("chunk_id") or c.get("id") for c in result.data]

            ndcg = ndcg_at_k(retrieved_ids, relevant_ids, K)
            rr = reciprocal_rank(retrieved_ids, relevant_ids)
            prec = precision_at_k(retrieved_ids, relevant_ids, K)
            rec = recall_at_k(retrieved_ids, relevant_ids, K)

            print(f"  NDCG@{K}={ndcg:.3f}, RR={rr:.3f}, P@{K}={prec:.3f}, R@{K}={rec:.3f}")

            results.append({
                "query": query,
                "success": True,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids[:K],
                "metrics": {
                    f"ndcg@{K}": ndcg,
                    "reciprocal_rank": rr,
                    f"precision@{K}": prec,
                    f"recall@{K}": rec,
                },
            })
            successful += 1

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append({"query": query, "success": False, "error": str(e)})

    # Calculate aggregate metrics
    successful_results = [r for r in results if r.get("success")]

    if not successful_results:
        print(f"\n{config_name}: No successful queries!")
        agg = {}
    else:
        ndcgs = [r["metrics"][f"ndcg@{K}"] for r in successful_results]
        rrs = [r["metrics"]["reciprocal_rank"] for r in successful_results]
        precs = [r["metrics"][f"precision@{K}"] for r in successful_results]
        recs = [r["metrics"][f"recall@{K}"] for r in successful_results]

        agg = {
            f"mean_ndcg@{K}": float(np.mean(ndcgs)),
            "mean_reciprocal_rank": float(np.mean(rrs)),
            f"mean_precision@{K}": float(np.mean(precs)),
            f"mean_recall@{K}": float(np.mean(recs)),
            f"std_ndcg@{K}": float(np.std(ndcgs)),
            "std_reciprocal_rank": float(np.std(rrs)),
            f"min_ndcg@{K}": float(np.min(ndcgs)),
            f"max_ndcg@{K}": float(np.max(ndcgs)),
        }

    # Get reranker status
    search_tool = registry.get_tool("search")
    has_reranker = search_tool.reranker is not None if search_tool else False

    output = {
        "setup": {
            "dataset": DATASET_PATH,
            "k": K,
            "use_hyde": use_hyde,
            "num_expands": num_expands,
            "enable_graph_boost": ENABLE_GRAPH_BOOST,
            "search_method": search_method,
            "reranker_available": has_reranker,
            "multi_layer_retrieval": True,
            "config_name": config_name,
        },
        "results": results,
        "aggregate_metrics": agg,
        "success_rate": successful / len(dataset),
        "num_queries": len(dataset),
        "num_successful": successful,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_path = Path(OUTPUT_DIR) / f"eval_{config_name}_k{K}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"Summary: {successful}/{len(dataset)} successful")
    if agg:
        print(f"Mean NDCG@{K}: {agg[f'mean_ndcg@{K}']:.4f}")
        print(f"Mean Recall@{K}: {agg[f'mean_recall@{K}']:.4f}")

    return config_name, agg, successful / len(dataset)


async def main():
    print("="*80)
    print("GRID SEARCH EVALUATION")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"k: {K}")
    print(f"HyDE values: {HYDE_VALUES}")
    print(f"num_expands values: {NUM_EXPANDS_VALUES}")
    print(f"search_method values: {SEARCH_METHOD_VALUES}")
    print(f"Total configurations: {len(HYDE_VALUES) * len(NUM_EXPANDS_VALUES) * len(SEARCH_METHOD_VALUES)}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} queries")

    # Load configuration
    print("Loading configuration...")
    config_path = Path("/app/config.json")
    with open(config_path) as f:
        full_config = json.load(f)

    runner_config = {
        "api_keys": {
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
        },
        "vector_store_path": "/app/vector_db",
        "models": full_config.get("models", {}),
        "storage": full_config.get("storage", {}),
        "agent_tools": full_config.get("agent_tools", {}),
        "knowledge_graph": full_config.get("knowledge_graph", {}),
        "neo4j": full_config.get("neo4j", {}),
        "multi_agent": full_config.get("multi_agent", {}),
    }

    print("Initializing MultiAgentRunner...")
    runner = MultiAgentRunner(runner_config)
    success = await runner.initialize()

    if not success:
        print("ERROR: Runner initialization failed!")
        sys.exit(1)

    registry = get_registry()
    print(f"Tool registry has {len(registry._tools)} tools")

    # Run grid search
    all_results = []

    configs = list(product(HYDE_VALUES, NUM_EXPANDS_VALUES, SEARCH_METHOD_VALUES))
    total_configs = len(configs)

    for idx, (use_hyde, num_expands, search_method) in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {idx}/{total_configs}")
        print(f"{'#'*80}")

        config_name, agg, success_rate = await evaluate_config(
            runner, registry, dataset, use_hyde, num_expands, search_method
        )

        all_results.append({
            "config_name": config_name,
            "use_hyde": use_hyde,
            "num_expands": num_expands,
            "search_method": search_method,
            "aggregate_metrics": agg,
            "success_rate": success_rate,
        })

    # Save summary
    summary_path = Path(OUTPUT_DIR) / f"grid_search_summary_k{K}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "k": K,
                "hyde_values": HYDE_VALUES,
                "num_expands_values": NUM_EXPANDS_VALUES,
                "search_method_values": SEARCH_METHOD_VALUES,
                "enable_graph_boost": ENABLE_GRAPH_BOOST,
            },
            "total_configurations": total_configs,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total configurations tested: {total_configs}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")
    print("\nTop 5 configurations by mean NDCG@100:")

    # Sort by NDCG
    valid_results = [r for r in all_results if r["aggregate_metrics"]]
    sorted_results = sorted(
        valid_results,
        key=lambda x: x["aggregate_metrics"].get(f"mean_ndcg@{K}", 0),
        reverse=True
    )

    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['config_name']}: NDCG@{K}={result['aggregate_metrics'][f'mean_ndcg@{K}']:.4f}, "
              f"Recall@{K}={result['aggregate_metrics'][f'mean_recall@{K}']:.4f}")

    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
