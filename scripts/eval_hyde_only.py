#!/usr/bin/env python3
"""HyDE only (no expansion)"""
import asyncio
import json
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, '/app')
from src.multi_agent.runner import MultiAgentRunner
from src.agent.tools import get_registry

DATASET_PATH = "/app/benchmark_dataset/retrieval.json"
OUTPUT_PATH = "/app/results/eval_hyde_only_k100.json"
K = 100
USE_HYDE = True
NUM_EXPANDS = 0
ENABLE_GRAPH_BOOST = False
SEARCH_METHOD = "hybrid"

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

async def main():
    print("Loading dataset...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} queries")

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

    print("\nStarting evaluation (HyDE only, no expansion)...\n")
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
                use_hyde=USE_HYDE,
                num_expands=NUM_EXPANDS,
                enable_graph_boost=ENABLE_GRAPH_BOOST,
                search_method=SEARCH_METHOD,
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

    successful_results = [r for r in results if r.get("success")]

    if not successful_results:
        print("\nNo successful queries!")
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

    search_tool = registry.get_tool("search")
    has_reranker = search_tool.reranker is not None if search_tool else False
    vector_store_has_multi_layer = True

    output = {
        "setup": {
            "dataset": DATASET_PATH,
            "k": K,
            "use_hyde": USE_HYDE,
            "num_expands": NUM_EXPANDS,
            "enable_graph_boost": ENABLE_GRAPH_BOOST,
            "search_method": SEARCH_METHOD,
            "reranker_available": has_reranker,
            "multi_layer_retrieval": vector_store_has_multi_layer,
        },
        "results": results,
        "aggregate_metrics": agg,
        "success_rate": successful / len(dataset),
        "num_queries": len(dataset),
        "num_successful": successful,
        "timestamp": datetime.now().isoformat(),
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {OUTPUT_PATH}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (HyDE only)")
    print("=" * 80)
    print(f"Queries: {len(dataset)} ({successful} successful)")
    print(f"Success Rate: {successful/len(dataset):.2%}")
    print(f"\nSetup:")
    print(f"  k={K}")
    print(f"  use_hyde={USE_HYDE}")
    print(f"  num_expands={NUM_EXPANDS}")
    print(f"\nAggregate Metrics:")
    for metric, value in agg.items():
        if metric.startswith('mean_'):
            print(f"  {metric}: {value:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
