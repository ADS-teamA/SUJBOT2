"""
Hybrid search tool benchmark (no agent orchestration).

- Uses SearchTool directly to retrieve top-k chunks
- Evaluates with the same metrics as the LegalBench-RAG benchmark
- Works with synthetic paraphrase datasets (question → ground_truth_chunk_id)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

# Add project root to import path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock heavy optional dependencies so imports succeed in minimal environments
MODULES_TO_MOCK = [
    "unstructured",
    "unstructured.partition",
    "unstructured.partition.auto",
    "unstructured.partition.pdf",
    "unstructured.partition.pptx",
    "unstructured.partition.docx",
    "unstructured.partition.image",
    "unstructured.partition.html",
    "unstructured.partition.xml",
    "unstructured.partition.md",
    "unstructured.partition.text",
    "unstructured.documents",
    "unstructured.documents.elements",
]
for module_name in MODULES_TO_MOCK:
    sys.modules[module_name] = MagicMock()

# Optional: load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.hybrid_search import HybridVectorStore
from src.embedding_generator import EmbeddingGenerator
from src.agent.config import AgentConfig
from src.agent.tools.tier1_basic import SearchTool
from src.benchmark.metrics import aggregate_metrics, compute_all_metrics
from src.cost_tracker import get_global_tracker, reset_global_tracker

logger = logging.getLogger(__name__)


def load_dataset(path: Path, max_queries: Optional[int] = None) -> List[Dict]:
    """Load synthetic paraphrase dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must contain a list of objects.")

    examples: List[Dict] = []
    for item in raw:
        query = item.get("question")
        chunk_id = item.get("ground_truth_chunk_id")
        doc_id = item.get("ground_truth_document_id")

        if not query or not chunk_id:
            logger.warning("Skipping entry missing query or ground_truth_chunk_id.")
            continue

        examples.append(
            {
                "query": query,
                "ground_truth_chunk_id": chunk_id,
                "ground_truth_document_id": doc_id,
            }
        )

    if max_queries is not None:
        examples = examples[: max(0, max_queries)]

    return examples


def initialize_components(vector_store_path: Path, enable_rerank: bool):
    """Load vector store, embedder, reranker (optional), and search tool."""
    logger.info("Loading hybrid vector store from %s", vector_store_path)
    vector_store = HybridVectorStore.load(vector_store_path)

    agent_config = AgentConfig.from_env(vector_store_path=vector_store_path)

    embedder = EmbeddingGenerator()
    store_dims = vector_store.faiss_store.dimensions
    if getattr(embedder, "dimensions", None) != store_dims:
        raise ValueError(
            f"Embedding dimension mismatch: embedder={getattr(embedder, 'dimensions', None)}, "
            f"store={store_dims}. Ensure config embedding model matches the indexed store."
        )

    reranker = None
    reranker_active = False
    if enable_rerank:
        try:
            from src.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name=agent_config.tool_config.reranker_model)
            reranker_active = True
        except Exception as e:
            logger.warning("Reranker unavailable, continuing without reranking: %s", e)

    search_tool = SearchTool(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        graph_retriever=None,
        knowledge_graph=None,
        context_assembler=None,
        config=agent_config.tool_config,
    )

    return search_tool, vector_store, reranker_active


def evaluate_query(
    search_tool: SearchTool,
    vector_store: HybridVectorStore,
    example: Dict,
    k: int,
    num_expands: int,
    enable_graph_boost: bool,
) -> Dict:
    """Run a single query through the hybrid search tool and compute metrics."""
    tracker = get_global_tracker()
    cost_before = tracker.get_total_cost()
    start_time = time.time()

    result = search_tool.execute(
        query=example["query"],
        k=k,
        num_expands=num_expands,
        enable_graph_boost=enable_graph_boost,
    )

    latency_ms = (time.time() - start_time) * 1000.0
    cost_delta = tracker.get_total_cost() - cost_before

    retrieved = result.data or []
    retrieved_chunk_ids = [c.get("chunk_id") for c in retrieved if isinstance(c, dict)]
    retrieved_document_ids = [c.get("document_id") for c in retrieved if isinstance(c, dict)]

    predicted_chunk = retrieved[0] if retrieved else {}
    predicted_text = predicted_chunk.get("content", "") if isinstance(predicted_chunk, dict) else ""
    predicted_chunk_id = predicted_chunk.get("chunk_id") if isinstance(predicted_chunk, dict) else None

    ground_truth_chunk_id = example["ground_truth_chunk_id"]
    ground_truth_chunk = vector_store.get_chunk_by_id(ground_truth_chunk_id) or {}
    ground_truth_text = ground_truth_chunk.get("content", "")

    hit_rank = None
    for idx, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid == ground_truth_chunk_id:
            hit_rank = idx
            break
    hit = hit_rank is not None

    metrics = compute_all_metrics(predicted_text, [ground_truth_text]) if ground_truth_text else {}

    rag_confidence = None
    if isinstance(result.metadata, dict):
        rag_meta = result.metadata.get("rag_confidence")
        if isinstance(rag_meta, dict):
            rag_confidence = rag_meta.get("overall_confidence")

    return {
        "query": example["query"],
        "ground_truth_chunk_id": ground_truth_chunk_id,
        "ground_truth_document_id": example.get("ground_truth_document_id"),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_document_ids": retrieved_document_ids,
        "predicted_chunk_id": predicted_chunk_id,
        "hit": hit,
        "hit_rank": hit_rank,
        "metrics": metrics,
        "rag_confidence": rag_confidence,
        "latency_ms": latency_ms,
        "cost_usd": max(0.0, cost_delta),
        "predicted_content_preview": predicted_text[:200] + ("..." if len(predicted_text) > 200 else ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the hybrid search tool (no agent).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_dataset/synthetic_paraphrases_bz_vr.json",
        help="Path to paraphrase dataset JSON.",
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="vector_db",
        help="Path to indexed hybrid store (default: vector_db)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries for quick runs.",
    )
    parser.add_argument(
        "--num-expands",
        type=int,
        default=0,
        help="Query expansions for SearchTool (default: 0, disables LLM paraphrasing).",
    )
    parser.add_argument(
        "--graph-boost",
        action="store_true",
        help="Enable graph boosting in SearchTool if graph retriever is available.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranker even if configured.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to store benchmark JSON (default: benchmark_results).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_path = Path(args.dataset)
    vector_store_path = Path(args.vector_store)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_dataset(dataset_path, max_queries=args.max_queries)
    if not examples:
        raise RuntimeError("No valid examples to evaluate.")

    logger.info("Loaded %d queries from %s", len(examples), dataset_path)

    search_tool, vector_store, reranker_active = initialize_components(
        vector_store_path, enable_rerank=not args.no_rerank
    )

    logger.info(
        "Hybrid search ready (k=%d, num_expands=%d, graph_boost=%s, rerank=%s)",
        args.k,
        args.num_expands,
        args.graph_boost,
        reranker_active,
    )

    reset_global_tracker()
    tracker = get_global_tracker()

    per_query_results: List[Dict] = []
    metric_results: List[Dict] = []
    hits = 0

    start_time = time.time()
    for idx, example in enumerate(examples, start=1):
        logger.info("Evaluating query %d/%d", idx, len(examples))
        result = evaluate_query(
            search_tool,
            vector_store,
            example,
            k=args.k,
            num_expands=args.num_expands,
            enable_graph_boost=args.graph_boost,
        )

        per_query_results.append(result)
        if result["metrics"]:
            metric_results.append(result["metrics"])

        if result["hit"]:
            hits += 1

    total_time_seconds = time.time() - start_time

    aggregated = aggregate_metrics(metric_results) if metric_results else {}

    hit_rate = hits / len(per_query_results) if per_query_results else 0.0
    avg_latency_ms = (
        total_time_seconds * 1000 / len(per_query_results) if per_query_results else 0.0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = dataset_path.stem.upper()
    output_path = output_dir / f"HYBRID_SEARCH_{dataset_tag}_K{args.k}_{timestamp}.json"

    summary = {
        "benchmark": "hybrid_search_tool",
        "dataset_path": str(dataset_path),
        "vector_store_path": str(vector_store_path),
        "k": args.k,
        "num_expands": args.num_expands,
        "graph_boost": args.graph_boost,
        "reranking": reranker_active,
        "total_queries": len(per_query_results),
        "hits": hits,
        "hit_rate": hit_rate,
        "aggregate_metrics": aggregated,
        "avg_latency_ms": avg_latency_ms,
        "total_time_seconds": total_time_seconds,
        "total_cost_usd": tracker.get_total_cost(),
        "per_query": per_query_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Benchmark complete. Results saved to %s", output_path)
    logger.info(
        "EM: %.3f | F1: %.3f | P: %.3f | R: %.3f | Hit@%d: %.3f | Avg Latency: %.0f ms | Cost: $%.4f",
        aggregated.get("exact_match", 0.0),
        aggregated.get("f1_score", 0.0),
        aggregated.get("precision", 0.0),
        aggregated.get("recall", 0.0),
        args.k,
        hit_rate,
        avg_latency_ms,
        tracker.get_total_cost(),
    )


if __name__ == "__main__":
    main()
