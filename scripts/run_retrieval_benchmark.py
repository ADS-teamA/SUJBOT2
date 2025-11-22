"""
Top-k retrieval benchmark for the RAG pipeline.

Evaluates whether the RAG retriever returns the ground-truth chunk
for synthetic questions generated from an indexed document.

Logic:
- For each question in the dataset JSON:
  - Embed the question using the configured embedding model.
  - Query the hybrid vector store (FAISS + BM25) for the top-k chunks.
  - Count the query as correct if the ground_truth_chunk_id is present
    in the returned top-k layer-3 chunks.

Defaults:
- Dataset: benchmark_dataset/synthetic_questions_bz_vr.json
- Vector store: vector_db
- k (top-k): 5

This script is read-only with respect to the vector store. It only reads
embeddings and metadata; no indexing or mutation is performed.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.hybrid_search import HybridVectorStore
from src.embedding_generator import EmbeddingGenerator
from src.cost_tracker import get_global_tracker, reset_global_tracker


logger = logging.getLogger(__name__)


@dataclass
class RetrievalExample:
    """Single retrieval example from the synthetic dataset."""

    question: str
    ground_truth_chunk_id: str
    ground_truth_document_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of a single retrieval evaluation."""

    index: int
    example: RetrievalExample
    retrieved_chunk_ids: List[str]
    retrieved_document_ids: List[Optional[str]]
    hit: bool
    hit_rank: Optional[int]
    latency_ms: float


def load_dataset(path: Path, max_queries: Optional[int] = None) -> List[RetrievalExample]:
    """Load synthetic questions dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must be a list of objects")

    examples: List[RetrievalExample] = []
    for item in raw:
        # Be defensive against missing keys
        question = item.get("question")
        chunk_id = item.get("ground_truth_chunk_id")

        if not question or not chunk_id:
            logger.warning("Skipping invalid entry without question or ground_truth_chunk_id")
            continue

        examples.append(
            RetrievalExample(
                question=question,
                ground_truth_chunk_id=chunk_id,
                ground_truth_document_id=item.get("ground_truth_document_id"),
            )
        )

    if max_queries is not None:
        examples = examples[: max(0, max_queries)]

    return examples


def run_retrieval_benchmark(
    dataset_path: Path,
    vector_store_path: Path,
    k: int = 5,
    max_queries: Optional[int] = None,
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Run retrieval@k benchmark over the synthetic questions dataset.

    Returns:
        Dict with aggregate statistics and per-query results (lightweight).
    """
    logger.info("Loading dataset...")
    examples = load_dataset(dataset_path, max_queries=max_queries)

    if not examples:
        raise ValueError("No valid examples found in dataset")

    logger.info(f"Loaded {len(examples)} examples from {dataset_path}")

    # Reset cost tracker so retrieval benchmark cost is isolated
    reset_global_tracker()

    logger.info(f"Loading hybrid vector store from {vector_store_path}...")
    store = HybridVectorStore.load(vector_store_path)

    # Initialize embedder (uses config.json to pick provider/model)
    embedder = EmbeddingGenerator()

    # Sanity check: embedding dimensionality must match store
    store_dims = store.faiss_store.dimensions
    if getattr(embedder, "dimensions", None) != store_dims:
        raise ValueError(
            f"Embedding model dimension {getattr(embedder, 'dimensions', None)} "
            f"does not match vector store dimension {store_dims}. "
            "Ensure config.json embedding_model matches the model used to build the store."
        )

    logger.info(
        f"Embedder and store ready: dim={store_dims}, "
        f"top-k={k}, total_examples={len(examples)}"
    )

    results: List[RetrievalResult] = []
    hits = 0

    progress = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc="Evaluating retrieval",
        unit="query",
    )

    start_time = time.time()

    for idx, example in progress:
        t0 = time.time()

        # Embed question
        embedding = embedder.embed_texts([example.question])
        # embed_texts returns (1, dim); pass 1D vector to retrieval
        query_embedding = embedding[0]

        # Hybrid hierarchical search (RAG pipeline retrieval)
        search_result = store.hierarchical_search(
            query_text=example.question,
            query_embedding=query_embedding,
            k_layer3=k,
        )

        layer3 = search_result.get("layer3", []) or []

        # Extract chunk/document IDs for top-k results
        retrieved_chunk_ids: List[str] = []
        retrieved_document_ids: List[Optional[str]] = []

        for r in layer3[:k]:
            retrieved_chunk_ids.append(r.get("chunk_id"))
            retrieved_document_ids.append(r.get("document_id"))

        latency_ms = (time.time() - t0) * 1000.0

        # Hit logic: ground truth chunk appears anywhere in top-k
        hit_rank: Optional[int] = None
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid == example.ground_truth_chunk_id:
                hit_rank = rank
                break

        hit = hit_rank is not None
        if hit:
            hits += 1

        # Record result
        results.append(
            RetrievalResult(
                index=idx,
                example=example,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_document_ids=retrieved_document_ids,
                hit=hit,
                hit_rank=hit_rank,
                latency_ms=latency_ms,
            )
        )

        # Update progress bar with running accuracy
        accuracy = hits / idx
        progress.set_postfix({"acc@k": f"{accuracy:.3f}"})

    total_time = time.time() - start_time
    total_queries = len(results)
    accuracy_at_k = hits / total_queries if total_queries > 0 else 0.0
    avg_latency_ms = (
        sum(r.latency_ms for r in results) / total_queries if total_queries > 0 else 0.0
    )

    # Cost (embeddings + any retrieval-related API usage)
    tracker = get_global_tracker()
    total_cost_usd = tracker.get_total_cost()

    logger.info("")
    logger.info("=" * 80)
    logger.info("RETRIEVAL BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Dataset:        {dataset_path}")
    logger.info(f"Vector store:   {vector_store_path}")
    logger.info(f"Top-k (layer3): {k}")
    logger.info(f"Total queries:  {total_queries}")
    logger.info(f"Total time:     {total_time:.1f}s")
    logger.info(f"Accuracy@{k}:   {accuracy_at_k:.4f} ({hits}/{total_queries})")
    logger.info(f"Avg latency:    {avg_latency_ms:.1f} ms/query")
    logger.info(f"Total cost:     ${total_cost_usd:.4f}")
    logger.info("=" * 80)

    # Build lightweight summary for optional serialization
    summary: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "vector_store_path": str(vector_store_path),
        "k": k,
        "total_queries": total_queries,
        "hits": hits,
        "accuracy_at_k": accuracy_at_k,
        "total_time_seconds": total_time,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": total_cost_usd,
        "per_query": [
            {
                "index": r.index,
                "question": r.example.question,
                "ground_truth_chunk_id": r.example.ground_truth_chunk_id,
                "ground_truth_document_id": r.example.ground_truth_document_id,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "retrieved_document_ids": r.retrieved_document_ids,
                "hit": r.hit,
                "hit_rank": r.hit_rank,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    }

    # Attach LLM agent setup metadata (for comparing different configurations)
    if agent_config is not None:
        summary["agent_config"] = agent_config

    return summary


async def generate_multi_agent_logs(
    dataset_path: Path,
    max_queries: Optional[int],
    config_path: Path,
    vector_store_path: Path,
    orchestrator_model: Optional[str],
    default_agent_model: Optional[str],
    per_agent_models: Dict[str, Optional[str]],
    log_file: Path,
) -> None:
    """
    Run the multi-agent system on the same questions and write all events to a JSONL log.

    This does NOT affect retrieval metrics; it is purely for observability:
    - tools used
    - agent start/complete events
    - final answers and metadata
    """
    if not config_path.exists():
        logger.error(f"Multi-agent config not found at {config_path}, skipping multi-agent logs")
        return

    # Load root config.json
    with config_path.open("r", encoding="utf-8") as f:
        root_config = json.load(f)

    multi_agent_cfg = root_config.get("multi_agent", {})
    if not multi_agent_cfg or not multi_agent_cfg.get("enabled", False):
        logger.error("multi_agent section missing or disabled in config, skipping multi-agent logs")
        return

    # Override orchestrator model if requested
    if orchestrator_model:
        multi_agent_cfg.setdefault("orchestrator", {})
        multi_agent_cfg["orchestrator"]["model"] = orchestrator_model

    # Override models for individual agents if requested
    agents_cfg = multi_agent_cfg.setdefault("agents", {})
    for agent_name, override in per_agent_models.items():
        if override:
            agent_cfg = agents_cfg.setdefault(agent_name, {})
            agent_cfg["model"] = override

    # Attach updated multi_agent config back to root config
    root_config["multi_agent"] = multi_agent_cfg

    # For FAISS-based tools, ensure vector_store_path is set (no-op for PostgreSQL backend)
    root_config["vector_store_path"] = str(vector_store_path)

    # Reset cost tracker for isolated multi-agent cost
    reset_global_tracker()

    # Import MultiAgentRunner lazily so retrieval benchmark can run
    # even if multi-agent dependencies (e.g., PostgreSQL) are missing.
    try:
        from src.multi_agent.runner import MultiAgentRunner  # type: ignore
    except Exception as e:
        logger.error(
            f"Multi-agent runner could not be imported ({e}). "
            f"Install PostgreSQL/psycopg or disable multi-agent logging."
        )
        return

    # Initialize multi-agent runner
    runner = MultiAgentRunner(root_config)
    ok = await runner.initialize()
    if not ok:
        logger.error("Failed to initialize multi-agent system; skipping multi-agent logs")
        return

    # Load questions (same as retrieval benchmark)
    examples = load_dataset(dataset_path, max_queries=max_queries)
    if not examples:
        logger.error("No examples available for multi-agent logging")
        return

    logger.info(f"Generating multi-agent logs for {len(examples)} queries -> {log_file}")

    with log_file.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(examples, start=1):
            question = example.question
            logger.info(f"[multi-agent log] Running query {idx}/{len(examples)}")

            async for event in runner.run_query(question, stream_progress=True):
                record = {
                    "query_index": idx,
                    "query": question,
                    "event": event,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Cost summary for multi-agent run
    tracker = get_global_tracker()
    ma_total_cost_usd = tracker.get_total_cost()

    runner.shutdown()
    logger.info(f"Multi-agent logs written to {log_file}")
    logger.info(f"Multi-agent total cost: ${ma_total_cost_usd:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the retrieval benchmark."""
    parser = argparse.ArgumentParser(
        description="Run top-k retrieval benchmark on synthetic RAG questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_dataset/synthetic_questions_bz_vr.json",
        help="Path to synthetic questions JSON file",
    )

    parser.add_argument(
        "--vector-store",
        type=str,
        default="vector_db",
        help="Path to hybrid vector store directory",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top chunks to consider for a hit",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit the number of queries (for quick tests)",
    )

    # --- LLM agent setup metadata (used for labeling runs) ---
    parser.add_argument(
        "--agent-setup",
        type=str,
        default=None,
        help="Short label for the LLM agent configuration (e.g., 'baseline', 'orchestrator_v2')",
    )

    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="LLM model used by the agent/orchestrator for QA runs (metadata only)",
    )

    parser.add_argument(
        "--orchestrator-prompt-id",
        type=str,
        default=None,
        help="Identifier of the orchestrator prompt variant used (metadata only)",
    )

    parser.add_argument(
        "--agent-notes",
        type=str,
        default=None,
        help="Free-form notes describing this agent setup (metadata only)",
    )

    # Per-agent model overrides for multi-agent system (optional)
    parser.add_argument(
        "--orchestrator-model",
        type=str,
        default=None,
        help="Override model for orchestrator agent (multi-agent run; default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default=None,
        help="Override model for 'extractor' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="Override model for 'classifier' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--compliance-model",
        type=str,
        default=None,
        help="Override model for 'compliance' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--risk-verifier-model",
        type=str,
        default=None,
        help="Override model for 'risk_verifier' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--citation-auditor-model",
        type=str,
        default=None,
        help="Override model for 'citation_auditor' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--gap-synthesizer-model",
        type=str,
        default=None,
        help="Override model for 'gap_synthesizer' agent (default: use config.json or --agent-model)",
    )
    parser.add_argument(
        "--requirement-extractor-model",
        type=str,
        default=None,
        help="Override model for 'requirement_extractor' agent (default: use config.json or --agent-model)",
    )

    # Multi-agent logging configuration
    parser.add_argument(
        "--multi-agent-log-file",
        type=str,
        default=None,
        help=(
            "Optional path to write multi-agent event logs as JSONL "
            "(tools used, agent events, final answers). "
            "If not set, multi-agent is not invoked."
        ),
    )

    parser.add_argument(
        "--multi-agent-config",
        type=str,
        default="config.json",
        help="Path to root multi-agent config JSON (default: config.json)",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional path to save JSON summary "
            "(e.g., benchmark_results/retrieval_bz_vr_top3.json)"
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_path = Path(args.dataset)
    vector_store_path = Path(args.vector_store)

    # Resolve per-agent model overrides (metadata + optional multi-agent run)
    default_agent_model = args.agent_model

    # Orchestrator model: explicit flag wins, else default_agent_model, else None (use config.json)
    orchestrator_model = args.orchestrator_model or default_agent_model

    per_agent_models: Dict[str, Optional[str]] = {
        "extractor": args.extractor_model or default_agent_model,
        "classifier": args.classifier_model or default_agent_model,
        "compliance": args.compliance_model or default_agent_model,
        "risk_verifier": args.risk_verifier_model or default_agent_model,
        "citation_auditor": args.citation_auditor_model or default_agent_model,
        "gap_synthesizer": args.gap_synthesizer_model or default_agent_model,
        "requirement_extractor": args.requirement_extractor_model or default_agent_model,
    }

    agent_config: Dict[str, Any] = {
        "setup": args.agent_setup,
        "default_agent_model": default_agent_model,
        "orchestrator": {
            "model": orchestrator_model,
        },
        "agents": per_agent_models,
        "orchestrator_prompt_id": args.orchestrator_prompt_id,
        "notes": args.agent_notes,
        # Task-specific benchmark prompt (metadata only, does not affect retrieval logic).
        # This can be reused as part of the RAG system prompt when running generative QA
        # on this benchmark.
        "benchmark_prompt": (
            "You are evaluating retrieval on synthetic Czech questions derived from the "
            "VR-1 research reactor safety report. All questions are intended to be answered "
            "from the VR-1 safety report document with document_id 'BZ_VR1'. When you run "
            "a RAG QA pipeline on these questions, instruct the system to restrict or "
            "strongly prioritize retrieval to document_id 'BZ_VR1' unless the query "
            "explicitly asks for comparison with other laws or documents."
        ),
    }

    logger.info("=" * 80)
    logger.info("RETRIEVAL@K BENCHMARK (SYNTHETIC QUESTIONS)")
    logger.info("=" * 80)
    logger.info(f"Dataset:      {dataset_path}")
    logger.info(f"Vector store: {vector_store_path}")
    logger.info(f"k (top-k):    {args.k}")
    logger.info(f"Max queries:  {args.max_queries or 'all'}")
    logger.info(f"Agent setup:  {args.agent_setup or 'n/a'}")
    logger.info(f"Default model:{default_agent_model or 'n/a'}")
    logger.info(f"Orch. model:  {orchestrator_model or 'config.json'}")
    logger.info(f"Orch. prompt: {args.orchestrator_prompt_id or 'n/a'}")
    logger.info("=" * 80)

    summary = run_retrieval_benchmark(
        dataset_path=dataset_path,
        vector_store_path=vector_store_path,
        k=args.k,
        max_queries=args.max_queries,
        agent_config=agent_config,
    )

    # Optional JSON export
    if args.output_json:
        base_path = Path(args.output_json)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if base_path.suffix.lower() == ".json":
            stem = base_path.stem
            parent = base_path.parent
        else:
            # Treat value as directory or prefix
            stem = base_path.name or "retrieval_benchmark"
            parent = base_path if base_path.suffix == "" else base_path.parent

        filename = f"{stem}_{timestamp}.json"
        output_path = parent / filename
        parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON summary to {output_path}")

    # Optional multi-agent log generation (separate from retrieval metrics)
    if args.multi_agent_log_file:
        base_log_path = Path(args.multi_agent_log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Preserve extension if provided, default to .jsonl
        if base_log_path.suffix:
            ext = base_log_path.suffix
            stem = base_log_path.stem
            parent = base_log_path.parent
        else:
            ext = ".jsonl"
            stem = base_log_path.name or "multiagent_logs"
            parent = base_log_path

        log_filename = f"{stem}_{timestamp}{ext}"
        log_path = parent / log_filename
        parent.mkdir(parents=True, exist_ok=True)

        try:
            asyncio.run(
                generate_multi_agent_logs(
                    dataset_path=dataset_path,
                    max_queries=args.max_queries,
                    config_path=Path(args.multi_agent_config),
                    vector_store_path=vector_store_path,
                    orchestrator_model=orchestrator_model,
                    default_agent_model=default_agent_model,
                    per_agent_models=per_agent_models,
                    log_file=log_path,
                )
            )
        except Exception as e:
            logger.error(f"Failed to generate multi-agent logs: {e}", exc_info=True)


if __name__ == "__main__":
    main()
