"""
Criteria QA benchmark for Czech nuclear safety criteria.

This script evaluates the FULL multi-agent RAG pipeline on a binary
YES/NO classification task defined in benchmark_dataset/criteria_qa_cs.json.

For each entry in the dataset:
  - Build a Czech query that includes:
      * The original criterion text (original_text_cs)
      * A concrete yes/no question (question_cs)
      * Explicit instructions to:
          - Search the indexed VR-1 safety report (BZ_VR1) via RAG
          - Decide whether the criterion is clearly satisfied
          - Output a final label line: "FINAL_LABEL: YES" or "FINAL_LABEL: NO"
  - Run the full multi-agent pipeline (orchestrator + agents + tools)
  - Extract the predicted label from the final answer
  - Compare it to ground_truth_label ("YES"/"NO") for accuracy

The CLI is designed to mirror the retrieval benchmark
(`scripts/run_retrieval_benchmark.py`) so you can:
  - Switch orchestrator / agent models
  - Point to different multi-agent config files
  - Control retrieval depth (k) via agent_tools.default_k
  - Optionally emit detailed multi-agent logs (JSONL)
"""

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from dotenv import load_dotenv

from src.cost_tracker import get_global_tracker, reset_global_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class CriteriaQAExample:
    """Single criteria QA example."""

    id: int
    criterion_id: int
    original_text_cs: str
    question_cs: str
    ground_truth_label: str  # "YES" / "NO"


def load_criteria_qa_dataset(
    path: Path, max_queries: Optional[int] = None
) -> List[CriteriaQAExample]:
    """
    Load criteria QA dataset from JSON.

    Expected format: list of objects with keys:
      - id
      - criterion_id
      - original_text_cs
      - question_cs
      - ground_truth_label ("YES"/"NO")
    """
    if not path.exists():
        raise FileNotFoundError(f"Criteria QA dataset not found at {path}")

    # Use utf-8-sig to handle potential BOM in JSON file
    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("criteria_qa_cs.json must be a JSON list of objects")

    examples: List[CriteriaQAExample] = []
    for item in raw:
        try:
            ex = CriteriaQAExample(
                id=int(item["id"]),
                criterion_id=int(item["criterion_id"]),
                original_text_cs=str(item["original_text_cs"]),
                question_cs=str(item["question_cs"]),
                ground_truth_label=str(item["ground_truth_label"]).strip().upper(),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid criteria QA entry: {e} | item={item}")
            continue

        if ex.ground_truth_label not in {"YES", "NO"}:
            logger.warning(
                f"Skipping entry id={ex.id}: unsupported ground_truth_label={ex.ground_truth_label!r}"
            )
            continue

        examples.append(ex)

    if max_queries is not None:
        examples = examples[: max(0, max_queries)]

    if not examples:
        raise ValueError("No valid Criteria QA examples loaded from dataset")

    logger.info(f"Loaded {len(examples)} criteria QA examples from {path}")
    return examples


# ---------------------------------------------------------------------------
# Benchmark-specific prompt & query construction
# ---------------------------------------------------------------------------


# High-level benchmark description (for metadata + reuse in system prompts if needed).
CRITERIA_QA_BENCHMARK_PROMPT = (
    "You are evaluating a binary YES/NO classification task on Czech nuclear "
    "safety criteria. Each example provides:\n"
    "- The original regulatory criterion text in Czech (original_text_cs)\n"
    "- A concrete audit question in Czech (question_cs) derived from that criterion\n\n"
    "Your goal is to use the full RAG pipeline (retrieval, analysis, and "
    "multi-agent reasoning) over the VR-1 research reactor safety report "
    "(document_id 'BZ_VR1') and decide whether the available documentation "
    "clearly and sufficiently demonstrates that the criterion is satisfied.\n\n"
    "Label semantics:\n"
    "- YES = The safety report contains clear, specific, and sufficient evidence "
    "that the criterion is fulfilled.\n"
    "- NO  = The safety report lacks such evidence, is vague, or indicates "
    "that the criterion is not fulfilled.\n\n"
    "For benchmarking, the run is considered correct if the final predicted "
    "label exactly matches the ground_truth_label ('YES' or 'NO')."
)


def build_query_for_example(example: CriteriaQAExample) -> str:
    """
    Build a Czech query that forces the multi-agent system to:
      - Use RAG over VR-1 BZ documentation
      - Decide YES/NO for the criterion
      - Emit a machine-readable FINAL_LABEL line.

    Important: per user request, do NOT include the original criterion text
    in the query—only the question and task prompt.
    """
    return f"""Jsi specializovaný hodnotící agent pro bezpečnostní dokumentaci jaderného zařízení.

Níže je pouze kontrolní otázka (v češtině), která se vztahuje k hodnoticímu kritériu.

Tvoje úloha:
1. Pomocí interních nástrojů prohledat indexovanou BEZPEČNOSTNÍ ZPRÁVU (BZ) výzkumného reaktoru VR-1
   (dokument s document_id 'BZ_VR1') a najít všechny relevantní části.
2. Na základě nalezených informací rozhodnout, zda BZ dané kritérium JASNĚ a DOSTATEČNĚ pokrývá.
3. Výsledek vyjádři jako binární štítek YES / NO podle těchto pravidel:
   - YES = BZ jednoznačně obsahuje konkrétní informace, ze kterých plyne, že kritérium je splněno.
   - NO  = BZ takové informace neobsahuje, jsou jen velmi obecné / nepřímé, nebo naznačují nesplnění.
   - Pokud si nejsi jistý a důkaz je slabý nebo chybí, zvol raději NO.

Odpověď:
- Stručně vysvětli své uvažování v češtině.
- Na úplně POSLEDNÍ řádek odpovědi napiš přesně:
  FINAL_LABEL: YES
  nebo
  FINAL_LABEL: NO
- Na tomto řádku nepřidávej žádný další text ani vysvětlení.

=== Kontrolní otázka k BZ ===
{example.question_cs}
"""


def select_balanced_examples(
    examples: List[CriteriaQAExample], per_label: int = 5
) -> List[CriteriaQAExample]:
    """
    Select a balanced subset: per_label examples for YES and NO.

    Raises:
        ValueError if insufficient examples for either label.
    """
    yes_examples = [e for e in examples if e.ground_truth_label == "YES"]
    no_examples = [e for e in examples if e.ground_truth_label == "NO"]

    if len(yes_examples) < per_label or len(no_examples) < per_label:
        raise ValueError(
            f"Not enough examples to build a balanced set "
            f"({per_label} YES and {per_label} NO). "
            f"Available: YES={len(yes_examples)}, NO={len(no_examples)}"
        )

    # Deterministic: take the first N of each
    balanced = yes_examples[:per_label] + no_examples[:per_label]
    return balanced


def extract_final_label(final_answer: str) -> Optional[str]:
    """
    Extract FINAL_LABEL: YES/NO from agent's final answer.

    Returns:
        "YES", "NO", or None if not found.
    """
    if not final_answer:
        return None

    # Search explicitly for FINAL_LABEL: YES/NO (case-insensitive, ignore spaces)
    match = re.search(r"FINAL_LABEL:\s*(YES|NO)\b", final_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look at the last non-empty line and try to parse it
    lines = [ln.strip() for ln in final_answer.splitlines() if ln.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    match = re.search(r"(YES|NO)\b", last_line, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Core benchmark runner (multi-agent, full pipeline)
# ---------------------------------------------------------------------------


async def run_criteria_qa_benchmark(
    dataset_path: Path,
    max_queries: Optional[int],
    config_path: Path,
    vector_store_path: Path,
    orchestrator_model: Optional[str],
    default_agent_model: Optional[str],
    per_agent_models: Dict[str, Optional[str]],
    retrieval_k: Optional[int],
    log_file: Optional[Path],
    agent_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run Criteria QA benchmark with the full multi-agent pipeline.

    Returns:
        Dict with aggregate statistics and per-query results.
    """
    # Load multi-agent root config
    if not config_path.exists():
        raise FileNotFoundError(f"Multi-agent config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        root_config = json.load(f)

    multi_agent_cfg = root_config.get("multi_agent", {})
    if not multi_agent_cfg or not multi_agent_cfg.get("enabled", False):
        raise ValueError(
            "multi_agent section missing or disabled in config. "
            "Enable it in config.json before running this benchmark."
        )

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

    # Override retrieval depth (default_k) via agent_tools if specified
    if retrieval_k is not None:
        agent_tools_cfg = root_config.setdefault("agent_tools", {})
        agent_tools_cfg["default_k"] = retrieval_k

    # For local benchmarks we often have FAISS vector_db but no PostgreSQL.
    # Force FAISS backend for tools, using the provided vector_store_path.
    storage_cfg = root_config.setdefault("storage", {})
    storage_cfg["backend"] = "faiss"

    # Attach updated multi_agent config back to root config
    root_config["multi_agent"] = multi_agent_cfg

    # For FAISS-based tools, ensure vector_store_path is set (no-op for PostgreSQL backend)
    root_config["vector_store_path"] = str(vector_store_path)

    # Reset cost tracker so benchmark cost is isolated
    reset_global_tracker()

    # Lazy import MultiAgentRunner to avoid hard dependency if not installed
    try:
        from src.multi_agent.runner import MultiAgentRunner  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Multi-agent runner could not be imported ({e}). "
            f"Install multi-agent dependencies or adjust config."
        ) from e

    # Initialize multi-agent runner
    runner = MultiAgentRunner(root_config)
    ok = await runner.initialize()
    if not ok:
        runner.shutdown()
        raise RuntimeError("Failed to initialize multi-agent system")

    # Load dataset
    examples = load_criteria_qa_dataset(dataset_path, max_queries=max_queries)
    # Enforce balanced subset: 5 YES, 5 NO (total 10)
    examples = select_balanced_examples(examples, per_label=5)

    # Optional logging of all events (JSONL)
    log_fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_file.open("w", encoding="utf-8")

    results: List[Dict[str, Any]] = []
    correct = 0
    total_latency_ms = 0.0
    total_cost_usd = 0.0

    progress = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc="Evaluating criteria QA",
        unit="query",
    )

    start_time = time.time()

    try:
        for idx, example in progress:
            query_text = build_query_for_example(example)

            t0 = time.time()
            final_event: Optional[Dict[str, Any]] = None

            async for event in runner.run_query(query_text, stream_progress=log_fh is not None):
                # Optional raw event logging (for debugging / analysis)
                if log_fh is not None:
                    record = {
                        "query_index": idx,
                        "example_id": example.id,
                        "criterion_id": example.criterion_id,
                        "query": query_text,
                        "event": event,
                    }
                    log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                if event.get("type") == "final":
                    final_event = event

            latency_ms = (time.time() - t0) * 1000.0
            total_latency_ms += latency_ms

            if not final_event:
                logger.error(f"No final event returned for example id={example.id}")
                predicted_label = None
                cost_usd = 0.0
                success = False
                raw_answer = ""
            else:
                raw_answer = final_event.get("final_answer", "") or ""
                predicted_label = extract_final_label(raw_answer)
                cost_cents = float(final_event.get("total_cost_cents", 0.0) or 0.0)
                cost_usd = cost_cents / 100.0
                total_cost_usd += cost_usd
                success = bool(final_event.get("success", False))

            gt_label = example.ground_truth_label
            is_correct = predicted_label is not None and predicted_label == gt_label
            if is_correct:
                correct += 1

            accuracy_so_far = correct / idx
            progress.set_postfix({"acc": f"{accuracy_so_far:.3f}"})

            results.append(
                {
                    "index": idx,
                    "id": example.id,
                    "criterion_id": example.criterion_id,
                    "question_cs": example.question_cs,
                    "original_text_cs": example.original_text_cs,
                    "ground_truth_label": gt_label,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "success": success,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "final_answer": raw_answer,
                }
            )

    finally:
        total_time = time.time() - start_time

        # Close log file if open
        if log_fh is not None:
            log_fh.close()

        # Shutdown multi-agent system
        runner.shutdown()

    total_queries = len(results)
    accuracy = correct / total_queries if total_queries > 0 else 0.0
    avg_latency_ms = total_latency_ms / total_queries if total_queries > 0 else 0.0

    # Cross-check cost with global tracker
    tracker = get_global_tracker()
    tracker_cost_usd = tracker.get_total_cost()

    logger.info("")
    logger.info("=" * 80)
    logger.info("CRITERIA QA BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Dataset:       {dataset_path}")
    logger.info(f"Vector store:  {vector_store_path}")
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Correct:       {correct}/{total_queries}")
    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"Total time:    {total_time:.1f}s")
    logger.info(f"Avg latency:   {avg_latency_ms:.1f} ms/query")
    logger.info(f"Total cost:    ${total_cost_usd:.4f} (tracker: ${tracker_cost_usd:.4f})")
    logger.info("=" * 80)

    summary: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "vector_store_path": str(vector_store_path),
        "total_queries": total_queries,
        "correct": correct,
        "accuracy": accuracy,
        "total_time_seconds": total_time,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": total_cost_usd,
        "tracker_total_cost_usd": tracker_cost_usd,
        "per_query": results,
    }

    if agent_config is not None:
        summary["agent_config"] = agent_config

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Criteria QA benchmark."""
    parser = argparse.ArgumentParser(
        description="Run Criteria QA benchmark (YES/NO labels) on Czech nuclear safety criteria",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_dataset/criteria_qa_cs.json",
        help="Path to Criteria QA JSON file",
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
        default=None,
        help=(
            "Retrieval depth (agent_tools.default_k) for RAG tools. "
            "If not set, uses value from config.json."
        ),
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
        help="LLM model used by the agent/orchestrator for QA runs (metadata + overrides)",
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
        help="Override model for orchestrator agent (default: use config.json or --agent-model)",
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
            "If not set, events are not logged."
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
            "(e.g., benchmark_results/criteria_qa_results.json)"
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


def main() -> None:
    args = parse_args()

    # Load environment variables from .env so provider factory can see API keys
    load_dotenv()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_path = Path(args.dataset)
    vector_store_path = Path(args.vector_store)

    # Resolve per-agent model overrides (metadata + overrides for multi-agent run)
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

    # Agent configuration metadata (attached to summary for experiment tracking)
    agent_config: Dict[str, Any] = {
        "setup": args.agent_setup,
        "default_agent_model": default_agent_model,
        "orchestrator": {
            "model": orchestrator_model,
        },
        "agents": per_agent_models,
        "orchestrator_prompt_id": args.orchestrator_prompt_id,
        "notes": args.agent_notes,
        "benchmark_prompt": CRITERIA_QA_BENCHMARK_PROMPT,
    }

    logger.info("=" * 80)
    logger.info("CRITERIA QA BENCHMARK (YES/NO, FULL PIPELINE)")
    logger.info("=" * 80)
    logger.info(f"Dataset:      {dataset_path}")
    logger.info(f"Vector store: {vector_store_path}")
    logger.info(f"Max queries:  {args.max_queries or 'all'}")
    logger.info(f"Retrieval k:  {args.k or 'config.json default'}")
    logger.info(f"Agent setup:  {args.agent_setup or 'n/a'}")
    logger.info(f"Default model:{default_agent_model or 'config.json'}")
    logger.info(f"Orch. model:  {orchestrator_model or 'config.json'}")
    logger.info(f"Orch. prompt: {args.orchestrator_prompt_id or 'n/a'}")
    logger.info("=" * 80)

    # Prepare optional multi-agent log path
    log_path: Optional[Path] = None
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
            stem = base_log_path.name or "criteria_qa_multiagent_logs"
            parent = base_log_path

        log_filename = f"{stem}_{timestamp}{ext}"
        log_path = parent / log_filename

    # Run benchmark (async)
    summary = asyncio.run(
        run_criteria_qa_benchmark(
            dataset_path=dataset_path,
            max_queries=args.max_queries,
            config_path=Path(args.multi_agent_config),
            vector_store_path=vector_store_path,
            orchestrator_model=orchestrator_model,
            default_agent_model=default_agent_model,
            per_agent_models=per_agent_models,
            retrieval_k=args.k,
            log_file=log_path,
            agent_config=agent_config,
        )
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
            stem = base_path.name or "criteria_qa_benchmark"
            parent = base_path if base_path.suffix == "" else base_path.parent

        filename = f"{stem}_{timestamp}.json"
        output_path = parent / filename
        parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
