#!/usr/bin/env python3
"""
LangSmith Evaluation Script for SUJBOT2 Multi-Agent System.

Evaluates the multi-agent RAG system using LangSmith's evaluation framework
with LLM-as-judge evaluators for semantic correctness, factual accuracy,
and completeness.

Usage:
    # Upload dataset and run evaluation
    uv run python scripts/langsmith_eval.py

    # Only upload dataset (no evaluation)
    uv run python scripts/langsmith_eval.py --upload-only

    # Run evaluation on existing dataset
    uv run python scripts/langsmith_eval.py --dataset-name "sujbot2-eval-qa"

    # Limit number of examples for testing
    uv run python scripts/langsmith_eval.py --limit 5

    # Use different judge model
    uv run python scripts/langsmith_eval.py --judge-model gpt-4o-mini

Requirements:
    - openevals (for LLM-as-judge)
    - langsmith (for dataset management and evaluation)
    - Documents in eval.json must be indexed in PostgreSQL
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langsmith import Client
from openevals.llm import create_llm_as_judge

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply nest_asyncio early to allow nested event loops
# This fixes Neo4j/graphiti_core "Future attached to different loop" errors
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM-as-Judge Prompts (Czech-aware)
# =============================================================================

SEMANTIC_CORRECTNESS_PROMPT = '''
Jsi evaluátor odpovědí AI systému. Porovnej odpověď systému s referenční odpovědí z hlediska SÉMANTICKÉ SPRÁVNOSTI.

Hodnoť, zda odpověď systému vyjadřuje STEJNÝ VÝZNAM jako referenční odpověď, i když používá jiná slova nebo formulace.

<inputs>
{inputs}
</inputs>

<reference_outputs>
{reference_outputs}
</reference_outputs>

<outputs>
{outputs}
</outputs>

Z inputs extrahuj "question" (otázka).
Z reference_outputs extrahuj "answer" (referenční odpověď).
Z outputs extrahuj "answer" (odpověď systému).

Kritéria hodnocení:
- 1.0: Plná sémantická shoda - odpověď vyjadřuje stejný význam
- 0.7-0.9: Většinová shoda - hlavní body jsou správné, mohou chybět detaily
- 0.4-0.6: Částečná shoda - některé body jsou správné, jiné chybí nebo jsou špatně
- 0.1-0.3: Minimální shoda - odpověď se jen okrajově dotýká správné odpovědi
- 0.0: Žádná shoda - odpověď je zcela mimo téma nebo špatná

Odpověz pouze číslem mezi 0.0 a 1.0.
'''

FACTUAL_ACCURACY_PROMPT = '''
Jsi evaluátor odpovědí AI systému. Ověř FAKTICKOU PŘESNOST odpovědi systému oproti referenční odpovědi.

Zaměř se zejména na:
- Číselné hodnoty (100 Wt, 500 Wt, 0.089 g, 15 g, atd.)
- Procenta a jednotky (%, Bq/cm², kPa)
- Technické názvy a označení (IRT-4M, UR-70, 08CH18N10T)
- Časové údaje (2030, 72 hodin, 3 měsíce, 10 000 let)
- Množství a počty (5-7 tyčí, 15 mm, 20 mm)

<reference_outputs>
{reference_outputs}
</reference_outputs>

<outputs>
{outputs}
</outputs>

Z reference_outputs extrahuj "answer" (referenční odpověď).
Z outputs extrahuj "answer" (odpověď systému).

Kritéria hodnocení:
- 1.0: Všechna fakta jsou správná
- 0.7-0.9: Drobné nepřesnosti, které nemění podstatu
- 0.4-0.6: Některá fakta chybí nebo jsou nepřesná
- 0.1-0.3: Závažné faktické chyby
- 0.0: Fakta jsou zcela špatná nebo vymyšlená

Odpověz pouze číslem mezi 0.0 a 1.0.
'''

COMPLETENESS_PROMPT = '''
Jsi evaluátor odpovědí AI systému. Hodnoť ÚPLNOST odpovědi systému - zda pokrývá všechny klíčové body z referenční odpovědi.

<reference_outputs>
{reference_outputs}
</reference_outputs>

<outputs>
{outputs}
</outputs>

Z reference_outputs extrahuj "answer" (referenční odpověď).
Z outputs extrahuj "answer" (odpověď systému).

Identifikuj klíčové body v referenční odpovědi a ověř, zda jsou obsaženy v odpovědi systému.

Kritéria hodnocení:
- 1.0: Všechny klíčové body jsou pokryty
- 0.7-0.9: Většina klíčových bodů je pokryta
- 0.4-0.6: Přibližně polovina klíčových bodů
- 0.1-0.3: Pouze minimum klíčových bodů
- 0.0: Žádné klíčové body nejsou pokryty

Odpověz pouze číslem mezi 0.0 a 1.0.
'''


# =============================================================================
# Multi-Agent Runner Integration
# =============================================================================

class MultiAgentEvaluator:
    """Wrapper for MultiAgentRunner that runs queries and returns answers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runner = None
        self._initialized = False

    async def initialize(self):
        """Initialize the multi-agent runner."""
        if self._initialized:
            return

        from src.multi_agent.runner import MultiAgentRunner

        self.runner = MultiAgentRunner(self.config)
        await self.runner.initialize()
        self._initialized = True
        logger.info("MultiAgentRunner initialized successfully")

    async def run_query(self, question: str) -> Dict[str, Any]:
        """Run a single query through the multi-agent system."""
        if not self._initialized:
            await self.initialize()

        result = None
        async for event in self.runner.run_query(question, stream_progress=False):
            if event.get("type") == "final":
                result = event

        if result is None:
            return {"answer": "ERROR: No response from multi-agent system"}

        # Extract tool usage from tool_executions
        tool_executions = result.get("tool_executions", [])
        tools_used = [te.get("tool_name", "unknown") for te in tool_executions]

        return {
            "answer": result.get("final_answer", "ERROR: No final answer"),
            "success": result.get("success", False),
            "agent_sequence": result.get("agent_sequence", []),
            "cost_cents": result.get("total_cost_cents", 0),
            "tools_used": tools_used,
        }

    async def shutdown(self):
        """Clean shutdown of runner resources."""
        if self.runner:
            try:
                await self.runner.shutdown_async()
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")


# =============================================================================
# LangSmith Integration
# =============================================================================

def load_eval_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON file."""
    with open(dataset_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def upload_dataset_to_langsmith(
    client: Client,
    eval_data: List[Dict[str, Any]],
    dataset_name: str = "sujbot2-eval-qa",
    description: str = "Czech legal/nuclear QA evaluation pairs",
    replace: bool = False,
) -> str:
    """Upload evaluation dataset to LangSmith."""
    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        if replace:
            logger.info(f"Deleting existing dataset '{dataset_name}' (ID: {existing.id})")
            client.delete_dataset(dataset_id=existing.id)
        else:
            logger.info(f"Dataset '{dataset_name}' already exists (ID: {existing.id})")
            return existing.id
    except Exception:
        pass  # Dataset doesn't exist, create it

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    logger.info(f"Created dataset '{dataset_name}' (ID: {dataset.id})")

    # Prepare examples
    examples = [
        {
            "inputs": {
                "question": item["question"],
                "source_document": item.get("source_document", ""),
            },
            "outputs": {"answer": item["answer"]},
            "metadata": {"id": item.get("id", i)},
        }
        for i, item in enumerate(eval_data)
    ]

    # Upload examples
    client.create_examples(dataset_id=dataset.id, examples=examples)
    logger.info(f"Uploaded {len(examples)} examples to dataset")

    return dataset.id


def create_evaluators(judge_model: str = "anthropic:claude-sonnet-4-5") -> List[Any]:
    """Create LLM-as-judge evaluators."""
    evaluators = [
        create_llm_as_judge(
            prompt=SEMANTIC_CORRECTNESS_PROMPT,
            feedback_key="semantic_correctness",
            model=judge_model,
            continuous=True,
        ),
        create_llm_as_judge(
            prompt=FACTUAL_ACCURACY_PROMPT,
            feedback_key="factual_accuracy",
            model=judge_model,
            continuous=True,
        ),
        create_llm_as_judge(
            prompt=COMPLETENESS_PROMPT,
            feedback_key="completeness",
            model=judge_model,
            continuous=True,
        ),
    ]
    logger.info(f"Created {len(evaluators)} evaluators with model: {judge_model}")
    return evaluators


def wrap_evaluator(evaluator):
    """Wrap openevals evaluator for LangSmith evaluate() signature."""
    def wrapped(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        return evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
    return wrapped


async def run_evaluation(
    client: Client,
    evaluator: MultiAgentEvaluator,
    dataset_name: str,
    evaluators: List[Any],
    eval_data: List[Dict[str, Any]],  # Original eval.json data for ordering
    experiment_prefix: str = "sujbot2-qa-eval",
    max_concurrency: int = 1,  # Use 1 to avoid async/DB pool conflicts
    limit: Optional[int] = None,
):
    """Run evaluation using LangSmith async evaluate."""
    logger.info(f"Starting evaluation on dataset: {dataset_name}")

    # Create async target function with response time tracking
    async def async_target_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async target function for LangSmith aevaluate()."""
        start_time = time.time()
        try:
            result = await evaluator.run_query(inputs["question"])
            result["response_time_ms"] = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {
                "answer": f"ERROR: {str(e)}",
                "response_time_ms": (time.time() - start_time) * 1000,
            }

    # Wrap evaluators for LangSmith signature
    wrapped_evaluators = [wrap_evaluator(e) for e in evaluators]

    # Use examples in ORDER from the original eval.json file
    # LangSmith's list_examples() doesn't preserve upload order, so we:
    # 1. Fetch all examples from LangSmith
    # 2. Reorder them to match eval.json order
    # 3. Take first N if limit is specified
    if limit:
        # Fetch all examples from LangSmith dataset
        all_examples = list(client.list_examples(dataset_name=dataset_name))

        # Create mapping from question text to Example object
        question_to_example: dict[str, Any] = {}
        for ex in all_examples:
            q = ex.inputs.get("question", "") if hasattr(ex.inputs, "get") else ""
            if q:
                question_to_example[q] = ex

        # Reorder examples to match eval.json order
        ordered_examples = []
        for item in eval_data:
            question = item["question"]
            if question in question_to_example:
                ordered_examples.append(question_to_example[question])

        # Take first N examples (now in JSON file order)
        examples_to_run = ordered_examples[:limit]
        logger.info(f"Using first {len(examples_to_run)} examples from eval.json (ordered)")
        data_source = examples_to_run
    else:
        # Use full dataset from LangSmith (order may vary)
        data_source = dataset_name

    # Run async evaluation - clean async-all-the-way pattern
    # Using aevaluate() avoids all nest_asyncio/event loop conflicts
    results = await client.aevaluate(
        async_target_function,
        data=data_source,
        evaluators=wrapped_evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        num_repetitions=1,
    )

    logger.info("Evaluation completed!")
    return results


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(
    results_data: List[Dict[str, Any]],
    experiment_prefix: str,
    judge_model: str,
    output_dir: Path = PROJECT_ROOT / "evaluations",
) -> Path:
    """
    Generate markdown summary of evaluation results.

    Args:
        results_data: List of evaluation results with outputs and feedback
        experiment_prefix: Name prefix used for the experiment
        judge_model: Model used for LLM-as-judge
        output_dir: Directory to write summary file

    Returns:
        Path to generated summary file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics from results
    semantic_scores = []
    factual_scores = []
    completeness_scores = []
    response_times = []
    costs = []
    per_question_data = []

    for i, result in enumerate(results_data):
        outputs = result.get("outputs", {})
        feedback = result.get("feedback", {})
        inputs = result.get("inputs", {})

        # Extract scores from feedback
        if "semantic_correctness" in feedback:
            score = feedback["semantic_correctness"].get("score")
            if score is not None:
                semantic_scores.append(float(score))

        if "factual_accuracy" in feedback:
            score = feedback["factual_accuracy"].get("score")
            if score is not None:
                factual_scores.append(float(score))

        if "completeness" in feedback:
            score = feedback["completeness"].get("score")
            if score is not None:
                completeness_scores.append(float(score))

        # Extract response time and cost from outputs
        if "response_time_ms" in outputs:
            response_times.append(outputs["response_time_ms"])

        if "cost_cents" in outputs:
            costs.append(outputs["cost_cents"])

        # Collect per-question data
        question = inputs.get("question", f"Question {i+1}")[:80]
        per_question_data.append({
            "question": question,
            "semantic": feedback.get("semantic_correctness", {}).get("score"),
            "factual": feedback.get("factual_accuracy", {}).get("score"),
            "completeness": feedback.get("completeness", {}).get("score"),
            "response_time_ms": outputs.get("response_time_ms"),
            "cost_cents": outputs.get("cost_cents"),
        })

    # Calculate statistics
    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0, "std": 0, "p50": 0, "p99": 0}
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "std": float(np.std(arr_np)),
            "p50": float(np.percentile(arr_np, 50)),
            "p99": float(np.percentile(arr_np, 99)),
        }

    semantic_stats = stats(semantic_scores)
    factual_stats = stats(factual_scores)
    completeness_stats = stats(completeness_scores)
    time_stats = stats(response_times)
    total_cost_cents = sum(costs)
    avg_cost_cents = total_cost_cents / len(costs) if costs else 0

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date_display = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build markdown content
    md_lines = [
        f"# Evaluation Summary: {experiment_prefix}",
        "",
        f"**Date:** {date_display}",
        f"**Examples:** {len(results_data)}",
        f"**Judge Model:** {judge_model}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean | Std | P50 | P99 |",
        "|--------|------|-----|-----|-----|",
        f"| semantic_correctness | {semantic_stats['mean']:.3f} | {semantic_stats['std']:.3f} | {semantic_stats['p50']:.3f} | {semantic_stats['p99']:.3f} |",
        f"| factual_accuracy | {factual_stats['mean']:.3f} | {factual_stats['std']:.3f} | {factual_stats['p50']:.3f} | {factual_stats['p99']:.3f} |",
        f"| completeness | {completeness_stats['mean']:.3f} | {completeness_stats['std']:.3f} | {completeness_stats['p50']:.3f} | {completeness_stats['p99']:.3f} |",
        "",
        "## Cost",
        "",
        f"- **Total:** ${total_cost_cents / 100:.4f}",
        f"- **Per query avg:** ${avg_cost_cents / 100:.4f}",
        "",
        "## Response Time",
        "",
        f"- **Mean:** {time_stats['mean'] / 1000:.2f}s",
        f"- **P50:** {time_stats['p50'] / 1000:.2f}s",
        f"- **P99:** {time_stats['p99'] / 1000:.2f}s",
        "",
        "## Per-Question Breakdown",
        "",
        "| # | Question | Semantic | Factual | Complete | Time (s) | Cost (¢) |",
        "|---|----------|----------|---------|----------|----------|----------|",
    ]

    for i, pq in enumerate(per_question_data):
        q_short = pq["question"][:50] + "..." if len(pq["question"]) > 50 else pq["question"]
        sem = f"{pq['semantic']:.2f}" if pq['semantic'] is not None else "-"
        fac = f"{pq['factual']:.2f}" if pq['factual'] is not None else "-"
        comp = f"{pq['completeness']:.2f}" if pq['completeness'] is not None else "-"
        time_s = f"{pq['response_time_ms'] / 1000:.2f}" if pq['response_time_ms'] else "-"
        cost_c = f"{pq['cost_cents']:.2f}" if pq['cost_cents'] else "-"
        md_lines.append(f"| {i+1} | {q_short} | {sem} | {fac} | {comp} | {time_s} | {cost_c} |")

    md_content = "\n".join(md_lines) + "\n"

    # Write markdown file
    filename = f"{timestamp}_{experiment_prefix}_{len(results_data)}examples.md"
    output_path = output_dir / filename
    output_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Summary written to: {output_path}")

    # Build and write JSON with detailed Q&A data
    json_data = {
        "metadata": {
            "date": date_display,
            "experiment_prefix": experiment_prefix,
            "judge_model": judge_model,
            "num_examples": len(results_data),
        },
        "metrics": {
            "semantic_correctness": semantic_stats,
            "factual_accuracy": factual_stats,
            "completeness": completeness_stats,
        },
        "cost": {
            "total_cents": total_cost_cents,
            "avg_per_query_cents": avg_cost_cents,
        },
        "response_time": {
            "mean_ms": time_stats["mean"],
            "p50_ms": time_stats["p50"],
            "p99_ms": time_stats["p99"],
        },
        "results": [],
    }

    # Add detailed per-question data
    for i, result in enumerate(results_data):
        inputs = result.get("inputs", {})
        outputs = result.get("outputs", {})
        reference = result.get("reference_outputs", {})
        feedback = result.get("feedback", {})

        json_data["results"].append({
            "id": i + 1,
            "question": inputs.get("question", ""),
            "source_document": inputs.get("source_document", ""),
            "reference_answer": reference.get("answer", ""),
            "model_answer": outputs.get("answer", ""),
            "scores": {
                "semantic_correctness": feedback.get("semantic_correctness", {}).get("score"),
                "factual_accuracy": feedback.get("factual_accuracy", {}).get("score"),
                "completeness": feedback.get("completeness", {}).get("score"),
            },
            "response_time_ms": outputs.get("response_time_ms"),
            "cost_cents": outputs.get("cost_cents"),
            "tools_used": outputs.get("tools_used", []),
        })

    # Write JSON file
    json_filename = f"{timestamp}_{experiment_prefix}_{len(results_data)}examples.json"
    json_path = output_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON data written to: {json_path}")

    return output_path


async def collect_results(results) -> List[Dict[str, Any]]:
    """
    Collect all results from async iterator and transform to summary format.

    LangSmith aevaluate() returns dicts with:
    - run: RunTree object with inputs, outputs
    - example: Example object with reference inputs/outputs
    - evaluation_results: {'results': [EvaluationResult(key, score), ...]}

    We transform these into the format expected by generate_summary():
    - inputs: dict with question
    - outputs: dict with answer, response_time_ms, cost_cents
    - feedback: dict mapping feedback_key to {score: float}
    """
    results_list = []
    async for result in results:
        if isinstance(result, dict) and "run" in result:
            # LangSmith aevaluate() returns dict with 'run', 'example', 'evaluation_results'
            run = result["run"]
            eval_results = result.get("evaluation_results", {})

            # Extract inputs - run.inputs is {'inputs': {...actual inputs...}}
            raw_inputs = run.inputs if hasattr(run, "inputs") else {}
            # Unwrap nested 'inputs' if present (LangSmith wraps them)
            inputs = raw_inputs.get("inputs", raw_inputs) if isinstance(raw_inputs, dict) else {}

            # Extract outputs directly from run
            outputs = run.outputs if hasattr(run, "outputs") else {}

            # Extract reference outputs (GT answers) from example
            example = result.get("example")
            reference_outputs = {}
            if example and hasattr(example, "outputs"):
                reference_outputs = example.outputs if example.outputs else {}

            # Transform evaluation_results to feedback format
            # eval_results = {'results': [EvaluationResult(key, score), ...]}
            feedback = {}
            eval_list = eval_results.get("results", []) if isinstance(eval_results, dict) else []
            for eval_result in eval_list:
                if hasattr(eval_result, "key") and hasattr(eval_result, "score"):
                    feedback[eval_result.key] = {"score": eval_result.score}

            results_list.append({
                "inputs": inputs,
                "outputs": outputs,
                "reference_outputs": reference_outputs,
                "feedback": feedback,
            })
        else:
            # Unexpected format - log and skip
            logger.warning(f"Unexpected result format: {type(result)}")

    logger.info(f"Collected {len(results_list)} results")
    return results_list


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation on SUJBOT2 multi-agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "eval.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sujbot2-eval-qa",
        help="LangSmith dataset name",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload dataset, don't run evaluation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="anthropic:claude-sonnet-4-5",
        choices=[
            "anthropic:claude-sonnet-4-5",
            "anthropic:claude-haiku-4-5",
            "openai:gpt-4o-mini",
            "openai:gpt-4o",
        ],
        help="Model to use for LLM-as-judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent evaluations (default: 1 to avoid DB pool conflicts)",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="sujbot2-qa-eval",
        help="Prefix for experiment name in LangSmith",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--replace-dataset",
        action="store_true",
        help="Delete and replace existing dataset with new examples",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate environment
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set in environment")
        sys.exit(1)

    # Load config
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Initialize LangSmith client
    client = Client()
    logger.info("LangSmith client initialized")

    # Load full dataset (limit is applied during evaluation, not upload)
    eval_data = load_eval_dataset(args.dataset_path)

    # Upload full dataset to LangSmith (or skip if exists)
    dataset_id = upload_dataset_to_langsmith(
        client, eval_data, args.dataset_name, replace=args.replace_dataset
    )

    if args.upload_only:
        logger.info("Dataset uploaded. Exiting (--upload-only mode)")
        return

    # Initialize evaluator
    evaluator = MultiAgentEvaluator(config)
    await evaluator.initialize()

    try:
        # Create evaluators
        evaluators = create_evaluators(args.judge_model)

        # Run evaluation
        results = await run_evaluation(
            client=client,
            evaluator=evaluator,
            dataset_name=args.dataset_name,
            evaluators=evaluators,
            eval_data=eval_data,  # Pass original JSON data for ordered examples
            experiment_prefix=args.experiment_prefix,
            max_concurrency=args.concurrency,
            limit=args.limit,
        )

        # Collect results for summary generation
        logger.info("Collecting results for summary generation...")
        results_data = await collect_results(results)
        logger.info(f"Collected {len(results_data)} results")

        # Generate summary
        summary_path = generate_summary(
            results_data=results_data,
            experiment_prefix=args.experiment_prefix,
            judge_model=args.judge_model,
        )

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Summary: {summary_path}")
        logger.info("View results in LangSmith UI:")
        logger.info("  https://smith.langchain.com/")

    finally:
        # Clean shutdown
        await evaluator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
