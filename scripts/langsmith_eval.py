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
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langsmith import Client
from openevals.llm import create_llm_as_judge

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

        return {
            "answer": result.get("final_answer", "ERROR: No final answer"),
            "success": result.get("success", False),
            "agent_sequence": result.get("agent_sequence", []),
            "cost_cents": result.get("total_cost_cents", 0),
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
) -> str:
    """Upload evaluation dataset to LangSmith."""
    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
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
    experiment_prefix: str = "sujbot2-qa-eval",
    max_concurrency: int = 1,  # Use 1 to avoid async/DB pool conflicts
    limit: Optional[int] = None,
):
    """Run evaluation using LangSmith async evaluate."""
    logger.info(f"Starting evaluation on dataset: {dataset_name}")

    # Create async target function - no event loop gymnastics needed
    async def async_target_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async target function for LangSmith aevaluate()."""
        try:
            result = await evaluator.run_query(inputs["question"])
            return result
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {"answer": f"ERROR: {str(e)}"}

    # Wrap evaluators for LangSmith signature
    wrapped_evaluators = [wrap_evaluator(e) for e in evaluators]

    # Run async evaluation - clean async-all-the-way pattern
    # Using aevaluate() avoids all nest_asyncio/event loop conflicts
    results = await client.aevaluate(
        async_target_function,
        data=dataset_name,
        evaluators=wrapped_evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        num_repetitions=1,
    )

    logger.info("Evaluation completed!")
    return results


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

    # Load and upload dataset
    eval_data = load_eval_dataset(args.dataset_path)
    if args.limit:
        eval_data = eval_data[: args.limit]
        logger.info(f"Limited to {len(eval_data)} examples")

    dataset_id = upload_dataset_to_langsmith(
        client, eval_data, args.dataset_name
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
            experiment_prefix=args.experiment_prefix,
            max_concurrency=args.concurrency,
            limit=args.limit,
        )

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info("View results in LangSmith UI:")
        logger.info("  https://smith.langchain.com/")

    finally:
        # Clean shutdown
        await evaluator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
