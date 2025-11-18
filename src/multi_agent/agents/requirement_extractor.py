"""
Requirement Extractor Agent - Atomic legal obligation extraction.

Responsibilities:
1. Extract atomic legal requirements from legal texts (laws, regulations, standards)
2. Decompose complex provisions into verifiable obligations
3. Classify requirements by granularity, severity, and applicability
4. Generate structured compliance checklist for downstream agents

Based on Legal AI Research (2024): Requirement-First Compliance Checking
"""

import logging
from typing import Any, Dict, List

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("requirement_extractor")
class RequirementExtractorAgent(BaseAgent):
    """
    Requirement Extractor Agent - Extracts atomic legal obligations.

    Decomposes legal texts into atomic, verifiable requirements following
    the Plan-and-Solve pattern (Zhou et al., 2023).
    """

    def __init__(self, config):
        """Initialize requirement extractor agent with config."""
        super().__init__(config)

        # Initialize provider (auto-detects from model name: claude/gpt/gemini)
        try:
            from src.agent.providers.factory import create_provider

            self.provider = create_provider(model=config.model)
            logger.info(f"Initialized provider for model: {config.model}")
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            raise ValueError(
                f"Failed to initialize LLM provider for model {config.model}. "
                f"Ensure API keys are configured in environment and model name is valid."
            ) from e

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("requirement_extractor")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"RequirementExtractorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract atomic requirements from legal texts (AUTONOMOUS).

        LLM autonomously decides which tools to call based on query.

        Workflow:
        1. Identify target legal text (from extractor output or direct law reference)
        2. Use hierarchical_search to retrieve relevant law sections
        3. Use definition_aligner to identify legal terminology requiring alignment
        4. Decompose provisions into atomic requirements
        5. Output structured checklist for ComplianceAgent

        Args:
            state: Current workflow state

        Returns:
            Updated state with requirement_checklist
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        logger.info("Running autonomous requirement extraction...")

        try:
            # Run autonomous tool calling loop
            # LLM decides which tools to call (hierarchical_search, graph_search, definition_aligner)
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=15  # Higher limit for complex legal text processing
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Extract structured checklist from final answer
            # The system prompt instructs LLM to output JSON with requirement checklist
            # For now, store raw answer - will enhance with JSON parsing
            requirement_extraction = {
                "checklist": final_answer,  # TODO: Parse JSON checklist
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0),
                "total_tool_cost_usd": agent_cost
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["requirement_extractor"] = requirement_extraction

            logger.info(
                f"Autonomous requirement extraction complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            # Do NOT set final_answer - this is intermediate step
            # ComplianceAgent will consume the checklist

            return state

        except Exception as e:
            logger.error(f"Autonomous requirement extraction failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"RequirementExtractor error: {str(e)}")
            return state

    # Note: No hardcoded methods needed - autonomous pattern handles everything via LLM
