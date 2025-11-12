"""
Compliance Agent - Regulatory compliance verification.

Responsibilities:
1. GDPR, CCPA, HIPAA, SOX compliance verification
2. Bidirectional checking (Contract → Law, Law → Contract)
3. Violation identification
4. Gap analysis for missing requirements
"""

import logging
from typing import Any, Dict, List

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("compliance")
class ComplianceAgent(BaseAgent):
    """
    Compliance Agent - Verifies regulatory compliance.

    Checks for GDPR, CCPA, HIPAA, SOX compliance and identifies
    violations and gaps using bidirectional verification.
    """

    def __init__(self, config):
        """Initialize compliance agent with config."""
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
        self.system_prompt = prompt_loader.get_prompt("compliance")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"ComplianceAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify compliance with regulations (AUTONOMOUS).

        LLM autonomously decides which tools to call based on query.

        Args:
            state: Current workflow state

        Returns:
            Updated state with compliance findings
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping compliance check")
            return state

        logger.info("Running autonomous compliance verification...")

        try:
            # Run autonomous tool calling loop
            # LLM decides which tools to call (graph_search, assess_confidence, etc.)
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=10
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Extract compliance findings from final answer if structured
            # (for now, store raw answer - can enhance with JSON parsing later)
            compliance_findings = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0),
                "total_tool_cost_usd": agent_cost
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["compliance"] = compliance_findings

            logger.info(
                f"Autonomous compliance check complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            # Set final answer if this is last agent in sequence
            agent_sequence = state.get("agent_sequence", [])
            current_index = agent_sequence.index("compliance") if "compliance" in agent_sequence else -1
            is_last = current_index == len(agent_sequence) - 1
            next_is_report = (
                current_index < len(agent_sequence) - 1
                and agent_sequence[current_index + 1] == "report_generator"
            )

            if is_last and not next_is_report:
                logger.info("Compliance is last agent - using autonomous answer")
                state["final_answer"] = final_answer

            return state

        except Exception as e:
            logger.error(f"Autonomous compliance verification failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Compliance error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
