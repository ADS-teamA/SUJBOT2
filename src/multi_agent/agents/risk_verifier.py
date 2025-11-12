"""
Risk Verifier Agent - Risk assessment and verification.

Responsibilities:
1. Risk identification (Legal, Financial, Operational, Compliance, Reputational)
2. Severity and likelihood assessment
3. Comparison with industry standards
4. Mitigation recommendations
"""

import logging
from typing import Any, Dict, List


from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("risk_verifier")
class RiskVerifierAgent(BaseAgent):
    """
    Risk Verifier Agent - Assesses and verifies risks.

    Identifies risks across 5 categories (Legal, Financial, Operational,
    Compliance, Reputational) and provides severity scores with mitigation.
    """

    def __init__(self, config):
        """Initialize risk verifier with config."""
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
        self.system_prompt = prompt_loader.get_prompt("risk_verifier")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"RiskVerifierAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess and verify risks (AUTONOMOUS).

        LLM autonomously decides which tools to call for risk analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with risk assessment
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping risk verification")
            return state

        logger.info("Running autonomous risk assessment...")

        try:
            # Run autonomous tool calling loop
            # LLM decides which tools to call (similarity_search, compare_documents, etc.)
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=10
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Store risk assessment
            risk_assessment = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0),
                "total_tool_cost_usd": agent_cost
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["risk_verifier"] = risk_assessment

            logger.info(
                f"Autonomous risk assessment complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous risk verification failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Risk verification error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
