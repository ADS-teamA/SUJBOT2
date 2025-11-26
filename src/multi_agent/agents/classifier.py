"""
Classifier Agent - Content categorization and organization.

Responsibilities:
1. Document type classification (Contract, Policy, Report, etc.)
2. Domain identification (Legal, Technical, Financial, etc.)
3. Complexity assessment
4. Language detection and sensitivity classification
"""

import logging
from typing import Any, Dict


from ..core.agent_base import BaseAgent
from ..core.agent_initializer import initialize_agent
from ..core.agent_registry import register_agent

logger = logging.getLogger(__name__)


@register_agent("classifier")
class ClassifierAgent(BaseAgent):
    """
    Classifier Agent - Categorizes documents and content.

    Classifies along multiple dimensions: document type, domain,
    complexity, language, and sensitivity level.
    """

    def __init__(self, config):
        """Initialize classifier with config."""
        super().__init__(config)

        # Initialize common components (provider, prompts, tools)
        components = initialize_agent(config, "classifier")
        self.provider = components.provider
        self.system_prompt = components.system_prompt
        self.tool_adapter = components.tool_adapter

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify query intent and complexity (AUTONOMOUS).

        LLM autonomously decides which tools to call for query classification.

        Args:
            state: Current workflow state

        Returns:
            Updated state with classification results
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping classifier")
            return state

        logger.info("Running autonomous classifier...")

        try:
            # Run autonomous tool calling loop
            # LLM decides which tools to call
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=10
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])
            agent_cost = result.get("total_tool_cost_usd", 0.0)

            # Store output
            output = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0),
                "total_tool_cost_usd": agent_cost
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["classifier"] = output

            logger.info(
                f"Autonomous classifier complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous classifier failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"classifier error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
