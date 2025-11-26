"""
Citation Auditor Agent - Citation verification and validation.

Responsibilities:
1. Citation existence verification
2. Citation accuracy checking (text matches)
3. Citation completeness validation
4. Citation format standardization
5. Broken reference detection
"""

import logging
import re
from typing import Any, Dict, List


from ..core.agent_base import BaseAgent
from ..core.agent_initializer import initialize_agent
from ..core.agent_registry import register_agent

logger = logging.getLogger(__name__)


@register_agent("citation_auditor")
class CitationAuditorAgent(BaseAgent):
    """
    Citation Auditor Agent - Verifies citation accuracy and completeness.

    Audits all citations to ensure they are accurate, complete, properly
    formatted, and point to accessible sources.
    """

    def __init__(self, config):
        """Initialize citation auditor with config."""
        super().__init__(config)

        # Initialize common components (provider, prompts, tools)
        components = initialize_agent(config, "citation_auditor")
        self.provider = components.provider
        self.system_prompt = components.system_prompt
        self.tool_adapter = components.tool_adapter

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit citations for accuracy (AUTONOMOUS).

        LLM autonomously decides which tools to call for citation verification.

        Args:
            state: Current workflow state

        Returns:
            Updated state with citation audit results
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping citation auditor")
            return state

        logger.info("Running autonomous citation auditor...")

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
            state["agent_outputs"]["citation_auditor"] = output

            logger.info(
                f"Autonomous citation auditor complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous citation auditor failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"citation auditor error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
