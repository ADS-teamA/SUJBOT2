"""
Gap Synthesizer Agent - Knowledge gap analysis and completeness assessment.

Responsibilities:
1. Regulatory gap identification (missing required clauses)
2. Coverage gap analysis (topics not fully addressed)
3. Consistency gap detection (contradictions)
4. Citation gap finding (claims without evidence)
5. Temporal gap identification (outdated information)
"""

import logging
from typing import Any, Dict, List


from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("gap_synthesizer")
class GapSynthesizerAgent(BaseAgent):
    """
    Gap Synthesizer Agent - Identifies knowledge gaps and missing information.

    Analyzes completeness across 5 gap types: Regulatory, Coverage,
    Consistency, Citation, and Temporal gaps.
    """

    def __init__(self, config):
        """Initialize gap synthesizer with config."""
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
        self.system_prompt = prompt_loader.get_prompt("gap_synthesizer")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"GapSynthesizerAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize gaps between requirements (AUTONOMOUS).

        LLM autonomously decides which tools to call for gap analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with identified gaps
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping gap synthesizer")
            return state

        logger.info("Running autonomous gap synthesizer...")

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
            state["agent_outputs"]["gap_synthesizer"] = output

            logger.info(
                f"Autonomous gap synthesizer complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous gap synthesizer failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"gap synthesizer error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
