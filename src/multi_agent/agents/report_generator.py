"""
Report Generator Agent - Final report compilation and synthesis.

Responsibilities:
1. Executive summary creation
2. Detailed findings compilation
3. Compliance matrix generation
4. Risk assessment summary
5. Citations and references consolidation
6. Recommendations prioritization
7. Appendix with metadata
"""

import logging
from typing import Any, Dict
from datetime import datetime


from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("report_generator")
class ReportGeneratorAgent(BaseAgent):
    """
    Report Generator Agent - Synthesizes all agent outputs into final report.

    Creates comprehensive, well-structured Markdown report with:
    - Executive summary
    - Detailed findings
    - Compliance matrix
    - Risk assessment
    - Citations
    - Recommendations
    - Appendix with execution metadata
    """

    def __init__(self, config):
        """Initialize report generator with config."""
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
        self.system_prompt = prompt_loader.get_prompt("report_generator")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"ReportGeneratorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report (AUTONOMOUS).

        LLM autonomously decides which tools to call for report generation.

        Args:
            state: Current workflow state

        Returns:
            Updated state with formatted report
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping report generator")
            return state

        logger.info("Running autonomous report generator...")

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
            report_gen_cost = result.get("total_tool_cost_usd", 0.0)

            # Aggregate costs from ALL agents (including report_generator)
            total_cost = report_gen_cost  # Start with report_generator's own cost
            agent_costs = {"report_generator": report_gen_cost}

            # Sum costs from previous agents
            for agent_name, agent_output in state.get("agent_outputs", {}).items():
                if agent_name != "report_generator" and isinstance(agent_output, dict):
                    agent_cost = agent_output.get("total_tool_cost_usd", 0.0)
                    if agent_cost > 0:
                        agent_costs[agent_name] = agent_cost
                        total_cost += agent_cost

            # Store output
            output = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0),
                "total_tool_cost_usd": report_gen_cost,
                "workflow_total_cost_usd": total_cost,  # Total for entire workflow
                "agent_costs": agent_costs  # Per-agent cost breakdown
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["report_generator"] = output

            # Append cost summary to final answer
            cost_summary = self._format_cost_summary(total_cost, agent_costs)
            final_answer_with_cost = f"{final_answer}\n\n{cost_summary}"

            # Set final_answer at top level for runner extraction
            state["final_answer"] = final_answer_with_cost

            logger.info(
                f"Autonomous report generator complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}, "
                f"total_workflow_cost=${total_cost:.6f}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous report generator failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"report generator error: {str(e)}")
            return state

    def _format_cost_summary(self, total_cost: float, agent_costs: dict) -> str:
        """
        Format cost summary for display to user.

        Args:
            total_cost: Total workflow cost in USD
            agent_costs: Dict mapping agent names to costs

        Returns:
            Formatted markdown cost summary
        """
        lines = [
            "---",
            "## 💰 API Cost Summary",
            f"**Total Workflow Cost:** ${total_cost:.6f}",
            ""
        ]

        # Per-agent breakdown
        if agent_costs:
            lines.append("**Per-Agent Breakdown:**")
            # Sort by cost descending
            sorted_agents = sorted(agent_costs.items(), key=lambda x: x[1], reverse=True)
            for agent_name, cost in sorted_agents:
                if cost > 0:
                    lines.append(f"- {agent_name}: ${cost:.6f}")
            lines.append("")

        # Cost interpretation
        if total_cost < 0.01:
            lines.append("_Cost: Minimal (< $0.01)_")
        elif total_cost < 0.05:
            lines.append("_Cost: Low ($0.01 - $0.05)_")
        elif total_cost < 0.20:
            lines.append("_Cost: Moderate ($0.05 - $0.20)_")
        else:
            lines.append("_Cost: High (> $0.20)_")

        lines.append("---")

        return "\n".join(lines)

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
