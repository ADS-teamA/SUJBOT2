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
from ..core.agent_initializer import initialize_agent
from ..core.agent_registry import register_agent

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

        # Initialize common components (provider, prompts, tools)
        components = initialize_agent(config, "compliance")
        self.provider = components.provider
        self.system_prompt = components.system_prompt
        self.tool_adapter = components.tool_adapter

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify compliance with regulations using checklist from RequirementExtractor (AUTONOMOUS).

        LLM autonomously decides which tools to call based on query.

        Args:
            state: Current workflow state

        Returns:
            Updated state with compliance findings

        Raises:
            ValueError: If requirement_extractor output is missing or invalid
        """
        query = state.get("query", "")

        # CRITICAL: ComplianceAgent REQUIRES RequirementExtractor output (SOTA 2024 - requirement-first)
        requirement_extractor_output = state.get("agent_outputs", {}).get("requirement_extractor", {})

        if not requirement_extractor_output:
            error_msg = (
                "ComplianceAgent error: Missing requirement_extractor output. "
                "Compliance agent requires checklist from RequirementExtractorAgent. "
                "Ensure orchestrator routes: extractor → requirement_extractor → compliance."
            )
            logger.error(error_msg)
            state["errors"] = state.get("errors", [])
            state["errors"].append(error_msg)
            return state

        # Validate and parse checklist JSON
        checklist_str = requirement_extractor_output.get("checklist", "")
        if not checklist_str:
            error_msg = (
                "ComplianceAgent error: requirement_extractor output missing 'checklist' field. "
                "RequirementExtractor must generate JSON checklist with atomic requirements."
            )
            logger.error(error_msg)
            state["errors"] = state.get("errors", [])
            state["errors"].append(error_msg)
            return state

        try:
            import json
            checklist_data = json.loads(checklist_str)

            # Validate checklist structure
            if "checklist" not in checklist_data:
                raise ValueError("Checklist JSON missing 'checklist' array")
            if not isinstance(checklist_data["checklist"], list):
                raise ValueError("'checklist' must be an array of requirements")
            if len(checklist_data["checklist"]) == 0:
                raise ValueError("Checklist is empty - no requirements to verify")

            # Log checklist summary
            num_requirements = len(checklist_data["checklist"])
            target_law = checklist_data.get("target_law", "unknown")
            logger.info(
                f"Parsed checklist: {num_requirements} requirements from {target_law}"
            )

        except json.JSONDecodeError as e:
            error_msg = (
                f"ComplianceAgent error: Invalid JSON from requirement_extractor: {str(e)}. "
                f"RequirementExtractor must output valid JSON with checklist structure."
            )
            logger.error(error_msg)
            state["errors"] = state.get("errors", [])
            state["errors"].append(error_msg)
            return state
        except ValueError as e:
            error_msg = f"ComplianceAgent error: Invalid checklist structure: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", [])
            state["errors"].append(error_msg)
            return state

        logger.info(f"Running checklist-based compliance verification for {num_requirements} requirements...")

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
