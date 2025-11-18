"""
Requirement Extractor Agent - Atomic legal obligation extraction.

Responsibilities:
1. Extract atomic legal requirements from legal texts (laws, regulations, standards)
2. Decompose complex provisions into verifiable obligations
3. Classify requirements by granularity, severity, and applicability
4. Generate structured compliance checklist for downstream agents

Based on Legal AI Research (2024): Requirement-First Compliance Checking
"""

import json
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
            # Parse JSON with error handling
            parsed_checklist = None
            checklist_items = []
            parse_error = None

            try:
                # Attempt to parse JSON from final answer
                # LLM may wrap JSON in markdown code blocks, so handle that
                checklist_text = final_answer.strip()

                # Remove markdown code blocks if present
                if checklist_text.startswith("```json"):
                    checklist_text = checklist_text[7:]  # Remove ```json
                elif checklist_text.startswith("```"):
                    checklist_text = checklist_text[3:]  # Remove ```
                if checklist_text.endswith("```"):
                    checklist_text = checklist_text[:-3]  # Remove trailing ```

                checklist_text = checklist_text.strip()

                # Parse JSON
                parsed_checklist = json.loads(checklist_text)

                # Validate structure
                if "checklist" in parsed_checklist:
                    checklist_items = parsed_checklist["checklist"]
                    logger.info(f"Successfully parsed {len(checklist_items)} requirements from checklist")
                else:
                    parse_error = "JSON missing 'checklist' field"
                    logger.warning(f"Parsed JSON but missing 'checklist' field: {list(parsed_checklist.keys())}")

            except json.JSONDecodeError as e:
                parse_error = f"JSON parsing error: {str(e)}"
                logger.warning(f"Failed to parse requirement checklist as JSON: {e}")
            except Exception as e:
                parse_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error parsing requirement checklist: {e}", exc_info=True)

            # Build requirement extraction output
            requirement_extraction = {
                "checklist": checklist_items,  # Parsed list of requirements (empty if parse failed)
                "raw_answer": final_answer,  # Keep raw answer for debugging/fallback
                "parsed_successfully": parsed_checklist is not None and parse_error is None,
                "parse_error": parse_error,
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
