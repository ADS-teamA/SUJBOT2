"""
Orchestrator Agent - Root coordinator for multi-agent workflow.

Responsibilities:
1. Query complexity analysis (0-100 scoring)
2. Query type classification (compliance, risk, synthesis, search, reporting)
3. Agent sequence determination based on routing rules
4. Workflow pattern selection (Simple, Standard, Complex)
"""

import json
import logging
from typing import Any, Dict, List, Optional
import re

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.state import QueryType, MultiAgentState
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader

logger = logging.getLogger(__name__)


@register_agent("orchestrator")
class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Analyzes query complexity and routes to appropriate agents.

    Uses LLM to analyze query characteristics and determine optimal agent sequence
    based on complexity scoring rubric and routing rules.
    """

    def __init__(self, config):
        """Initialize orchestrator with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("orchestrator")

        # Routing configuration
        self.complexity_threshold_low = 30
        self.complexity_threshold_high = 70

        logger.info(f"OrchestratorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query and determine routing.

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with:
                - complexity_score (0-100)
                - query_type (QueryType enum)
                - agent_sequence (List[str])
        """
        query = state.get("query", "")

        if not query:
            logger.error("No query provided in state")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No query provided for orchestration")
            return state

        logger.info(f"Analyzing query: {query[:100]}...")

        try:
            # Call LLM for complexity analysis and routing
            routing_decision = await self._analyze_and_route(query)

            # Update state with routing decision
            state["complexity_score"] = routing_decision["complexity_score"]
            state["query_type"] = routing_decision["query_type"]
            state["agent_sequence"] = routing_decision["agent_sequence"]

            # Track orchestrator output
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["orchestrator"] = {
                "complexity_score": routing_decision["complexity_score"],
                "query_type": routing_decision["query_type"],
                "agent_sequence": routing_decision["agent_sequence"],
                "reasoning": routing_decision.get("reasoning", "")
            }

            logger.info(
                f"Routing decision: complexity={routing_decision['complexity_score']}, "
                f"type={routing_decision['query_type']}, "
                f"sequence={routing_decision['agent_sequence']}"
            )

            return state

        except Exception as e:
            from ..core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.CRITICAL,
                agent_name="orchestrator",
                context={"query": query[:200]}
            )

            logger.error(
                f"[{error_id}] Orchestration failed: {type(e).__name__}: {e}. "
                f"Check: (1) Anthropic/OpenAI API key is valid, (2) model name is correct, "
                f"(3) prompt is under token limit, (4) network connection is stable.",
                exc_info=True
            )

            state["errors"] = state.get("errors", [])
            state["errors"].append(f"[{error_id}] Orchestration failed: {type(e).__name__}: {str(e)}")

            # DO NOT silently fall back - user must know orchestration failed
            state["execution_phase"] = "error"
            state["final_answer"] = (
                f"Query analysis failed [{error_id}]. "
                f"Unable to determine optimal workflow for your query. "
                f"Error: {type(e).__name__}. "
                f"Please check system configuration and try again."
            )

            return state

    async def _analyze_and_route(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query complexity and determine routing.

        Args:
            query: User query

        Returns:
            Dict with:
                - complexity_score (int 0-100)
                - query_type (str)
                - agent_sequence (List[str])
                - reasoning (str)
        """
        # Prepare analysis prompt
        user_message = f"""Analyze this query and provide routing decision:

Query: {query}

Provide your analysis in the exact JSON format specified in the system prompt."""

        # Call Anthropic API
        try:
            # Prepare API parameters
            api_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.system_prompt,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            }

            # Add prompt caching if enabled
            if self.config.enable_prompt_caching:
                api_params["system"] = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

            response = self.client.messages.create(**api_params)

            # Extract text response
            response_text = response.content[0].text

            # Parse JSON response
            routing_decision = self._parse_routing_response(response_text)

            # Validate routing decision
            self._validate_routing_decision(routing_decision)

            return routing_decision

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            raise

    def _parse_routing_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract routing decision.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed routing decision dict
        """
        try:
            # Try to extract JSON from response
            # Look for JSON block in markdown code fence or plain text
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            # Parse JSON
            routing_decision = json.loads(json_str)

            return routing_decision

        except Exception as e:
            logger.error(f"Failed to parse routing response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Could not parse routing decision: {e}")

    def _validate_routing_decision(self, decision: Dict[str, Any]) -> None:
        """
        Validate routing decision has required fields.

        Args:
            decision: Routing decision dict

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["complexity_score", "query_type", "agent_sequence"]

        for field in required_fields:
            if field not in decision:
                raise ValueError(f"Missing required field: {field}")

        # Validate complexity score range
        score = decision["complexity_score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            raise ValueError(f"Invalid complexity_score: {score} (must be 0-100)")

        # Validate agent sequence is non-empty list
        sequence = decision["agent_sequence"]
        if not isinstance(sequence, list) or len(sequence) == 0:
            raise ValueError(f"Invalid agent_sequence: {sequence} (must be non-empty list)")

        # Validate query type
        valid_types = ["compliance", "risk", "synthesis", "search", "reporting"]
        query_type = decision["query_type"]
        if query_type not in valid_types:
            logger.warning(
                f"Unknown query_type: {query_type}, expected one of {valid_types}"
            )

    def get_workflow_pattern(self, complexity_score: int) -> str:
        """
        Determine workflow pattern based on complexity.

        Args:
            complexity_score: Complexity score (0-100)

        Returns:
            Workflow pattern name
        """
        if complexity_score < self.complexity_threshold_low:
            return "simple"
        elif complexity_score < self.complexity_threshold_high:
            return "standard"
        else:
            return "complex"

    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities of each agent for routing decisions.

        Returns:
            Dict mapping agent names to their capabilities/keywords
        """
        return {
            "extractor": ["retrieve", "find", "search", "locate", "get"],
            "classifier": ["categorize", "organize", "classify", "type", "sort"],
            "compliance": ["compliance", "gdpr", "ccpa", "hipaa", "regulation", "legal"],
            "risk_verifier": ["risk", "safety", "liability", "impact", "assess"],
            "citation_auditor": ["citation", "source", "reference", "verify", "validate"],
            "gap_synthesizer": ["gap", "missing", "complete", "comprehensive", "coverage"],
            "report_generator": ["report", "summary", "compile", "generate", "final"]
        }

    def suggest_agents_for_query(self, query: str) -> List[str]:
        """
        Suggest agents based on keyword matching (fallback method).

        Args:
            query: User query

        Returns:
            List of suggested agent names
        """
        query_lower = query.lower()
        capabilities = self.get_agent_capabilities()

        suggested = []

        # Check for keyword matches
        for agent, keywords in capabilities.items():
            if any(keyword in query_lower for keyword in keywords):
                if agent not in suggested:
                    suggested.append(agent)

        # Always include extractor at start (unless already present)
        if "extractor" not in suggested:
            suggested.insert(0, "extractor")

        # Always include report_generator at end (unless already present)
        if "report_generator" not in suggested:
            suggested.append("report_generator")

        return suggested
