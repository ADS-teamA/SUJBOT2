"""
Orchestrator Agent - Single point of user communication.

Responsibilities (Dual-Phase):
PHASE 1 - ROUTING:
1. Query complexity analysis (0-100 scoring)
2. Query type classification (compliance, risk, synthesis, search, reporting)
3. Agent sequence determination based on routing rules

PHASE 2 - SYNTHESIS:
4. Final answer generation from agent outputs
5. Citation integration and language matching
6. Cost tracking and report formatting
"""

import json
import logging
from typing import Any, Dict, List, Optional
import re

from ..core.agent_base import BaseAgent
from ..core.state import QueryType, MultiAgentState
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader

logger = logging.getLogger(__name__)


@register_agent("orchestrator")
class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Single point of user communication (dual-phase).

    PHASE 1 (Routing): Analyzes query complexity and routes to appropriate agents.
    PHASE 2 (Synthesis): Generates final answer from agent outputs.

    Uses LLM to analyze query characteristics and determine optimal agent sequence
    based on complexity scoring rubric and routing rules. After agents complete,
    synthesizes their outputs into final user-facing answer.
    """

    def __init__(self, config, vector_store=None, agent_registry=None):
        """Initialize orchestrator with config."""
        super().__init__(config)

        # Initialize provider (auto-detects from model name: claude/gpt/gemini)
        # Provider factory loads API keys from environment variables
        try:
            from src.agent.providers.factory import create_provider

            # Provider factory auto-detects provider from model name
            # and loads appropriate API key from environment
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
        self.system_prompt = prompt_loader.get_prompt("orchestrator")

        # Routing configuration
        self.complexity_threshold_low = 30
        self.complexity_threshold_high = 70

        # Initialize orchestrator-specific tools
        from ..tools.orchestrator_tools import create_orchestrator_tools
        self.orchestrator_tool_schemas, self.orchestrator_tools_instance = create_orchestrator_tools(
            vector_store=vector_store,
            agent_registry=agent_registry
        )

        logger.info(f"OrchestratorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dual-phase execution: Routing OR Synthesis.

        Detects phase based on state contents:
        - ROUTING: state has no agent_outputs (first call)
        - SYNTHESIS: state has agent_outputs (second call after agents complete)

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with routing decision OR final answer
        """
        query = state.get("query", "")

        if not query:
            logger.error("No query provided in state")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No query provided for orchestration")
            return state

        # Detect phase: if agent_outputs exist (and not just orchestrator), we're in SYNTHESIS phase
        agent_outputs = state.get("agent_outputs", {})
        has_non_orchestrator_outputs = any(
            agent_name != "orchestrator" for agent_name in agent_outputs.keys()
        )

        if has_non_orchestrator_outputs:
            # PHASE 2: SYNTHESIS - generate final answer from agent outputs
            logger.info("PHASE 2: Synthesis - generating final answer from agent outputs")
            return await self._synthesize_final_answer(state)
        else:
            # PHASE 1: ROUTING - analyze query and determine agent sequence
            logger.info(f"PHASE 1: Routing - analyzing query: {query[:100]}...")
            return await self._route_query(state)

    async def _route_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 1: Route query to appropriate agents.

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with routing decision
        """
        query = state.get("query", "")

        try:
            # Call LLM for complexity analysis and routing
            routing_decision = await self._analyze_and_route(query)

            # Update state with routing decision
            state["complexity_score"] = routing_decision["complexity_score"]
            state["query_type"] = routing_decision["query_type"]
            state["agent_sequence"] = routing_decision["agent_sequence"]

            # If orchestrator provided final_answer directly (for greetings/simple queries),
            # store it in state so runner can return it without building workflow
            if "final_answer" in routing_decision and routing_decision["final_answer"]:
                state["final_answer"] = routing_decision["final_answer"]
                logger.info("Orchestrator provided direct answer without agents")

            # Track orchestrator output
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["orchestrator"] = {
                "complexity_score": routing_decision["complexity_score"],
                "query_type": routing_decision["query_type"],
                "agent_sequence": routing_decision["agent_sequence"],
                "reasoning": routing_decision.get("reasoning", ""),
                "final_answer": routing_decision.get("final_answer")
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

        # Call LLM provider (Anthropic/OpenAI/Google via unified interface)
        try:
            # Prepare system prompt (with caching if enabled)
            if self.config.enable_prompt_caching:
                # Anthropic-style prompt caching
                system = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                system = self.system_prompt

            # Tool execution loop - orchestrator may call tools before making routing decision
            messages = [{"role": "user", "content": user_message}]
            max_tool_iterations = 3  # Prevent infinite loops

            for iteration in range(max_tool_iterations):
                # Call provider's unified create_message API (async)
                response = await self.provider.create_message(
                    messages=messages,
                    tools=self.orchestrator_tool_schemas,
                    system=system,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )

                # Log response structure for debugging
                content_types = [block.get("type") if isinstance(block, dict) else getattr(block, "type", "unknown")
                                for block in response.content]
                logger.info(f"Iteration {iteration}: stop_reason={response.stop_reason}, content_blocks={content_types}, text_length={len(response.text)}")

                # Check if LLM wants to use tools
                if response.stop_reason == "tool_use":
                    # Extract tool calls from response (content is list of dicts)
                    tool_calls = [
                        block for block in response.content
                        if isinstance(block, dict) and block.get("type") == "tool_use"
                    ]

                    if not tool_calls:
                        # No actual tool calls despite stop_reason - treat as text response
                        response_text = response.text if hasattr(response, 'text') else ""
                        logger.debug(f"Empty tool_calls list, extracted text: {response_text[:200]}")
                        break

                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Execute tools and collect results
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        tool_input = tool_call.get("input", {})

                        logger.info(f"Orchestrator calling tool: {tool_name}")

                        # Execute tool using tools instance
                        try:
                            if tool_name == "list_available_documents":
                                result = self.orchestrator_tools_instance.list_available_documents()
                            elif tool_name == "list_available_agents":
                                result = self.orchestrator_tools_instance.list_available_agents()
                            else:
                                logger.warning(f"Unknown tool: {tool_name}")
                                result = {"error": f"Unknown tool: {tool_name}"}

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.get("id"),
                                "content": json.dumps(result)
                            })
                            logger.debug(f"Tool {tool_name} result: {result}")
                        except Exception as e:
                            logger.error(f"Tool {tool_name} failed: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.get("id"),
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True
                            })

                    # Add tool results to conversation
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    # Continue loop to get next response
                    continue

                # LLM provided text response - extract routing decision
                response_text = response.text
                logger.debug(f"Text response received: {response_text[:200]}")
                break

            else:
                # Max iterations reached without final answer
                raise ValueError(
                    f"Orchestrator exceeded max tool iterations ({max_tool_iterations}). "
                    f"LLM did not provide routing decision."
                )

            # Parse JSON response
            routing_decision = self._parse_routing_response(response_text)

            # Validate routing decision
            self._validate_routing_decision(routing_decision)

            return routing_decision

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            raise

    async def _synthesize_final_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 2: Synthesize final answer from agent outputs.

        Args:
            state: Current workflow state with agent_outputs

        Returns:
            Updated state with final_answer
        """
        query = state.get("query", "")
        agent_outputs = state.get("agent_outputs", {})
        complexity_score = state.get("complexity_score", 50)

        logger.info(f"Synthesizing final answer from {len(agent_outputs)} agent outputs...")

        try:
            # Build synthesis context from agent outputs
            synthesis_context = self._build_synthesis_context(state)

            # Prepare synthesis prompt
            user_message = f"""Generate final answer for this query using agent outputs.

Original Query: {query}
Complexity Score: {complexity_score}

Agent Outputs:
{synthesis_context}

Generate final answer following the synthesis instructions in your system prompt.
Ensure language matching and proper citations."""

            # Call LLM for synthesis (no tool calling in synthesis phase)
            if self.config.enable_prompt_caching:
                system = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                system = self.system_prompt

            response = await self.provider.create_message(
                messages=[{"role": "user", "content": user_message}],
                tools=None,  # No tools in synthesis phase
                system=system,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            final_answer = response.text

            # Aggregate costs from ALL agents
            from src.cost_tracker import get_global_tracker
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()

            # Format cost summary (append to final answer)
            cost_summary = self._format_cost_summary(total_cost_usd, agent_outputs)
            final_answer_with_cost = f"{final_answer}\n\n{cost_summary}"

            # Update state
            state["final_answer"] = final_answer_with_cost
            state["agent_outputs"]["orchestrator"]["synthesis"] = {
                "final_answer": final_answer,
                "total_cost_usd": total_cost_usd
            }

            logger.info(f"Synthesis complete: {len(final_answer)} chars, cost=${total_cost_usd:.6f}")

            return state

        except Exception as e:
            from ..core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                agent_name="orchestrator",
                context={"query": query[:200], "num_agents": len(agent_outputs)}
            )

            logger.error(
                f"[{error_id}] Synthesis failed: {type(e).__name__}: {e}",
                exc_info=True
            )

            state["errors"] = state.get("errors", [])
            state["errors"].append(f"[{error_id}] Synthesis failed: {type(e).__name__}: {str(e)}")

            # Provide fallback answer
            state["final_answer"] = (
                f"Answer generation failed [{error_id}]. "
                f"Agents completed their analysis, but final synthesis encountered an error. "
                f"Error: {type(e).__name__}. "
                f"Please try rephrasing your query."
            )

            return state

    def _build_synthesis_context(self, state: Dict[str, Any]) -> str:
        """
        Build synthesis context from agent outputs.

        Args:
            state: Current workflow state

        Returns:
            Formatted string with agent outputs
        """
        agent_outputs = state.get("agent_outputs", {})
        context_parts = []

        # Order agents by execution sequence (if available)
        agent_sequence = state.get("agent_sequence", [])

        # Add outputs in sequence order
        for agent_name in agent_sequence:
            if agent_name in agent_outputs and agent_name != "orchestrator":
                output = agent_outputs[agent_name]
                context_parts.append(f"### {agent_name.upper()} OUTPUT:")
                context_parts.append(json.dumps(output, indent=2, ensure_ascii=False))
                context_parts.append("")

        # Add any remaining outputs not in sequence
        for agent_name, output in agent_outputs.items():
            if agent_name != "orchestrator" and agent_name not in agent_sequence:
                context_parts.append(f"### {agent_name.upper()} OUTPUT:")
                context_parts.append(json.dumps(output, indent=2, ensure_ascii=False))
                context_parts.append("")

        return "\n".join(context_parts)

    def _format_cost_summary(self, total_cost: float, agent_outputs: dict) -> str:
        """
        Format cost summary for display to user.

        Args:
            total_cost: Total workflow cost in USD
            agent_outputs: Dict mapping agent names to outputs

        Returns:
            Formatted markdown cost summary
        """
        lines = [
            "---",
            "## 💰 API Cost Summary",
            f"**Total Workflow Cost:** ${total_cost:.6f}",
            ""
        ]

        # Per-agent breakdown (if available)
        agent_costs = {}
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and "total_tool_cost_usd" in output:
                agent_cost = output["total_tool_cost_usd"]
                if agent_cost > 0:
                    agent_costs[agent_name] = agent_cost

        if agent_costs:
            lines.append("**Per-Agent Breakdown:**")
            # Sort by cost descending
            sorted_agents = sorted(agent_costs.items(), key=lambda x: x[1], reverse=True)
            for agent_name, cost in sorted_agents:
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
            logger.error(f"Response text was: {response_text[:500]}")  # Log first 500 chars
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

        # Validate agent sequence is list (can be empty for direct answers)
        sequence = decision["agent_sequence"]
        if not isinstance(sequence, list):
            raise ValueError(f"Invalid agent_sequence: {sequence} (must be a list)")

        # If agent_sequence is empty, final_answer must be provided
        if len(sequence) == 0 and "final_answer" not in decision:
            raise ValueError(
                "Empty agent_sequence requires final_answer field "
                "(for direct responses without agent pipeline)"
            )

        # Validate query type
        valid_types = ["simple_search", "cross_doc", "compliance", "risk", "synthesis", "reporting", "unknown"]
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

    # Removed rule-based fallback methods (get_agent_capabilities, suggest_agents_for_query)
    # All routing decisions are now made autonomously by the LLM orchestrator
