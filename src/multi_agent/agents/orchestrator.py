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
import time  # For LLM response time measurement
from typing import Any, Dict, List, Optional
import re

from ..core.agent_base import BaseAgent
from ..core.agent_initializer import initialize_agent
from ..core.agent_registry import register_agent
from ..core.event_bus import EventType
from ..core.state import QueryType, MultiAgentState
from src.agent.providers.factory import detect_provider_from_model

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

        # Initialize common components (provider, prompts, tools)
        components = initialize_agent(config, "orchestrator")
        self.provider = components.provider
        self.system_prompt = components.system_prompt

        # Routing configuration
        self.complexity_threshold_low = 30
        self.complexity_threshold_high = 70

        # Initialize orchestrator tools using registry (tier1 tools)
        from src.agent.tools import get_registry

        self.tool_registry = get_registry()

        # Get tool schemas for LLM (only document listing tool needed for routing)
        available_tools = ["get_document_list"]
        self.orchestrator_tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description or "No description",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            }
            for tool_name in available_tools
            if (tool := self.tool_registry.get_tool(tool_name))
        ]

        logger.info(f"OrchestratorAgent initialized with model: {config.model}, tools: {available_tools}")

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
            errors = state.get("errors", [])
            errors.append("No query provided for orchestration")
            return {"errors": errors}

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

            # Extract event bus from state (injected by runner)
            event_bus = state.get("event_bus")
            if event_bus:
                await event_bus.emit(
                    event_type=EventType.AGENT_START,
                    data={
                        "agent": "orchestrator",
                        "message": "Analyzing query complexity and routing..."
                    },
                    agent_name="orchestrator"
                )

            return await self._route_query(state)

    async def _route_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 1: Route query to appropriate agents.

        Uses unified LLM analysis for:
        - Follow-up detection and query rewriting
        - Complexity scoring
        - Vagueness scoring
        - Query type classification

        Args:
            state: Current workflow state with query

        Returns:
            Updated state with routing decision and unified analysis
        """
        query = state.get("query", "")
        conversation_history = state.get("conversation_history", [])
        original_query = query  # Store original for logging

        try:
            # UNIFIED: Single LLM call handles ALL analysis (follow-up, complexity, vagueness, type)
            # The orchestrator system prompt now includes UNIFIED QUERY ANALYSIS section
            routing_decision = await self._analyze_and_route(query, conversation_history)

            # Extract unified analysis from routing decision
            analysis = routing_decision.get("analysis", {})

            # Handle follow-up rewrite if LLM detected it
            if analysis.get("is_follow_up") and analysis.get("follow_up_rewrite"):
                rewritten = analysis["follow_up_rewrite"]
                if rewritten and rewritten != query and len(rewritten) > 10:
                    logger.info(f"LLM detected follow-up, rewritten: '{rewritten[:80]}...'")
                    # Update query in state so all downstream agents use the rewritten version
                    state["query"] = rewritten
                    state["original_query"] = original_query
                    query = rewritten

            # Update state with routing decision
            state["complexity_score"] = routing_decision["complexity_score"]
            state["query_type"] = routing_decision["query_type"]
            state["agent_sequence"] = routing_decision["agent_sequence"]

            # Store unified analysis for downstream use (HITL quality detector, etc.)
            state["unified_analysis"] = analysis

            # If orchestrator provided final_answer directly (for greetings/simple queries),
            # store it in state so runner can return it without building workflow
            if "final_answer" in routing_decision and routing_decision["final_answer"]:
                state["final_answer"] = routing_decision["final_answer"]
                logger.info("Orchestrator provided direct answer without agents")

            # Track orchestrator output (include unified analysis)
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["orchestrator"] = {
                "complexity_score": routing_decision["complexity_score"],
                "query_type": routing_decision["query_type"],
                "agent_sequence": routing_decision["agent_sequence"],
                "reasoning": routing_decision.get("reasoning", ""),
                "final_answer": routing_decision.get("final_answer"),
                "analysis": analysis  # Include unified analysis in output
            }

            # Log unified analysis results
            logger.info(
                f"Routing decision: complexity={routing_decision['complexity_score']}, "
                f"type={routing_decision['query_type']}, "
                f"sequence={routing_decision['agent_sequence']}, "
                f"is_follow_up={analysis.get('is_follow_up', False)}, "
                f"vagueness={analysis.get('vagueness_score', 'N/A')}"
            )

            # Return ONLY changed keys (partial update) to avoid LangGraph state conflicts
            update = {
                "complexity_score": routing_decision["complexity_score"],
                "query_type": routing_decision["query_type"],
                "agent_sequence": routing_decision["agent_sequence"],
                "agent_outputs": agent_outputs,
                "unified_analysis": analysis  # Include in state update
            }

            # Add query update if it was rewritten
            if state.get("original_query"):
                update["query"] = state["query"]
                update["original_query"] = state["original_query"]

            # Add final_answer if orchestrator provided direct response (no agents)
            if "final_answer" in routing_decision and routing_decision["final_answer"]:
                update["final_answer"] = routing_decision["final_answer"]

            return update

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

            errors = state.get("errors", [])
            errors.append(f"[{error_id}] Orchestration failed: {type(e).__name__}: {str(e)}")

            # Return ONLY changed keys (partial update)
            return {
                "errors": errors,
                "execution_phase": "error",
                "final_answer": (
                    f"Query analysis failed [{error_id}]. "
                    f"Unable to determine optimal workflow for your query. "
                    f"Error: {type(e).__name__}. "
                    f"Please check system configuration and try again."
                )
            }

    # NOTE: _rewrite_query_with_context has been REMOVED
    # Follow-up detection and query rewriting is now handled by unified analysis
    # in the orchestrator system prompt. The LLM returns analysis.is_follow_up
    # and analysis.follow_up_rewrite directly in the routing response.

    async def _analyze_and_route(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze query complexity and determine routing.

        Args:
            query: User query
            conversation_history: Previous messages for context

        Returns:
            Dict with:
                - complexity_score (int 0-100)
                - query_type (str)
                - agent_sequence (List[str])
                - reasoning (str)
        """
        # Build conversation context if available
        history_context = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-10:]:  # Last 10 messages max
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")[:500]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            history_context = "\n".join(history_lines)

        # Prepare analysis prompt with conversation context
        if history_context:
            user_message = f"""Analyze this query and provide routing decision:

CONVERSATION HISTORY (for context):
{history_context}

CURRENT QUERY: {query}

Use the conversation history to understand follow-up questions and resolve references.
Provide your analysis in the exact JSON format specified in the system prompt."""
        else:
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
                # Call provider's unified create_message API (synchronous) with timing
                llm_start_time = time.time()
                try:
                    response = self.provider.create_message(
                        messages=messages,
                        tools=self.orchestrator_tool_schemas,
                        system=system,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature
                    )
                    llm_response_time_ms = (time.time() - llm_start_time) * 1000
                except Exception as e:
                    # Measure time to failure - valuable for debugging timeout/performance issues
                    llm_response_time_ms = (time.time() - llm_start_time) * 1000
                    logger.error(
                        f"LLM call FAILED after {llm_response_time_ms:.2f}ms for orchestrator: {e}",
                        exc_info=True
                    )
                    # Re-raise to allow upstream error handling
                    raise

                # Track token usage and cost
                from src.cost_tracker import get_global_tracker
                tracker = get_global_tracker()

                # Extract usage from response
                if hasattr(response, 'usage'):
                    usage = response.usage
                    # BUGFIX: usage is a dict, not object - use .get() instead of getattr()
                    input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
                    # Correct field names for Anthropic API
                    cache_read_tokens = usage.get('cache_read_input_tokens', 0)
                    cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)

                    # Use SSOT provider detection
                    try:
                        provider = detect_provider_from_model(self.config.model)
                    except ValueError:
                        provider = 'anthropic'  # Fallback

                    tracker.track_llm(
                        provider=provider,
                        model=self.config.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        operation="agent_orchestrator",
                        cache_creation_tokens=cache_creation_tokens,
                        cache_read_tokens=cache_read_tokens,
                        response_time_ms=llm_response_time_ms
                    )
                    logger.debug(f"Tracked orchestrator routing: {input_tokens} in, {output_tokens} out")
                else:
                    # Warn when response time is measured but not tracked
                    logger.warning(
                        f"Response missing usage data for orchestrator - "
                        f"response time ({llm_response_time_ms:.2f}ms) not tracked in cost tracker. "
                        f"Response type: {type(response)}, has 'usage' attr: {hasattr(response, 'usage')}"
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

                        # Emit tool start event
                        event_bus = None
                        # Try to get event bus from self (not stored) or context?
                        # Since _analyze_and_route doesn't have state, we can't easily access event_bus here
                        # UNLESS we pass it down.
                        # But for now, let's rely on the initial AGENT_START.
                        # To do this properly, we'd need to refactor _analyze_and_route to accept event_bus.
                        # Let's skip tool events for now in orchestrator to keep changes minimal and safe.

                        # Execute tool using registry
                        try:
                            # Get tool from registry
                            tool = self.tool_registry.get_tool(tool_name)

                            if not tool:
                                logger.warning(f"Unknown tool: {tool_name}")
                                result = {"error": f"Unknown tool: {tool_name}"}
                            else:
                                # Execute tool and convert ToolResult to dict
                                tool_result = tool.execute(**tool_input)

                                if tool_result.success:
                                    result = tool_result.data
                                else:
                                    result = {"error": tool_result.error}

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.get("id"),
                                "content": json.dumps(result)
                            })
                            logger.debug(f"Tool {tool_name} result: {result}")
                        except Exception as e:
                            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
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
        conversation_history = state.get("conversation_history", [])

        logger.info(f"Synthesizing final answer from {len(agent_outputs)} agent outputs...")

        try:
            # Build synthesis context from agent outputs
            synthesis_context = self._build_synthesis_context(state)

            # Build conversation context if available
            history_context = ""
            if conversation_history:
                history_lines = []
                for msg in conversation_history[-10:]:  # Last 10 messages max
                    role = msg.get("role", "unknown").upper()
                    content = msg.get("content", "")[:500]  # Truncate long messages
                    history_lines.append(f"{role}: {content}")
                history_context = "\n".join(history_lines)

            # Prepare synthesis prompt with optional conversation context
            if history_context:
                user_message = f"""Generate final answer for this query using agent outputs.

CONVERSATION HISTORY (for context):
{history_context}

CURRENT QUERY: {query}
Complexity Score: {complexity_score}

Agent Outputs:
{synthesis_context}

Use the conversation history to maintain continuity and reference previous topics if relevant.
Generate final answer following the synthesis instructions in your system prompt.
Ensure language matching and proper citations."""
            else:
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

            # Call LLM for synthesis with timing
            import time
            llm_start_time = time.time()
            response = self.provider.create_message(
                messages=[{"role": "user", "content": user_message}],
                tools=None,  # No tools in synthesis phase
                system=system,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            llm_response_time_ms = (time.time() - llm_start_time) * 1000

            # Track token usage and cost for synthesis
            from src.cost_tracker import get_global_tracker
            tracker = get_global_tracker()

            if hasattr(response, 'usage'):
                usage = response.usage
                # BUGFIX: usage is a dict, not object - use .get() instead of getattr()
                input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
                output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
                # Correct field names for Anthropic API
                cache_read_tokens = usage.get('cache_read_input_tokens', 0)
                cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)

                # Use SSOT provider detection
                try:
                    provider = detect_provider_from_model(self.config.model)
                except ValueError:
                    provider = 'anthropic'  # Fallback

                tracker.track_llm(
                    provider=provider,
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation="agent_orchestrator",
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                    response_time_ms=llm_response_time_ms
                )
                logger.debug(f"Tracked orchestrator synthesis: {input_tokens} in, {output_tokens} out")

            final_answer = response.text

            # Check if orchestrator requested iteration (JSON response)
            needs_iteration = False
            next_agents = []
            iteration_reason = ""
            partial_answer = ""

            # Get complexity score to check if iteration is allowed
            complexity_score = state.get("complexity_score", 50)

            try:
                # Try parsing response as JSON (iteration request)
                import json
                iteration_request = json.loads(final_answer.strip())

                if iteration_request.get("needs_iteration"):
                    # SAFEGUARD: Block iteration for simple queries (complexity < 40)
                    # This prevents unnecessary double execution of agents
                    if complexity_score < 40:
                        logger.warning(
                            f"Iteration BLOCKED for simple query (complexity={complexity_score}). "
                            f"Requested agents: {iteration_request.get('next_agents', [])}. "
                            f"Using partial answer instead."
                        )
                        # Use partial answer as final
                        partial_answer = iteration_request.get("partial_answer", "")
                        if partial_answer:
                            final_answer = partial_answer
                        # Don't set needs_iteration = True
                    else:
                        needs_iteration = True
                        next_agents = iteration_request.get("next_agents", [])
                        iteration_reason = iteration_request.get("iteration_reason", "")
                        partial_answer = iteration_request.get("partial_answer", "")

                        logger.info(
                            f"Orchestrator requested iteration: {len(next_agents)} agents "
                            f"({', '.join(next_agents)}) - Reason: {iteration_reason}"
                        )

                        # Replace final_answer with partial answer for this iteration
                        final_answer = partial_answer
            except (json.JSONDecodeError, ValueError):
                # Not JSON - normal final answer
                pass

            # Aggregate costs from ALL agents (stored in state for SSE event emission)
            total_cost_usd = tracker.get_total_cost()

            # Build agent_outputs update (preserve existing outputs)
            agent_outputs = state.get("agent_outputs", {})
            if "orchestrator" not in agent_outputs:
                agent_outputs["orchestrator"] = {}
            agent_outputs["orchestrator"]["synthesis"] = {
                "final_answer": final_answer,
                "total_cost_usd": total_cost_usd
            }

            # Return ONLY changed keys (partial update) to avoid LangGraph state conflicts
            update = {
                "final_answer": final_answer,
                "agent_outputs": agent_outputs
            }

            # Add iteration flags if orchestrator requested it
            if needs_iteration:
                iteration_count = state.get("iteration_count", 0) + 1
                update.update({
                    "needs_iteration": True,
                    "next_agents": next_agents,
                    "iteration_reason": iteration_reason,
                    "iteration_count": iteration_count
                })

                logger.info(
                    f"Synthesis complete with iteration request (iteration {iteration_count}): "
                    f"{len(next_agents)} agents - Reason: {iteration_reason}"
                )
            else:
                update["needs_iteration"] = False
                logger.info(f"Synthesis complete: {len(final_answer)} chars, cost=${total_cost_usd:.6f}")

            return update

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

            errors = state.get("errors", [])
            errors.append(f"[{error_id}] Synthesis failed: {type(e).__name__}: {str(e)}")

            # Return ONLY changed keys (partial update)
            return {
                "errors": errors,
                "final_answer": (
                    f"Answer generation failed [{error_id}]. "
                    f"Agents completed their analysis, but final synthesis encountered an error. "
                    f"Error: {type(e).__name__}. "
                    f"Please try rephrasing your query."
                )
            }

    # Debug/metadata fields to exclude from synthesis context (token optimization)
    _EXCLUDE_FIELDS = {
        "iterations", "retrieval_method", "total_tool_cost_usd",
        "tool_calls_made", "chunks", "expanded_results", "_internal"
    }

    def _build_synthesis_context(self, state: Dict[str, Any], compress: bool = False) -> str:
        """
        Build synthesis context from agent outputs.

        Token-optimized: Filters debug/metadata fields and uses compact JSON.
        Expected savings: ~25-30% compared to full JSON dumps (or ~60-70% with compression).

        Args:
            state: Current workflow state
            compress: If True, use aggressive compression for downstream agents
                     (~100-200 tokens per agent instead of ~500-1000)

        Returns:
            Formatted string with agent outputs (compact JSON, no indent)
        """
        from ..core.agent_base import BaseAgent

        agent_outputs = state.get("agent_outputs", {})
        context_parts = []

        # Order agents by execution sequence (if available)
        agent_sequence = state.get("agent_sequence", [])

        def filter_output(output: dict) -> dict:
            """Keep all content fields, exclude only debug/metadata."""
            return {k: v for k, v in output.items() if k not in self._EXCLUDE_FIELDS}

        def process_output(output: dict) -> dict:
            """Process output based on compression setting."""
            if compress:
                return BaseAgent.compress_output_for_downstream(output)
            return filter_output(output)

        # Add outputs in sequence order
        for agent_name in agent_sequence:
            if agent_name in agent_outputs and agent_name != "orchestrator":
                output = agent_outputs[agent_name]
                processed = process_output(output)
                context_parts.append(f"### {agent_name.upper()} OUTPUT:")
                # Compact JSON: no indent, minimal separators
                context_parts.append(json.dumps(processed, ensure_ascii=False, separators=(',', ':')))
                context_parts.append("")

        # Add any remaining outputs not in sequence
        for agent_name, output in agent_outputs.items():
            if agent_name != "orchestrator" and agent_name not in agent_sequence:
                processed = process_output(output)
                context_parts.append(f"### {agent_name.upper()} OUTPUT:")
                context_parts.append(json.dumps(processed, ensure_ascii=False, separators=(',', ':')))
                context_parts.append("")

        return "\n".join(context_parts)

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

        # Validate and normalize query type
        valid_types = ["simple_search", "cross_doc", "compliance", "risk", "synthesis", "reporting", "unknown"]
        query_type = decision["query_type"]
        if query_type not in valid_types:
            logger.warning(
                f"Invalid query_type '{query_type}' from LLM, converting to 'unknown'. "
                f"Expected one of {valid_types}"
            )
            # Normalize invalid query_type to 'unknown' (defensive programming)
            # This handles LLM returning values like 'greeting' which aren't in QueryType enum
            decision["query_type"] = "unknown"

        # Validate unified analysis structure if present (graceful handling)
        if "analysis" in decision:
            analysis = decision["analysis"]

            # Validate vagueness_score range and clamp if needed
            if "vagueness_score" in analysis:
                score = analysis["vagueness_score"]
                if not isinstance(score, (int, float)):
                    logger.warning(f"Invalid vagueness_score type: {type(score)}, setting to 0.5")
                    analysis["vagueness_score"] = 0.5
                elif score < 0.0 or score > 1.0:
                    logger.warning(f"vagueness_score {score} out of range, clamping to [0,1]")
                    analysis["vagueness_score"] = max(0.0, min(1.0, float(score)))

            # Ensure boolean fields are actually booleans
            for bool_field in ["is_follow_up", "needs_clarification"]:
                if bool_field in analysis and not isinstance(analysis[bool_field], bool):
                    analysis[bool_field] = bool(analysis[bool_field])

            # Validate semantic_type if present
            valid_semantic_types = [
                "greeting", "specific_factual", "comparative",
                "procedural", "analytical", "compliance_check"
            ]
            if "semantic_type" in analysis and analysis["semantic_type"] not in valid_semantic_types:
                logger.warning(f"Invalid semantic_type '{analysis['semantic_type']}', defaulting to 'specific_factual'")
                analysis["semantic_type"] = "specific_factual"
        else:
            # If LLM didn't return analysis, create empty default (backwards compatibility)
            decision["analysis"] = {
                "is_follow_up": False,
                "follow_up_rewrite": None,
                "vagueness_score": 0.5,
                "needs_clarification": False,
                "semantic_type": "specific_factual"
            }
            logger.debug("LLM didn't return analysis field, using defaults")

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
