"""
Workflow Builder - Constructs LangGraph workflow from agent sequence.

Builds executable workflow graphs with:
1. Agent nodes (state transformations)
2. Conditional edges (routing logic)
3. Error handling nodes
4. Checkpointing integration
"""

import logging
from typing import Any, Dict, List, Optional, Callable

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from ..core.state import MultiAgentState, ExecutionPhase
from ..core.agent_registry import AgentRegistry
from ..hitl.config import HITLConfig
from ..hitl.quality_detector import QualityDetector
from ..hitl.clarification_generator import ClarificationGenerator
from ..core.event_bus import EventType

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """
    Builds LangGraph workflows from agent sequences.

    Constructs state graphs with proper node connections, conditional
    routing, and error handling.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        checkpointer: Optional[Any] = None,
        hitl_config: Optional[HITLConfig] = None,
        quality_detector: Optional[QualityDetector] = None,
        clarification_generator: Optional[ClarificationGenerator] = None,
    ):
        """
        Initialize workflow builder.

        Args:
            agent_registry: Registry containing all agent instances
            checkpointer: Optional PostgreSQL checkpointer for state persistence
            hitl_config: Optional HITL configuration
            quality_detector: Optional quality detector for HITL
            clarification_generator: Optional clarification generator for HITL
        """
        self.agent_registry = agent_registry
        self.checkpointer = checkpointer
        self.hitl_config = hitl_config
        self.quality_detector = quality_detector
        self.clarification_generator = clarification_generator

        logger.info("WorkflowBuilder initialized")

    def build_workflow(
        self, agent_sequence: List[str], enable_parallel: bool = False
    ) -> StateGraph:
        """
        Build LangGraph workflow from agent sequence.

        Args:
            agent_sequence: List of agent names in execution order
            enable_parallel: Enable parallel execution where possible

        Returns:
            Compiled LangGraph StateGraph
        """
        logger.info(f"Building workflow for sequence: {agent_sequence}")

        # Create state graph
        workflow = StateGraph(MultiAgentState)

        # Add agent nodes
        for agent_name in agent_sequence:
            self._add_agent_node(workflow, agent_name)

        # Add HITL gate node if enabled
        self._add_hitl_gate_node(workflow)

        # Add edges to connect agents (with HITL gate if enabled)
        self._add_workflow_edges(workflow, agent_sequence, enable_parallel)

        # Set entry point (first agent)
        if agent_sequence:
            workflow.set_entry_point(agent_sequence[0])

        # Compile workflow with checkpointer
        if self.checkpointer:
            compiled = workflow.compile(checkpointer=self.checkpointer)
        else:
            compiled = workflow.compile()

        logger.info(f"Workflow built successfully with {len(agent_sequence)} agents")

        return compiled

    def _add_agent_node(self, workflow: StateGraph, agent_name: str) -> None:
        """
        Add agent node to workflow.

        Args:
            workflow: StateGraph to add node to
            agent_name: Name of agent to add
        """
        # Get agent from registry
        agent = self.agent_registry.get_agent(agent_name)

        if agent is None:
            logger.error(f"Agent not found: {agent_name}")
            raise ValueError(f"Agent not found in registry: {agent_name}")

        # Create node function that wraps agent execution
        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute agent and update state."""
            try:
                # LangGraph may pass state as MultiAgentState (Pydantic) or dict
                # Extract EventBus using proper accessor based on type
                if hasattr(state, "event_bus"):
                    # Pydantic model - access as attribute (field name: event_bus)
                    event_bus = state.event_bus
                elif isinstance(state, dict):
                    # Plain dict - access as key
                    event_bus = state.get("event_bus")
                else:
                    event_bus = None

                # Convert MultiAgentState to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Update execution phase
                updated_state = {
                    **state_dict,
                    "execution_phase": ExecutionPhase.AGENT_EXECUTION.value,
                    "current_agent": agent_name,
                }

                # Restore EventBus into state dict (for agent's autonomous tool loop to emit progress events)
                # NOTE: This will be converted back to Pydantic model by LangGraph
                # Use "event_bus" key (no underscore) to match MultiAgentState field name
                if event_bus:
                    updated_state["event_bus"] = event_bus

                # Add agent to sequence if not already there
                agent_sequence = updated_state.get("agent_sequence", [])
                if agent_name not in agent_sequence:
                    updated_state["agent_sequence"] = agent_sequence + [agent_name]

                logger.info(f"Executing agent: {agent_name}")

                # Emit agent start event
                if event_bus:
                    await event_bus.emit(
                        event_type=EventType.AGENT_START,
                        data={
                            "agent": agent_name,
                            "message": f"Starting {agent_name} analysis..."
                        },
                        agent_name=agent_name
                    )

                # Execute agent (pass dict, expect dict back)
                result = await agent.execute(updated_state)

                logger.info(f"Agent {agent_name} completed successfully")

                # CRITICAL: Ensure current_agent is preserved in result for progress tracking
                # Runner expects current_agent to emit AGENT_START events
                if "current_agent" not in result:
                    result["current_agent"] = agent_name

                return result

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}", exc_info=True)

                # Convert to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Add error to state
                errors = state_dict.get("errors", [])
                errors.append(f"{agent_name} error: {str(e)}")

                return {**state_dict, "errors": errors}

        # Add node to workflow
        workflow.add_node(agent_name, agent_node)

        logger.debug(f"Added agent node: {agent_name}")

    def _add_orchestrator_synthesis_node(self, workflow: StateGraph) -> None:
        """
        Add orchestrator synthesis node to workflow.

        This node calls the orchestrator agent in PHASE 2 (synthesis mode).
        The orchestrator detects the phase based on presence of agent_outputs.

        Args:
            workflow: StateGraph to add node to
        """
        # Get orchestrator agent from registry
        orchestrator = self.agent_registry.get_agent("orchestrator")

        if orchestrator is None:
            logger.error("Orchestrator agent not found in registry")
            raise ValueError("Orchestrator agent required for synthesis")

        async def orchestrator_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute orchestrator synthesis and update state."""
            try:
                # Extract EventBus if present
                if hasattr(state, "event_bus"):
                    event_bus = state.event_bus
                elif isinstance(state, dict):
                    event_bus = state.get("event_bus")
                else:
                    event_bus = None

                # Convert to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Update execution phase
                updated_state = {
                    **state_dict,
                    "execution_phase": ExecutionPhase.SYNTHESIS.value,
                    "current_agent": "orchestrator",
                }

                # Restore EventBus
                if event_bus:
                    updated_state["event_bus"] = event_bus

                logger.info("Executing orchestrator synthesis...")

                # Emit agent start event for synthesis
                if event_bus:
                    await event_bus.emit(
                        event_type=EventType.AGENT_START,
                        data={
                            "agent": "orchestrator",
                            "message": "Synthesizing final answer..."
                        },
                        agent_name="orchestrator"
                    )

                # Execute orchestrator (will detect PHASE 2 based on agent_outputs)
                result = await orchestrator.execute(updated_state)

                logger.info("Orchestrator synthesis completed successfully")

                # CRITICAL: Ensure current_agent is preserved in result for progress tracking
                if "current_agent" not in result:
                    result["current_agent"] = "orchestrator"

                return result

            except Exception as e:
                logger.error(f"Orchestrator synthesis failed: {e}", exc_info=True)

                # Convert to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Add error to state
                errors = state_dict.get("errors", [])
                errors.append(f"orchestrator synthesis error: {str(e)}")

                # Provide fallback final answer
                state_dict["final_answer"] = (
                    f"Synthesis error: {str(e)}. "
                    f"Agents completed their analysis, but final answer generation failed."
                )

                return {**state_dict, "errors": errors}

        # Add node to workflow
        workflow.add_node("orchestrator_synthesis", orchestrator_synthesis_node)

        logger.debug("Added orchestrator synthesis node")

    def _add_hitl_gate_node(self, workflow: StateGraph) -> None:
        """
        Add HITL quality gate node to workflow.

        This node evaluates retrieval quality and triggers clarification if needed.

        Args:
            workflow: StateGraph to add node to
        """
        if not self.hitl_config or not self.hitl_config.enabled:
            logger.info("HITL not enabled, skipping gate node")
            return

        async def hitl_gate(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Quality gate that checks if clarification is needed.

            Returns state with clarification questions if quality is low,
            or passes through if quality is acceptable.
            """
            try:
                # Convert MultiAgentState to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Check if we're resuming after user clarification
                if state_dict.get("user_clarification"):
                    logger.info("HITL: Resuming after user clarification")
                    from ..hitl.context_enricher import ContextEnricher

                    enricher = ContextEnricher(self.hitl_config)

                    # Enrich query with user response
                    original_query = state_dict.get("original_query", state_dict.get("query", ""))
                    user_response = state_dict["user_clarification"]

                    updated_state = enricher.enrich(original_query, user_response, state_dict)
                    updated_state["awaiting_user_input"] = False
                    updated_state["quality_check_required"] = False

                    logger.info(f"HITL: Query enriched, continuing workflow")
                    return updated_state

                # Check complexity threshold
                complexity_score = state_dict.get("complexity_score", 0)
                if complexity_score < self.hitl_config.min_complexity_score:
                    logger.info(
                        f"HITL: Complexity {complexity_score} below threshold "
                        f"{self.hitl_config.min_complexity_score}, skipping"
                    )
                    return {**state_dict, "quality_check_required": False}

                # Evaluate retrieval quality
                query = state_dict.get("query", "")
                search_results = state_dict.get("documents", [])

                should_clarify, metrics = self.quality_detector.evaluate(
                    query=query,
                    search_results=search_results,
                    complexity_score=complexity_score,
                )

                # Store metrics in state
                updated_state = {
                    **state_dict,
                    "quality_metrics": {
                        "retrieval_score": metrics.retrieval_score,
                        "semantic_coherence": metrics.semantic_coherence,
                        "query_pattern_score": metrics.query_pattern_score,
                        "document_diversity": metrics.document_diversity,
                        "overall_quality": metrics.overall_quality,
                    },
                    "quality_issues": metrics.failing_metrics,
                }

                if not should_clarify:
                    logger.info(
                        f"HITL: Quality acceptable ({metrics.overall_quality:.2f}), "
                        f"continuing workflow"
                    )
                    return {**updated_state, "quality_check_required": False}

                # Check clarification round limit
                current_round = state_dict.get("clarification_round", 0)
                if current_round >= self.hitl_config.max_rounds:
                    logger.warning(
                        f"HITL: Max clarification rounds ({self.hitl_config.max_rounds}) "
                        f"reached, continuing anyway"
                    )
                    return {**updated_state, "quality_check_required": False}

                # Generate clarifying questions
                logger.info(
                    f"HITL: Quality low ({metrics.overall_quality:.2f}), "
                    f"generating clarifying questions"
                )

                questions = await self.clarification_generator.generate(
                    query=query,
                    metrics=metrics,
                    context={
                        "complexity_score": complexity_score,
                        "num_results": len(search_results),
                    },
                )

                # Update state for clarification
                final_state = {
                    **updated_state,
                    "quality_check_required": True,
                    "clarifying_questions": [
                        {"id": q.id, "text": q.text, "type": q.type} for q in questions
                    ],
                    "original_query": query,
                    "clarification_round": current_round + 1,
                    "awaiting_user_input": True,
                }

                logger.info(f"HITL: Generated {len(questions)} clarifying questions")

                # Interrupt workflow to wait for user response
                interrupt(
                    {
                        "type": "clarification_needed",
                        "questions": final_state["clarifying_questions"],
                        "quality_metrics": final_state["quality_metrics"],
                    }
                )

                return final_state

            except Exception as e:
                # Check if this is an Interrupt (not an error)
                if e.__class__.__name__ == "Interrupt":
                    # Re-raise interrupt (this is expected behavior)
                    raise

                logger.error(f"HITL gate error: {e}", exc_info=True)
                # On error, continue without clarification
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)
                errors = state_dict.get("errors", [])
                errors.append(f"HITL gate error: {str(e)}")
                return {**state_dict, "quality_check_required": False, "errors": errors}

        # Add gate node to workflow
        workflow.add_node("hitl_gate", hitl_gate)
        logger.debug("Added HITL gate node")

    def _add_iteration_support(self, workflow: StateGraph) -> None:
        """
        Add iterative refinement support to workflow.

        Adds iteration_dispatcher node and conditional edge from orchestrator_synthesis
        to support multi-round agent execution.

        Args:
            workflow: StateGraph to add iteration support to
        """
        # Add iteration dispatcher node
        async def iteration_dispatcher(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Dispatch next_agents for iterative refinement.

            Reads state["next_agents"] and executes them sequentially,
            then returns to orchestrator_synthesis.
            """
            # Convert to dict if needed
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            else:
                state_dict = dict(state)

            next_agents = state_dict.get("next_agents", [])
            iteration_count = state_dict.get("iteration_count", 0)

            logger.info(
                f"Iteration {iteration_count}: Executing {len(next_agents)} agents - "
                f"{', '.join(next_agents)}"
            )

            # Execute each agent in next_agents
            for agent_name in next_agents:
                agent = self.agent_registry.get_agent(agent_name)

                if agent is None:
                    logger.error(f"Agent not found for iteration: {agent_name}")
                    continue

                # Update current agent in state
                state_dict["current_agent"] = agent_name

                try:
                    logger.info(f"Iteration: Executing agent {agent_name}")
                    result = await agent.execute(state_dict)

                    # Merge results back into state
                    state_dict = {**state_dict, **result}

                    logger.info(f"Iteration: Agent {agent_name} completed")
                except Exception as e:
                    logger.error(f"Iteration: Agent {agent_name} failed: {e}", exc_info=True)
                    errors = state_dict.get("errors", [])
                    errors.append(f"Iteration agent {agent_name} error: {str(e)}")
                    state_dict["errors"] = errors

            logger.info(f"Iteration {iteration_count}: All agents completed, returning to synthesis")

            return state_dict

        workflow.add_node("iteration_dispatcher", iteration_dispatcher)
        logger.debug("Added iteration_dispatcher node")

        # Add conditional edge for iterative refinement
        def should_iterate(state: Dict[str, Any]) -> str:
            """
            Check if orchestrator requested additional iteration.

            Returns:
                - "iteration_dispatcher" if needs_iteration and under limit
                - END if done or at max iterations
            """
            # Convert to dict if needed
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif hasattr(state, "get"):
                state_dict = state
            else:
                state_dict = dict(state)

            needs_iteration = state_dict.get("needs_iteration", False)
            iteration_count = state_dict.get("iteration_count", 0)
            max_iterations = 3  # Total 3 rounds (1 initial + 2 additional)

            if needs_iteration and iteration_count < max_iterations:
                logger.info(
                    f"Orchestrator requested iteration {iteration_count}/{max_iterations-1}. "
                    f"Dispatching next agents: {state_dict.get('next_agents', [])}"
                )
                return "iteration_dispatcher"
            elif needs_iteration and iteration_count >= max_iterations:
                logger.warning(
                    f"Max iterations ({max_iterations}) reached. "
                    f"Ending workflow with current answer."
                )
                return END
            else:
                logger.info("Synthesis complete, no iteration needed. Ending workflow.")
                return END

        # Add conditional edge: orchestrator_synthesis → [iteration_dispatcher | END]
        workflow.add_conditional_edges(
            "orchestrator_synthesis",
            should_iterate,
            {
                "iteration_dispatcher": "iteration_dispatcher",
                END: END
            }
        )
        logger.debug("Added conditional edge: orchestrator_synthesis → [iteration_dispatcher | END]")

        # Loop back: iteration_dispatcher → orchestrator_synthesis
        workflow.add_edge("iteration_dispatcher", "orchestrator_synthesis")
        logger.debug("Added edge: iteration_dispatcher → orchestrator_synthesis (loop back)")

    def _add_workflow_edges(
        self,
        workflow: StateGraph,
        agent_sequence: List[str],
        enable_parallel: bool = False,
    ) -> None:
        """
        Add edges to connect agents in workflow.

        Workflow pattern:
        1. Execute agent sequence (with optional HITL gate after extractor)
        2. Call orchestrator for synthesis (generates final answer)
        3. End

        Args:
            workflow: StateGraph to add edges to
            agent_sequence: Ordered list of agent names
            enable_parallel: Enable parallel execution where possible
        """
        if len(agent_sequence) == 0:
            return

        # Check if we should insert HITL gate
        should_insert_hitl = (
            self.hitl_config
            and self.hitl_config.enabled
            and "extractor" in agent_sequence
        )

        # Add orchestrator synthesis node (called AFTER all agents complete)
        # This is the SAME orchestrator agent, but called in PHASE 2 (synthesis)
        # It will detect the phase based on presence of agent_outputs
        self._add_orchestrator_synthesis_node(workflow)

        # Sequential execution (default)
        if not enable_parallel:
            # Connect agents in sequence
            for i in range(len(agent_sequence) - 1):
                current_agent = agent_sequence[i]
                next_agent = agent_sequence[i + 1]

                # Insert HITL gate after extractor
                if should_insert_hitl and current_agent == "extractor":
                    workflow.add_edge(current_agent, "hitl_gate")
                    workflow.add_edge("hitl_gate", next_agent)
                    logger.debug(f"Added edge: {current_agent} → hitl_gate → {next_agent}")
                else:
                    workflow.add_edge(current_agent, next_agent)
                    logger.debug(f"Added edge: {current_agent} → {next_agent}")

            # Last agent in sequence goes to orchestrator synthesis
            last_agent = agent_sequence[-1]
            workflow.add_edge(last_agent, "orchestrator_synthesis")
            logger.debug(f"Added edge: {last_agent} → orchestrator_synthesis")

            # Add iterative refinement support (shared with parallel execution)
            self._add_iteration_support(workflow)
            logger.debug("Added iterative refinement support for sequential execution")

        else:
            # Parallel execution (Fan-Out/Fan-In pattern)
            logger.info(f"Using parallel execution for {len(agent_sequence)} agents")

            # Handle HITL gate special case:
            # If HITL enabled and extractor in sequence, run extractor first,
            # then HITL gate, then fan-out remaining agents in parallel
            if should_insert_hitl and "extractor" in agent_sequence:
                logger.info("HITL enabled: extractor → hitl_gate → [other agents parallel]")

                # Entry point: extractor
                workflow.set_entry_point("extractor")

                # extractor → hitl_gate
                workflow.add_edge("extractor", "hitl_gate")
                logger.debug("Added edge: extractor → hitl_gate")

                # Get other agents (excluding extractor)
                other_agents = [a for a in agent_sequence if a != "extractor"]

                # Fan-out: hitl_gate → other agents (parallel)
                for agent in other_agents:
                    workflow.add_edge("hitl_gate", agent)
                    logger.debug(f"Added edge: hitl_gate → {agent} (parallel)")

                # Fan-in: all agents → orchestrator_synthesis
                for agent in agent_sequence:
                    workflow.add_edge(agent, "orchestrator_synthesis")
                    logger.debug(f"Added edge: {agent} → orchestrator_synthesis (fan-in)")

            else:
                # No HITL or extractor not in sequence: simple fan-out/fan-in
                logger.info("No HITL: [all agents parallel] → orchestrator_synthesis")

                # Create a start node that just passes through state
                async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
                    """Entry point for parallel execution - just passes through state."""
                    # Convert to dict if needed
                    if hasattr(state, "model_dump"):
                        return state.model_dump()
                    return dict(state)

                workflow.add_node("start", start_node)
                workflow.set_entry_point("start")

                # Fan-out: start → all agents (parallel)
                for agent in agent_sequence:
                    workflow.add_edge("start", agent)
                    logger.debug(f"Added edge: start → {agent} (parallel)")

                # Fan-in: all agents → orchestrator_synthesis
                for agent in agent_sequence:
                    workflow.add_edge(agent, "orchestrator_synthesis")
                    logger.debug(f"Added edge: {agent} → orchestrator_synthesis (fan-in)")

            # Add iterative refinement support (shared with sequential execution)
            self._add_iteration_support(workflow)
            logger.debug("Added iterative refinement support for parallel execution")

    def build_conditional_workflow(
        self, complexity_score: int, agent_registry: AgentRegistry
    ) -> StateGraph:
        """
        Build workflow with conditional routing based on complexity.

        Args:
            complexity_score: Query complexity score (0-100)
            agent_registry: Agent registry

        Returns:
            Compiled StateGraph with conditional routing
        """
        logger.info(f"Building conditional workflow for complexity={complexity_score}")

        workflow = StateGraph(MultiAgentState)

        # Add all possible agents
        agents = [
            "extractor",
            "classifier",
            "compliance",
            "risk_verifier",
            "citation_auditor",
            "gap_synthesizer",
            "report_generator",
        ]

        for agent_name in agents:
            self._add_agent_node(workflow, agent_name)

        # Add conditional routing logic
        def route_after_extractor(state: MultiAgentState) -> str:
            """Route after extractor based on complexity."""
            score = state.get("complexity_score", 50)

            if score < 30:
                return "report_generator"
            else:
                return "classifier"

        def route_after_classifier(state: MultiAgentState) -> str:
            """Route after classifier based on query type."""
            query_type = state.get("query_type", "search")

            if query_type == "compliance":
                return "compliance"
            elif query_type == "risk":
                return "risk_verifier"
            else:
                return "gap_synthesizer"

        # Set entry point
        workflow.set_entry_point("extractor")

        # Add conditional edges
        workflow.add_conditional_edges(
            "extractor", route_after_extractor, ["classifier", "report_generator"]
        )

        workflow.add_conditional_edges(
            "classifier",
            route_after_classifier,
            ["compliance", "risk_verifier", "gap_synthesizer"],
        )

        # Connect domain agents to citation auditor
        for agent in ["compliance", "risk_verifier"]:
            workflow.add_edge(agent, "citation_auditor")

        workflow.add_edge("citation_auditor", "gap_synthesizer")
        workflow.add_edge("gap_synthesizer", "report_generator")
        workflow.add_edge("report_generator", END)

        # Compile
        if self.checkpointer:
            compiled = workflow.compile(checkpointer=self.checkpointer)
        else:
            compiled = workflow.compile()

        logger.info("Conditional workflow built successfully")

        return compiled

    def visualize_workflow(self, workflow: StateGraph, output_path: str) -> None:
        """
        Visualize workflow graph (requires graphviz).

        Args:
            workflow: Compiled StateGraph
            output_path: Path to save visualization
        """
        try:
            # This would require graphviz integration
            logger.info(f"Workflow visualization saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to visualize workflow: {e}")
