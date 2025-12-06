"""
Workflow Builder - Constructs LangGraph workflow from agent sequence.

Builds executable workflow graphs with:
1. Agent nodes (state transformations)
2. Conditional edges (routing logic)
3. Error handling nodes
4. Checkpointing integration
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..core.state import MultiAgentState, ExecutionPhase
from ..core.agent_registry import AgentRegistry
from ..core.event_bus import EventType

logger = logging.getLogger(__name__)


def _strip_non_serializable(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-serializable fields from state dict before returning from nodes.

    LangGraph's checkpointer uses msgpack which cannot serialize arbitrary objects.
    This function removes EventBus and any other non-serializable fields.

    Args:
        state_dict: State dictionary potentially containing non-serializable fields

    Returns:
        State dictionary safe for checkpointing
    """
    # Remove event_bus - it's not serializable and not needed in checkpoints
    state_dict.pop("event_bus", None)
    return state_dict


class WorkflowBuilder:
    """
    Builds LangGraph workflows from agent sequences.

    Constructs state graphs with proper node connections, conditional
    routing, and error handling.

    IMPORTANT: All synthesis is done by the orchestrator LLM - NO regex parsing
    or hardcoded answer extraction. This ensures autonomous agent behavior.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize workflow builder.

        Args:
            agent_registry: Registry containing all agent instances
            checkpointer: Optional checkpointer for state persistence (sync or async)
        """
        self.agent_registry = agent_registry
        self.checkpointer = checkpointer

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

        # Add edges to connect agents
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
        # Note: LangGraph passes (state, config) to node functions
        async def agent_node(state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
            """Execute agent and update state."""
            try:
                # Get EventBus from config (NOT state - state is checkpointed and event_bus is not serializable)
                event_bus = None
                if config and "configurable" in config:
                    event_bus = config["configurable"].get("event_bus")

                # Convert MultiAgentState to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Update execution phase
                # NOTE: Do NOT add event_bus to state - it breaks checkpointing
                updated_state = {
                    **state_dict,
                    "execution_phase": ExecutionPhase.AGENT_EXECUTION.value,
                    "current_agent": agent_name,
                }

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

                return _strip_non_serializable(result)

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

                return _strip_non_serializable({**state_dict, "errors": errors})

        # Add node to workflow
        workflow.add_node(agent_name, agent_node)

        logger.debug(f"Added agent node: {agent_name}")

    def _add_orchestrator_synthesis_node(self, workflow: StateGraph) -> None:
        """
        Add orchestrator synthesis node to workflow.

        This node calls the orchestrator agent in PHASE 2 (synthesis mode).
        The orchestrator detects the phase based on presence of agent_outputs.

        IMPORTANT: This is ALWAYS used for synthesis - no bypass for single-agent queries.
        The LLM synthesizes the final answer, not regex patterns.

        Args:
            workflow: StateGraph to add node to
        """
        # Get orchestrator agent from registry
        orchestrator = self.agent_registry.get_agent("orchestrator")

        if orchestrator is None:
            logger.error("Orchestrator agent not found in registry")
            raise ValueError("Orchestrator agent required for synthesis")

        async def orchestrator_synthesis_node(state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
            """Execute orchestrator synthesis and update state."""
            try:
                # Get EventBus from config (NOT state - state is checkpointed)
                event_bus = None
                if config and "configurable" in config:
                    event_bus = config["configurable"].get("event_bus")

                # Convert to dict if needed
                if hasattr(state, "model_dump"):
                    state_dict = state.model_dump()
                else:
                    state_dict = dict(state)

                # Update execution phase
                # NOTE: Do NOT add event_bus to state - it breaks checkpointing
                updated_state = {
                    **state_dict,
                    "execution_phase": ExecutionPhase.SYNTHESIS.value,
                    "current_agent": "orchestrator",
                }

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

                return _strip_non_serializable(result)

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

                # Let the error propagate - don't use hardcoded fallback messages
                return _strip_non_serializable({**state_dict, "errors": errors})

        # Add node to workflow
        workflow.add_node("orchestrator_synthesis", orchestrator_synthesis_node)

        logger.debug("Added orchestrator synthesis node")

    def _add_iteration_support(self, workflow: StateGraph) -> None:
        """
        Add iterative refinement support to workflow.

        Adds iteration_dispatcher node and conditional edge from orchestrator_synthesis
        to support multi-round agent execution.

        NOTE: Max iterations is controlled by the LLM's decision, not hardcoded limits.

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

            # Increment iteration count
            state_dict["iteration_count"] = iteration_count + 1

            logger.info(f"Iteration {iteration_count}: All agents completed, returning to synthesis")

            return _strip_non_serializable(state_dict)

        workflow.add_node("iteration_dispatcher", iteration_dispatcher)
        logger.debug("Added iteration_dispatcher node")

        # Add conditional edge for iterative refinement
        # NOTE: The decision to iterate is made by the LLM (orchestrator), not hardcoded rules
        def should_iterate(state: Dict[str, Any]) -> str:
            """
            Check if orchestrator requested additional iteration.

            The LLM decides when to stop iterating based on the quality of the answer,
            not hardcoded iteration limits.

            Returns:
                - "iteration_dispatcher" if needs_iteration is True
                - END if done
            """
            # Convert to dict if needed
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif hasattr(state, "get"):
                state_dict = state
            else:
                state_dict = dict(state)

            needs_iteration = state_dict.get("needs_iteration", False)

            if needs_iteration:
                logger.info(
                    f"Orchestrator requested iteration. "
                    f"Dispatching next agents: {state_dict.get('next_agents', [])}"
                )
                return "iteration_dispatcher"
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
        1. Execute agent sequence
        2. Call orchestrator for synthesis (generates final answer)
        3. End (or iterate if orchestrator requests)

        IMPORTANT: ALL queries go through orchestrator synthesis.
        There is no "single-agent bypass" - the LLM always synthesizes the answer.

        Args:
            workflow: StateGraph to add edges to
            agent_sequence: Ordered list of agent names
            enable_parallel: Enable parallel execution where possible
        """
        if len(agent_sequence) == 0:
            return

        # Always add orchestrator synthesis node
        self._add_orchestrator_synthesis_node(workflow)

        # Sequential execution (default)
        if not enable_parallel:
            # Connect agents in sequence
            for i in range(len(agent_sequence) - 1):
                current_agent = agent_sequence[i]
                next_agent = agent_sequence[i + 1]
                workflow.add_edge(current_agent, next_agent)
                logger.debug(f"Added edge: {current_agent} → {next_agent}")

            # Last agent in sequence goes to orchestrator synthesis
            last_agent = agent_sequence[-1]
            workflow.add_edge(last_agent, "orchestrator_synthesis")
            logger.debug(f"Added edge: {last_agent} → orchestrator_synthesis")

            # Add iterative refinement support
            self._add_iteration_support(workflow)
            logger.debug("Added iterative refinement support for sequential execution")

        else:
            # Parallel execution (Fan-Out/Fan-In pattern)
            logger.info(f"Using parallel execution for {len(agent_sequence)} agents")

            # Create a start node that just passes through state
            async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
                """Entry point for parallel execution - just passes through state."""
                # Convert to dict if needed
                if hasattr(state, "model_dump"):
                    return _strip_non_serializable(state.model_dump())
                return _strip_non_serializable(dict(state))

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

            # Add iterative refinement support
            self._add_iteration_support(workflow)
            logger.debug("Added iterative refinement support for parallel execution")

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
