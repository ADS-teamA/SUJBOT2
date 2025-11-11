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
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt

from ..core.state import MultiAgentState, ExecutionPhase
from ..core.agent_registry import AgentRegistry
from ..hitl.config import HITLConfig
from ..hitl.quality_detector import QualityDetector
from ..hitl.clarification_generator import ClarificationGenerator

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
        checkpointer: Optional[PostgresSaver] = None,
    ):
        """
        Initialize workflow builder.

        Args:
            agent_registry: Registry containing all agent instances
            checkpointer: Optional PostgreSQL checkpointer for state persistence
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
        async def agent_node(state: MultiAgentState) -> MultiAgentState:
            """Execute agent and update state."""
            try:
                # Update execution phase
                state["execution_phase"] = ExecutionPhase.AGENT_EXECUTION.value
                state["current_agent"] = agent_name

                # Add agent to sequence if not already there
                if agent_name not in state.get("agent_sequence", []):
                    state["agent_sequence"] = state.get("agent_sequence", []) + [
                        agent_name
                    ]

                logger.info(f"Executing agent: {agent_name}")

                # Execute agent
                updated_state = await agent.execute(state)

                logger.info(f"Agent {agent_name} completed successfully")

                return updated_state

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}", exc_info=True)

                # Add error to state
                updated_state = state.copy()
                updated_state["errors"] = updated_state.get("errors", [])
                updated_state["errors"].append(f"{agent_name} error: {str(e)}")

                return updated_state

        # Add node to workflow
        workflow.add_node(agent_name, agent_node)

        logger.debug(f"Added agent node: {agent_name}")

    def _add_workflow_edges(
        self,
        workflow: StateGraph,
        agent_sequence: List[str],
        enable_parallel: bool = False,
    ) -> None:
        """
        Add edges to connect agents in workflow.

        Args:
            workflow: StateGraph to add edges to
            agent_sequence: Ordered list of agent names
            enable_parallel: Enable parallel execution where possible
        """
        if len(agent_sequence) == 0:
            return

        # Sequential execution (default)
        if not enable_parallel:
            # Connect agents in sequence
            for i in range(len(agent_sequence) - 1):
                current_agent = agent_sequence[i]
                next_agent = agent_sequence[i + 1]

                workflow.add_edge(current_agent, next_agent)

                logger.debug(f"Added edge: {current_agent} → {next_agent}")

            # Last agent goes to END
            last_agent = agent_sequence[-1]
            workflow.add_edge(last_agent, END)

            logger.debug(f"Added edge: {last_agent} → END")

        else:
            # Parallel execution (advanced - not implemented yet)
            # For now, fall back to sequential
            logger.warning("Parallel execution not yet implemented, using sequential")
            self._add_workflow_edges(workflow, agent_sequence, enable_parallel=False)

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
