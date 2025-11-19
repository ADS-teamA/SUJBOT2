"""
Test LangGraph state merging with reducer functions.

Verifies that the InvalidUpdateError fix works correctly in real
LangGraph workflows with parallel execution (fan-out/fan-in).
"""

import pytest
from langgraph.graph import StateGraph, END
from src.multi_agent.core.state import MultiAgentState


@pytest.mark.asyncio
async def test_parallel_execution_state_merging():
    """
    Test that multiple agents can update state in parallel without conflicts.

    This simulates the exact scenario that caused the original error:
    - Multiple nodes updating state simultaneously
    - LangGraph merging updates using reducer functions
    - No InvalidUpdateError: "Can receive only one value per step"
    """

    # Create a simple workflow with parallel execution
    workflow = StateGraph(MultiAgentState)

    # Define two agents that run in parallel
    async def agent1(state: MultiAgentState) -> dict:
        """Agent 1 simulates document extraction."""
        return {
            "agent_sequence": ["agent1"],
            "agent_outputs": {"agent1": {"docs_found": 10}},
            "documents": [{"doc_id": "doc1", "score": 0.9}],
            "errors": [],
        }

    async def agent2(state: MultiAgentState) -> dict:
        """Agent 2 simulates classification."""
        return {
            "agent_sequence": ["agent2"],
            "agent_outputs": {"agent2": {"category": "compliance"}},
            "documents": [{"doc_id": "doc2", "score": 0.8}],
            "complexity_score": 75,
            "errors": [],
        }

    # Add nodes
    workflow.add_node("agent1", agent1)
    workflow.add_node("agent2", agent2)

    # Create fan-out: both agents run in parallel
    # This is the scenario that triggers the original error
    async def start_node(state: MultiAgentState) -> dict:
        """Entry point for parallel execution."""
        return {}  # Return empty dict - LangGraph will merge with existing state

    workflow.add_node("start", start_node)
    workflow.set_entry_point("start")

    # Fan-out: start → [agent1, agent2] (parallel)
    workflow.add_edge("start", "agent1")
    workflow.add_edge("start", "agent2")

    # Fan-in: [agent1, agent2] → END
    # This is where LangGraph merges state from both agents
    workflow.add_edge("agent1", END)
    workflow.add_edge("agent2", END)

    # Compile workflow
    app = workflow.compile()

    # Execute workflow with initial state
    initial_state = {"query": "What are the compliance requirements?"}

    # This should NOT raise InvalidUpdateError anymore
    result = await app.ainvoke(initial_state)

    # Verify state was merged correctly using reducers
    assert result["query"] == "What are the compliance requirements?"  # keep_first
    assert set(result["agent_sequence"]) == {"agent1", "agent2"}  # merge_lists_unique
    assert "agent1" in result["agent_outputs"]  # merge_dicts
    assert "agent2" in result["agent_outputs"]  # merge_dicts
    assert len(result["documents"]) == 2  # operator.add
    assert result["complexity_score"] == 75  # take_max (max of 0 and 75)
    assert result["errors"] == []  # operator.add (empty lists)


@pytest.mark.asyncio
async def test_sequential_execution_no_conflicts():
    """
    Test that sequential execution still works correctly.

    Ensures the reducer functions don't break normal sequential flows.
    """

    workflow = StateGraph(MultiAgentState)

    async def agent1(state: MultiAgentState) -> dict:
        return {
            "agent_sequence": ["agent1"],
            "complexity_score": 30,
        }

    async def agent2(state: MultiAgentState) -> dict:
        return {
            "agent_sequence": ["agent2"],
            "complexity_score": 50,
        }

    workflow.add_node("agent1", agent1)
    workflow.add_node("agent2", agent2)

    workflow.set_entry_point("agent1")
    workflow.add_edge("agent1", "agent2")
    workflow.add_edge("agent2", END)

    app = workflow.compile()

    result = await app.ainvoke({"query": "Test query"})

    assert result["query"] == "Test query"
    assert result["agent_sequence"] == ["agent1", "agent2"]
    assert result["complexity_score"] == 50  # take_max keeps highest


@pytest.mark.asyncio
async def test_partial_state_updates_accepted():
    """
    Test that agents can return partial state updates.

    This was the root cause of the validation error - LangGraph receives
    partial updates (not all fields), which need to be merged using reducers.
    """

    workflow = StateGraph(MultiAgentState)

    async def agent1(state: MultiAgentState) -> dict:
        # Agent only updates a subset of fields (partial update)
        return {
            "agent_sequence": ["agent1"],
            "complexity_score": 25,
        }

    async def agent2(state: MultiAgentState) -> dict:
        # Another agent updates different fields (partial update)
        return {
            "agent_sequence": ["agent2"],
            "agent_outputs": {"agent2": {"result": "done"}},
            "final_answer": "Answer from agent2",
        }

    workflow.add_node("agent1", agent1)
    workflow.add_node("agent2", agent2)

    # Sequential execution
    workflow.set_entry_point("agent1")
    workflow.add_edge("agent1", "agent2")
    workflow.add_edge("agent2", END)

    app = workflow.compile()

    # Initial state has query
    result = await app.ainvoke({"query": "Test query"})

    # Verify all updates were merged correctly
    assert result["query"] == "Test query"  # Preserved from initial state
    assert result["agent_sequence"] == ["agent1", "agent2"]
    assert result["complexity_score"] == 25
    assert "agent2" in result["agent_outputs"]
    assert result["final_answer"] == "Answer from agent2"


@pytest.mark.asyncio
async def test_reducer_deduplication():
    """
    Test that merge_lists_unique reducer prevents duplicate agent entries.

    If an agent appears in both branches of parallel execution,
    it should only appear once in the final sequence.
    """

    workflow = StateGraph(MultiAgentState)

    async def agent1(state: MultiAgentState) -> dict:
        return {
            "agent_sequence": ["extractor", "agent1"],
        }

    async def agent2(state: MultiAgentState) -> dict:
        return {
            "agent_sequence": ["extractor", "agent2"],  # Duplicate "extractor"
        }

    workflow.add_node("agent1", agent1)
    workflow.add_node("agent2", agent2)

    async def start(state: MultiAgentState) -> dict:
        return {}

    workflow.add_node("start", start)
    workflow.set_entry_point("start")

    # Parallel execution
    workflow.add_edge("start", "agent1")
    workflow.add_edge("start", "agent2")
    workflow.add_edge("agent1", END)
    workflow.add_edge("agent2", END)

    app = workflow.compile()

    result = await app.ainvoke({"query": "Test"})

    # "extractor" should appear only once despite being in both branches
    assert result["agent_sequence"] == ["extractor", "agent1", "agent2"]
