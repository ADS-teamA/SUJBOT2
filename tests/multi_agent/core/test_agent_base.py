"""
Tests for BaseAgent and autonomous tool calling loop.

CRITICAL COMPONENT: This is the foundation of the autonomous agentic architecture.
All 7 specialized agents inherit from BaseAgent and use _run_autonomous_tool_loop().
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.multi_agent.core.agent_base import BaseAgent, AgentConfig


# ============================================================================
# Test Agent Implementation
# ============================================================================

class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple implementation that calls autonomous tool loop."""
        system_prompt = "You are a test agent."

        result = await self._run_autonomous_tool_loop(
            system_prompt=system_prompt,
            state=state,
            max_iterations=10
        )

        return {
            **state,
            "agent_outputs": {
                self.config.name: {
                    "final_answer": result["final_answer"],
                    "tool_calls": result["tool_calls"],
                    "iterations": result["iterations"]
                }
            }
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    from src.multi_agent.core.agent_base import AgentRole, AgentTier

    return AgentConfig(
        name="test_agent",
        role=AgentRole.EXTRACT,
        tier=AgentTier.WORKER,
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.1,
        tools={"search", "get_info"},
        enable_prompt_caching=True
    )


@pytest.fixture
async def test_agent(agent_config, mock_provider):
    """Create test agent instance."""
    agent = TestAgent(agent_config)
    agent.provider = mock_provider  # Inject mock provider
    return agent


# ============================================================================
# Test: LLM Decides Tool Calls Autonomously
# ============================================================================

@pytest.mark.asyncio
async def test_llm_decides_tool_calls_autonomously(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    CRITICAL: LLM should decide which tools to call, not hardcoded logic.

    This is the core of autonomous agentic architecture (CLAUDE.md Constraint #0).
    """
    state = mock_state(query="Test query")

    # Mock LLM to request tool_a, then tool_b based on result
    mock_provider.create_message.side_effect = [
        # First call: LLM requests tool_a
        mock_llm_response(tool_calls=[
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "search",
                "input": {"query": "test"}
            }
        ], stop_reason="tool_use"),
        # Second call: After tool_a result, LLM requests tool_b
        mock_llm_response(tool_calls=[
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "get_info",
                "input": {"doc_id": "doc1"}
            }
        ], stop_reason="tool_use"),
        # Third call: LLM provides final answer
        mock_llm_response(text="Final answer based on tool results", stop_reason="end_turn")
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent.execute(state)

    # Verify LLM autonomously decided tool sequence
    assert "agent_outputs" in result
    agent_output = result["agent_outputs"]["test_agent"]

    assert len(agent_output["tool_calls"]) == 2
    assert agent_output["tool_calls"][0]["tool"] == "search"
    assert agent_output["tool_calls"][1]["tool"] == "get_info"
    assert agent_output["iterations"] < 10  # Completed before max
    assert "Final answer" in agent_output["final_answer"]


# ============================================================================
# Test: Max Iterations Forces Completion
# ============================================================================

@pytest.mark.asyncio
async def test_max_iterations_forces_completion(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Should force completion after max_iterations without infinite loop.

    Prevents API cost drain from runaway tool calling.
    """
    state = mock_state(query="Complex query")

    # Mock LLM to always request more tools (never provide final answer)
    mock_provider.create_message.return_value = mock_llm_response(
        tool_calls=[
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "search",
                "input": {"query": "query_1"}
            }
        ],
        stop_reason="tool_use"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Keep calling tools",
            state=state,
            max_iterations=3
        )

    # Verify forced completion at max iterations
    assert result["iterations"] == 3
    assert "maximum reasoning steps reached" in result["final_answer"].lower() or \
           "iteration" in result["final_answer"].lower()


# ============================================================================
# Test: Cumulative Tool Cost Tracking
# ============================================================================

@pytest.mark.asyncio
async def test_tracks_cumulative_tool_costs(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Should track total API cost across all tool calls.

    Critical for cost monitoring and budget enforcement.
    """
    state = mock_state(query="Test query")

    # Mock LLM to call 3 tools
    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[{"type": "tool_use", "id": "1", "name": "search", "input": {}}], stop_reason="tool_use"),
        mock_llm_response(tool_calls=[{"type": "tool_use", "id": "2", "name": "get_info", "input": {}}], stop_reason="tool_use"),
        mock_llm_response(tool_calls=[{"type": "tool_use", "id": "3", "name": "search", "input": {}}], stop_reason="tool_use"),
        mock_llm_response(text="Final answer", stop_reason="end_turn")
    ]

    # Mock tool adapter to return different costs
    mock_tool_adapter.execute.side_effect = [
        {"success": True, "data": "result1", "metadata": {"api_cost_usd": 0.001}},
        {"success": True, "data": "result2", "metadata": {"api_cost_usd": 0.002}},
        {"success": True, "data": "result3", "metadata": {"api_cost_usd": 0.003}}
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=10
        )

    # Verify cumulative cost tracking
    assert "total_tool_cost_usd" in result
    assert result["total_tool_cost_usd"] == pytest.approx(0.006, rel=1e-9)

    # Verify individual tool costs tracked
    assert len(result["tool_calls"]) == 3
    assert sum(call["api_cost_usd"] for call in result["tool_calls"]) == pytest.approx(0.006, rel=1e-9)


# ============================================================================
# Test: Critical Tool Failure Surfacing
# ============================================================================

@pytest.mark.asyncio
async def test_surfaces_critical_tool_failures_to_user(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Critical tool failures should be added to state["errors"] for user notification.

    This is the fix from PR review - prevents silent failures.
    """
    state = mock_state(query="Test query")

    # Mock LLM to call hierarchical_search (critical tool)
    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[
            {"type": "tool_use", "id": "1", "name": "hierarchical_search", "input": {"query": "test"}}
        ], stop_reason="tool_use"),
        mock_llm_response(text="Partial answer", stop_reason="end_turn")
    ]

    # Mock critical tool failure
    mock_tool_adapter.execute.return_value = {
        "success": False,
        "error": "Database connection failed",
        "data": None,
        "metadata": {"api_cost_usd": 0.0}
    }

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Verify error surfaced to state
    assert "errors" in state
    assert len(state["errors"]) > 0
    assert any("hierarchical_search" in err for err in state["errors"])
    assert any("Database connection failed" in err for err in state["errors"])


# ============================================================================
# Test: Non-Critical Tool Failure Logged But Not Surfaced
# ============================================================================

@pytest.mark.asyncio
async def test_non_critical_tool_failure_logged_only(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state, caplog
):
    """
    Non-critical tool failures should be logged but not surface to user.

    Only critical tools (hierarchical_search, similarity_search, etc.) surface errors.
    """
    state = mock_state(query="Test query")

    # Mock LLM to call non-critical tool
    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[
            {"type": "tool_use", "id": "1", "name": "some_other_tool", "input": {}}
        ], stop_reason="tool_use"),
        mock_llm_response(text="Answer", stop_reason="end_turn")
    ]

    # Mock non-critical tool failure
    mock_tool_adapter.execute.return_value = {
        "success": False,
        "error": "Tool unavailable",
        "data": None,
        "metadata": {}
    }

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Verify error logged but NOT in state["errors"]
    assert "Tool execution failed" in caplog.text
    assert "some_other_tool" in caplog.text

    # Non-critical tool failure should NOT be in state errors
    assert "errors" not in state or len(state.get("errors", [])) == 0


# ============================================================================
# Test: Tool Result Format
# ============================================================================

@pytest.mark.asyncio
async def test_tool_results_have_error_flag(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Tool results should include is_error flag for debugging.

    This helps identify which tools failed in execution history.
    """
    state = mock_state()

    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[{"type": "tool_use", "id": "1", "name": "search", "input": {}}], stop_reason="tool_use"),
        mock_llm_response(text="Answer", stop_reason="end_turn")
    ]

    # Mock tool failure
    mock_tool_adapter.execute.return_value = {
        "success": False,
        "error": "Tool error",
        "data": None,
        "metadata": {}
    }

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Verify tool call history includes success flag
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["success"] is False


# ============================================================================
# Test: Provider Initialization
# ============================================================================

@pytest.mark.asyncio
async def test_provider_initialization_creates_correct_provider(agent_config):
    """
    Agent should create provider for configured model.

    Tests the __init__ provider creation logic.
    """
    with patch("src.multi_agent.core.agent_base.create_provider") as mock_create:
        mock_create.return_value = Mock()

        agent = TestAgent(agent_config)

        mock_create.assert_called_once_with(model="claude-sonnet-4-5-20250929")
        assert agent.provider is not None


@pytest.mark.asyncio
async def test_provider_initialization_failure_raises_valueerror(agent_config):
    """
    Provider initialization failure should raise ValueError with helpful message.

    Tests error handling in __init__.
    """
    with patch("src.multi_agent.core.agent_base.create_provider") as mock_create:
        mock_create.side_effect = Exception("Invalid API key")

        with pytest.raises(ValueError, match="Failed to initialize LLM provider"):
            TestAgent(agent_config)


# ============================================================================
# Test: Execute Template Method Pattern
# ============================================================================

@pytest.mark.asyncio
async def test_execute_calls_execute_impl(test_agent, mock_state):
    """
    BaseAgent.execute() should call execute_impl() (template method pattern).

    Verifies error handling wrapper works correctly.
    """
    state = mock_state()

    with patch.object(test_agent, 'execute_impl', new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = {**state, "result": "success"}

        result = await test_agent.execute(state)

        mock_impl.assert_called_once_with(state)
        assert result["result"] == "success"


@pytest.mark.asyncio
async def test_execute_handles_execute_impl_errors(test_agent, mock_state):
    """
    BaseAgent.execute() should catch and wrap execute_impl() errors.

    Ensures consistent error handling across all agents.
    """
    state = mock_state()

    with patch.object(test_agent, 'execute_impl', new_callable=AsyncMock) as mock_impl:
        mock_impl.side_effect = Exception("Agent execution failed")

        result = await test_agent.execute(state)

        assert "errors" in result
        assert any("Agent execution failed" in err for err in result["errors"])


# ============================================================================
# Test: Prompt Loading
# ============================================================================

def test_load_prompt_from_file(test_agent, tmp_path):
    """
    Should load prompt from prompts/agents/{agent_name}.txt file.

    Tests hot-reloadable prompt system.
    """
    # Create temporary prompt file
    prompt_file = tmp_path / "test_agent.txt"
    prompt_file.write_text("You are a test agent.\nYour role is: {role}")

    with patch("src.multi_agent.core.agent_base.Path") as mock_path:
        mock_path.return_value = tmp_path
        prompt = test_agent._load_prompt(context={"role": "testing"})

        assert "You are a test agent" in prompt
        assert "testing" in prompt


def test_load_prompt_fallback_on_missing_file(test_agent):
    """
    Should use fallback prompt if file doesn't exist.

    Prevents crashes when prompt files are missing.
    """
    with patch("src.multi_agent.core.agent_base.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        prompt = test_agent._load_prompt()

        assert len(prompt) > 0  # Should have fallback content
        assert "test_agent" in prompt.lower()


# ============================================================================
# Test: Tool Schema Generation
# ============================================================================

@pytest.mark.asyncio
async def test_tool_schemas_passed_to_llm(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Tool schemas should be passed to LLM for autonomous tool selection.

    LLM needs schemas to know which tools are available and how to call them.
    """
    state = mock_state()

    mock_provider.create_message.return_value = mock_llm_response(text="Answer")

    # Mock tool schemas
    mock_tool_adapter.get_tool_schema.side_effect = [
        {"name": "search", "description": "Search tool", "input_schema": {}},
        {"name": "get_info", "description": "Info tool", "input_schema": {}}
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Verify create_message called with tools parameter
    call_args = mock_provider.create_message.call_args
    assert "tools" in call_args.kwargs
    assert len(call_args.kwargs["tools"]) == 2


# ============================================================================
# Test: Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_handles_empty_tool_calls_list(
    test_agent, mock_provider, mock_llm_response, mock_state
):
    """
    Should handle LLM returning empty tool_calls list gracefully.

    Edge case: LLM returns tool_use stop_reason but no tools.
    """
    state = mock_state()

    mock_provider.create_message.return_value = mock_llm_response(
        tool_calls=[],  # Empty list
        stop_reason="tool_use"
    )

    with patch("src.multi_agent.core.agent_base.tool_adapter"):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=3
        )

    # Should complete without crashing
    assert "final_answer" in result


@pytest.mark.asyncio
async def test_handles_malformed_tool_use_dict(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state, caplog
):
    """
    Should handle malformed tool_use dict gracefully.

    Edge case: Missing 'name' or 'input' keys in tool_use.
    """
    state = mock_state()

    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[
            {"type": "tool_use", "id": "1"}  # Missing 'name' and 'input'
        ], stop_reason="tool_use"),
        mock_llm_response(text="Answer", stop_reason="end_turn")
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Should not crash, should log error
    assert "final_answer" in result


@pytest.mark.asyncio
async def test_handles_none_tool_result(
    test_agent, mock_provider, mock_tool_adapter, mock_llm_response, mock_state
):
    """
    Should handle tool adapter returning None gracefully.

    Edge case: Tool adapter crashes and returns None instead of dict.
    """
    state = mock_state()

    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[{"type": "tool_use", "id": "1", "name": "search", "input": {}}], stop_reason="tool_use"),
        mock_llm_response(text="Answer", stop_reason="end_turn")
    ]

    mock_tool_adapter.execute.return_value = None  # Invalid return

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await test_agent._run_autonomous_tool_loop(
            system_prompt="Test",
            state=state,
            max_iterations=5
        )

    # Should handle gracefully without crashing
    assert "final_answer" in result
