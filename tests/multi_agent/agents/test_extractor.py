"""
Tests for Extractor agent.

The Extractor agent is responsible for retrieving relevant documents and chunks
from the vector store based on the user query.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from src.multi_agent.agents.extractor import ExtractorAgent
from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def extractor_config():
    """Configuration for Extractor agent."""
    return AgentConfig(
        name="extractor",
        role=AgentRole.EXTRACT,
        tier=AgentTier.WORKER,
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=0.1,
        tools={
            "hierarchical_search",
            "similarity_search",
            "get_document_info",
            "list_available_documents"
        },
        enable_prompt_caching=True
    )


@pytest.fixture
def extractor_agent(extractor_config, mock_anthropic_provider):
    """Create Extractor agent instance."""
    with patch("builtins.open", mock_open(read_data="You are an extractor agent.")):
        agent = ExtractorAgent(config=extractor_config)
        # Override provider after initialization
        agent.provider = mock_anthropic_provider
    return agent


# ============================================================================
# Configuration Tests
# ============================================================================

def test_extractor_has_correct_tools(extractor_agent):
    """Extractor should have retrieval-focused tools."""
    expected_tools = {
        "hierarchical_search",
        "similarity_search",
        "get_document_info",
        "list_available_documents"
    }
    assert extractor_agent.config.tools == expected_tools


def test_extractor_role_is_extract(extractor_agent):
    """Extractor should have EXTRACT role."""
    assert extractor_agent.config.role == AgentRole.EXTRACT


def test_extractor_tier_is_worker(extractor_agent):
    """Extractor should be WORKER tier."""
    assert extractor_agent.config.tier == AgentTier.WORKER


def test_extractor_loads_prompt_from_file():
    """Extractor should load system prompt from prompts/agents/extractor.txt."""
    mock_prompt = "You are an extractor agent specialized in document retrieval."

    with patch("builtins.open", mock_open(read_data=mock_prompt)):
        config = AgentConfig(
            name="extractor",
            role=AgentRole.EXTRACT,
            tier=AgentTier.WORKER,
            model="claude-sonnet-4-5-20250929",
            tools={"hierarchical_search"}
        )
        agent = ExtractorAgent(config=config, provider=Mock())

    assert agent.system_prompt == mock_prompt


# ============================================================================
# Autonomous Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_extractor_uses_autonomous_tool_loop(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """CRITICAL: Extractor must use autonomous loop, NOT hardcoded flow."""
    state = mock_state(query="Find GDPR compliance requirements")

    # Mock LLM to autonomously call hierarchical_search
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR compliance requirements", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Found 10 relevant documents about GDPR compliance.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    # Verify autonomous behavior
    agent_output = result["agent_outputs"]["extractor"]
    assert "tool_calls" in agent_output
    assert len(agent_output["tool_calls"]) == 1
    assert agent_output["tool_calls"][0]["tool"] == "hierarchical_search"
    assert agent_output["final_answer"] == "Found 10 relevant documents about GDPR compliance."


@pytest.mark.asyncio
async def test_extractor_can_call_multiple_tools(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Extractor should autonomously decide to call multiple tools if needed."""
    state = mock_state(query="Compare privacy policies in doc1 and doc2")

    # Mock LLM to call hierarchical_search → get_document_info
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "privacy policy", "top_k": 5}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "get_document_info",
                "input": {"document_id": "doc1"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Retrieved privacy policies from both documents.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    agent_output = result["agent_outputs"]["extractor"]
    assert len(agent_output["tool_calls"]) == 2
    assert agent_output["tool_calls"][0]["tool"] == "hierarchical_search"
    assert agent_output["tool_calls"][1]["tool"] == "get_document_info"


@pytest.mark.asyncio
async def test_extractor_respects_max_iterations(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Extractor should stop after max_iterations even if LLM keeps calling tools."""
    state = mock_state(query="Test query")

    # Mock LLM to always request tools (never returns final answer)
    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        tool_calls=[{
            "type": "tool_use",
            "id": "call_1",
            "name": "hierarchical_search",
            "input": {"query": "test", "top_k": 10}
        }],
        stop_reason="tool_use"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    agent_output = result["agent_outputs"]["extractor"]
    # Should stop at max_iterations (default 10)
    assert len(agent_output["tool_calls"]) <= 10
    assert "iterations" in agent_output


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_extractor_handles_tool_failure(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Extractor should handle tool failures gracefully."""
    state = mock_state(query="Test query")

    # Mock tool failure
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=False,
        error="Vector store unavailable"
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "test", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Unable to retrieve documents due to system error.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    # Should surface critical tool failure
    assert "errors" in result
    assert any("hierarchical_search" in err for err in result["errors"])


@pytest.mark.asyncio
async def test_extractor_handles_missing_query(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response
):
    """Extractor should handle missing query in state."""
    state = {"query_type": "retrieval"}  # Missing 'query' key

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="No query provided.",
        stop_reason="end_turn"
    )

    result = await extractor_agent.execute(state)

    # Should not crash, return graceful response
    assert "agent_outputs" in result
    assert "extractor" in result["agent_outputs"]


@pytest.mark.asyncio
async def test_extractor_handles_provider_error(
    extractor_agent,
    mock_anthropic_provider,
    mock_state
):
    """Extractor should handle LLM provider errors gracefully."""
    state = mock_state(query="Test query")

    # Mock provider to raise exception
    mock_anthropic_provider.create_message.side_effect = Exception("API rate limit exceeded")

    with pytest.raises(Exception, match="API rate limit exceeded"):
        await extractor_agent.execute(state)


# ============================================================================
# State Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_extractor_preserves_existing_agent_outputs(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Extractor should preserve outputs from previous agents."""
    state = mock_state(query="Test query")
    state["agent_outputs"] = {
        "orchestrator": {"routing_decision": "extractor"}
    }

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Retrieved documents.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    # Should preserve orchestrator output
    assert "orchestrator" in result["agent_outputs"]
    assert result["agent_outputs"]["orchestrator"]["routing_decision"] == "extractor"

    # Should add extractor output
    assert "extractor" in result["agent_outputs"]


@pytest.mark.asyncio
async def test_extractor_adds_execution_metadata(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Extractor should add execution metadata to output."""
    state = mock_state(query="Test query")

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Retrieved documents.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    agent_output = result["agent_outputs"]["extractor"]
    assert "iterations" in agent_output
    assert "tool_calls" in agent_output
    assert "final_answer" in agent_output


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_extractor_full_retrieval_flow(
    extractor_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """End-to-end test of Extractor agent retrieval flow."""
    state = mock_state(query="What are GDPR data retention requirements?")

    # Mock successful hierarchical search
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=True,
        data={
            "layer3": [
                {
                    "chunk_id": "gdpr:sec5:0",
                    "text": "Personal data shall be kept for no longer than necessary.",
                    "relevance_score": 0.95,
                    "document_id": "gdpr"
                }
            ]
        }
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR data retention", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Found GDPR data retention requirements: personal data must be kept for no longer than necessary.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await extractor_agent.execute(state)

    # Verify complete flow
    assert result["agent_outputs"]["extractor"]["final_answer"].startswith("Found GDPR")
    assert len(result["agent_outputs"]["extractor"]["tool_calls"]) == 1
    assert mock_tool_adapter.execute.call_count == 1
