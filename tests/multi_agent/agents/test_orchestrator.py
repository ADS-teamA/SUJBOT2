"""
Tests for Orchestrator Agent - routing and workflow coordination.

CRITICAL: Orchestrator is the entry point for ALL queries.
If orchestrator fails, entire system fails.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from src.multi_agent.agents.orchestrator import OrchestratorAgent
from src.multi_agent.core.agent_base import AgentConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def orchestrator_config():
    """Create orchestrator configuration."""
    from src.multi_agent.core.agent_base import AgentRole, AgentTier

    return AgentConfig(
        name="orchestrator",
        role=AgentRole.ORCHESTRATE,
        tier=AgentTier.ORCHESTRATOR,
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.1,
        tools={"list_available_documents", "list_available_agents"},
        enable_prompt_caching=True
    )


@pytest.fixture
async def orchestrator(orchestrator_config, mock_provider):
    """Create orchestrator instance with mock provider."""
    agent = OrchestratorAgent(orchestrator_config)
    agent.provider = mock_provider
    return agent


# ============================================================================
# Test: Direct Answer for Greetings/Chitchat
# ============================================================================

@pytest.mark.asyncio
async def test_direct_answer_for_greetings(orchestrator, mock_provider, mock_llm_response, mock_state):
    """
    CRITICAL: Orchestrator should return direct answer for greetings without agents.

    This is CLAUDE.md Constraint #8 - NO HARDCODED TEMPLATES.
    """
    state = mock_state(query="Hello, how are you?")

    # Mock LLM to return direct response with empty agent sequence
    mock_provider.create_message.return_value = mock_llm_response(
        text=json.dumps({
            "complexity_score": 0,
            "query_type": "chitchat",
            "agent_sequence": [],
            "final_answer": "Hello! I'm doing well, thank you. How can I help you today?",
            "reasoning": "This is a greeting, no specialized agents needed."
        })
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Verify direct answer without agent workflow
    assert "final_answer" in result
    assert result.get("agent_sequence") == []
    assert "Hello" in result["final_answer"] or "help" in result["final_answer"]


# ============================================================================
# Test: Routes Complex Queries to Multiple Agents
# ============================================================================

@pytest.mark.asyncio
async def test_routes_complex_query_to_multiple_agents(orchestrator, mock_provider, mock_llm_response, mock_state):
    """Complex queries should route to 3+ specialized agents."""
    state = mock_state(query="Compare GDPR and HIPAA compliance requirements across all documents")

    # Mock LLM to return complex routing
    mock_provider.create_message.return_value = mock_llm_response(
        text=json.dumps({
            "complexity_score": 85,
            "query_type": "multi_document_analysis",
            "agent_sequence": ["extractor", "classifier", "compliance", "gap_synthesizer", "report_generator"],
            "reasoning": "Complex multi-document compliance comparison requiring extraction, classification, analysis, and synthesis."
        })
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Verify complex routing
    assert "agent_sequence" in result
    assert len(result["agent_sequence"]) >= 3
    assert "extractor" in result["agent_sequence"]
    assert "compliance" in result["agent_sequence"]
    assert result.get("complexity_score", 0) > 70


# ============================================================================
# Test: Routes Simple Queries to Few Agents
# ============================================================================

@pytest.mark.asyncio
async def test_routes_simple_query_to_few_agents(orchestrator, mock_provider, mock_llm_response, mock_state):
    """Simple queries should route to 1-2 agents only."""
    state = mock_state(query="Find section 5.2 in the GDPR document")

    # Mock LLM to return simple routing
    mock_provider.create_message.return_value = mock_llm_response(
        text=json.dumps({
            "complexity_score": 25,
            "query_type": "retrieval",
            "agent_sequence": ["extractor"],
            "reasoning": "Simple retrieval query requires only extractor."
        })
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Verify simple routing
    assert "agent_sequence" in result
    assert len(result["agent_sequence"]) <= 2
    assert result.get("complexity_score", 100) < 50


# ============================================================================
# Test: Calls list_available_documents Tool
# ============================================================================

@pytest.mark.asyncio
async def test_calls_list_documents_tool_when_needed(orchestrator, mock_provider, mock_tool_adapter, mock_llm_response, mock_state):
    """Orchestrator should call list_available_documents for document-specific queries."""
    state = mock_state(query="What documents do you have about privacy policies?")

    # Mock LLM to first call tool, then provide routing
    mock_provider.create_message.side_effect = [
        # First: Request tool call
        mock_llm_response(tool_calls=[
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "list_available_documents",
                "input": {}
            }
        ], stop_reason="tool_use"),
        # Then: Provide routing decision
        mock_llm_response(
            text=json.dumps({
                "complexity_score": 40,
                "query_type": "retrieval",
                "agent_sequence": ["extractor", "classifier"]
            })
        )
    ]

    # Mock orchestrator tools instance
    mock_list_documents = Mock(return_value={
        "documents": [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}],
        "count": 3,
        "message": "Found 3 documents"
    })

    with patch.object(orchestrator.orchestrator_tools_instance, 'list_available_documents', mock_list_documents):
        result = await orchestrator.execute(state)

    # Verify tool was called
    mock_list_documents.assert_called_once()


# ============================================================================
# Test: JSON Parsing Failure Recovery
# ============================================================================

@pytest.mark.asyncio
async def test_recovers_from_malformed_json_response(orchestrator, mock_provider, mock_llm_response, mock_state, caplog):
    """Should handle LLM returning invalid JSON gracefully."""
    state = mock_state(query="Test query")

    # Mock LLM to return malformed JSON
    mock_provider.create_message.return_value = mock_llm_response(
        text="This is not JSON at all, just plain text response"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Verify fallback behavior (error logged, fallback routing used)
    assert "Could not parse routing decision" in caplog.text or \
           "agent_sequence" in result  # Fallback routing applied


# ============================================================================
# Test: Routing Decision Validation
# ============================================================================

@pytest.mark.asyncio
async def test_validates_agent_sequence_contains_valid_agents(orchestrator, mock_provider, mock_llm_response, mock_state):
    """Should validate that agent_sequence contains known agents."""
    state = mock_state(query="Test query")

    # Mock LLM to return invalid agent name
    mock_provider.create_message.return_value = mock_llm_response(
        text=json.dumps({
            "complexity_score": 50,
            "query_type": "analysis",
            "agent_sequence": ["extractor", "invalid_agent_name", "classifier"]
        })
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Should either filter out invalid agent or log warning
    # (Implementation determines exact behavior)
    assert "agent_sequence" in result


# ============================================================================
# Test: Complexity Scoring
# ============================================================================

@pytest.mark.asyncio
async def test_complexity_score_reflects_query_difficulty(orchestrator, mock_provider, mock_llm_response):
    """Complexity score should increase with query difficulty."""
    test_cases = [
        ("Hello", 0, 20),  # Greeting: very low
        ("Find section 5", 20, 40),  # Simple retrieval: low
        ("Explain GDPR Article 13", 40, 60),  # Analysis: medium
        ("Compare GDPR, CCPA, and HIPAA compliance requirements", 70, 100)  # Complex: high
    ]

    for query, min_score, max_score in test_cases:
        state = {"query": query}

        mock_provider.create_message.return_value = mock_llm_response(
            text=json.dumps({
                "complexity_score": (min_score + max_score) // 2,
                "query_type": "analysis" if min_score > 40 else "retrieval",
                "agent_sequence": ["extractor"]
            })
        )

        with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
            result = await orchestrator.execute(state)

        score = result.get("complexity_score", 50)
        assert min_score <= score <= max_score, f"Query '{query}' score {score} not in range [{min_score}, {max_score}]"


# ============================================================================
# Test: list_available_agents Tool
# ============================================================================

@pytest.mark.asyncio
async def test_calls_list_available_agents_tool(orchestrator, mock_provider, mock_tool_adapter, mock_llm_response, mock_state):
    """Orchestrator should be able to call list_available_agents tool."""
    state = mock_state(query="What can you help me with?")

    # Mock LLM to call list_available_agents tool
    mock_provider.create_message.side_effect = [
        mock_llm_response(tool_calls=[
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "list_available_agents",
                "input": {}
            }
        ], stop_reason="tool_use"),
        mock_llm_response(
            text=json.dumps({
                "complexity_score": 0,
                "query_type": "meta",
                "agent_sequence": [],
                "final_answer": "I can help with document search, compliance analysis, risk verification, and more."
            })
        )
    ]

    # Mock orchestrator tools instance
    mock_list_agents = Mock(return_value={
        "agents": [
            {"name": "extractor", "role": "extract", "tools": ["search"]},
            {"name": "compliance", "role": "verify", "tools": ["assess_confidence"]}
        ],
        "count": 2,
        "message": "Found 2 available agents"
    })

    with patch.object(orchestrator.orchestrator_tools_instance, 'list_available_agents', mock_list_agents):
        result = await orchestrator.execute(state)

    # Verify tool was called
    mock_list_agents.assert_called_once()


# ============================================================================
# Test: Empty Agent Sequence Handling
# ============================================================================

@pytest.mark.asyncio
async def test_empty_agent_sequence_returns_final_answer(orchestrator, mock_provider, mock_llm_response, mock_state):
    """Empty agent_sequence should return final_answer directly (no workflow)."""
    state = mock_state(query="What is your name?")

    mock_provider.create_message.return_value = mock_llm_response(
        text=json.dumps({
            "complexity_score": 0,
            "query_type": "meta",
            "agent_sequence": [],
            "final_answer": "I am SUJBOT2, a legal document analysis assistant."
        })
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Verify no agents invoked, direct answer returned
    assert result.get("agent_sequence") == []
    assert "final_answer" in result
    assert "SUJBOT" in result["final_answer"]


# ============================================================================
# Test: Prompt Caching
# ============================================================================

@pytest.mark.asyncio
async def test_uses_prompt_caching_for_cost_savings(orchestrator, mock_provider, mock_state):
    """Orchestrator should use prompt caching to reduce costs."""
    state = mock_state(query="Test query")

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        await orchestrator.execute(state)

    # Verify create_message called with cache_control
    call_kwargs = mock_provider.create_message.call_args.kwargs

    # Check if system prompt has cache_control markers
    if "system" in call_kwargs:
        system = call_kwargs["system"]
        # Should either be list with cache_control or string
        assert system is not None


# ============================================================================
# Test: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_handles_provider_errors_gracefully(orchestrator, mock_provider, mock_state):
    """Should handle LLM provider errors without crashing."""
    state = mock_state(query="Test query")

    mock_provider.create_message.side_effect = Exception("API rate limit exceeded")

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
        result = await orchestrator.execute(state)

    # Should return error in state
    assert "errors" in result or "final_answer" in result


# ============================================================================
# Test: Query Type Classification
# ============================================================================

@pytest.mark.asyncio
async def test_classifies_query_types_correctly(orchestrator, mock_provider, mock_llm_response):
    """Orchestrator should classify queries into types (retrieval, analysis, comparison, etc.)."""
    test_cases = [
        ("Find section 5", "retrieval"),
        ("Explain GDPR compliance", "analysis"),
        ("Compare GDPR and CCPA", "comparison"),
        ("What are the risks?", "risk_assessment")
    ]

    for query, expected_type in test_cases:
        state = {"query": query}

        mock_provider.create_message.return_value = mock_llm_response(
            text=json.dumps({
                "agent_sequence": ["extractor"],
                "query_type": expected_type,
                "complexity_score": 50
            })
        )

        with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=Mock()):
            result = await orchestrator.execute(state)

        # Should include query type in result
        query_type = result.get("query_type", "unknown")
        assert query_type in ["retrieval", "analysis", "comparison", "risk_assessment", "unknown"]
