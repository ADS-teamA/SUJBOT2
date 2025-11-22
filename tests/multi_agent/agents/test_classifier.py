"""
Tests for Classifier agent.

The Classifier agent is responsible for analyzing query complexity,
document types, and determining the appropriate processing strategy.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from src.multi_agent.agents.classifier import ClassifierAgent
from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def classifier_config():
    """Configuration for Classifier agent."""
    return AgentConfig(
        name="classifier",
        role=AgentRole.CLASSIFY,
        tier=AgentTier.WORKER,
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.0,  # Deterministic classification
        tools={
            "get_document_info",
            "list_available_documents",
            "similarity_search"
        },
        enable_prompt_caching=True
    )


@pytest.fixture
def classifier_agent(classifier_config, mock_anthropic_provider):
    """Create Classifier agent instance."""
    with patch("builtins.open", mock_open(read_data="You are a classifier agent.")):
        agent = ClassifierAgent(config=classifier_config)
        # Override provider after initialization
        agent.provider = mock_anthropic_provider
    return agent


# ============================================================================
# Configuration Tests
# ============================================================================

def test_classifier_has_correct_tools(classifier_agent):
    """Classifier should have document inspection tools."""
    expected_tools = {
        "get_document_info",
        "list_available_documents",
        "similarity_search"
    }
    assert classifier_agent.config.tools == expected_tools


def test_classifier_role_is_classify(classifier_agent):
    """Classifier should have CLASSIFY role."""
    assert classifier_agent.config.role == AgentRole.CLASSIFY


def test_classifier_tier_is_worker(classifier_agent):
    """Classifier should be WORKER tier."""
    assert classifier_agent.config.tier == AgentTier.WORKER


def test_classifier_uses_zero_temperature(classifier_config):
    """Classifier should use temperature=0.0 for deterministic classification."""
    assert classifier_config.temperature == 0.0


def test_classifier_loads_prompt_from_file():
    """Classifier should load system prompt from prompts/agents/classifier.txt."""
    mock_prompt = "You are a classifier agent specialized in document categorization."

    with patch("builtins.open", mock_open(read_data=mock_prompt)):
        config = AgentConfig(
            name="classifier",
            role=AgentRole.CLASSIFY,
            tier=AgentTier.WORKER,
            model="claude-sonnet-4-5-20250929",
            tools={"get_document_info"}
        )
        agent = ClassifierAgent(config=config, provider=Mock())

    assert agent.system_prompt == mock_prompt


# ============================================================================
# Autonomous Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_classifier_uses_autonomous_tool_loop(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """CRITICAL: Classifier must use autonomous loop, NOT hardcoded flow."""
    state = mock_state(query="What type of document is doc1?")

    # Mock LLM to autonomously call get_document_info
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_document_info",
                "input": {"document_id": "doc1"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Document doc1 is a legal regulation (GDPR).",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    # Verify autonomous behavior
    agent_output = result["agent_outputs"]["classifier"]
    assert "tool_calls" in agent_output
    assert len(agent_output["tool_calls"]) == 1
    assert agent_output["tool_calls"][0]["tool"] == "get_document_info"
    assert "legal regulation" in agent_output["final_answer"]


@pytest.mark.asyncio
async def test_classifier_can_inspect_multiple_documents(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Classifier should autonomously inspect multiple documents if needed."""
    state = mock_state(query="Compare document types of doc1, doc2, and doc3")

    # Mock LLM to call get_document_info multiple times
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_document_info",
                "input": {"document_id": "doc1"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "get_document_info",
                "input": {"document_id": "doc2"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_3",
                "name": "get_document_info",
                "input": {"document_id": "doc3"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="doc1 is a regulation, doc2 is a standard, doc3 is a guideline.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert len(agent_output["tool_calls"]) == 3
    assert all(tc["tool"] == "get_document_info" for tc in agent_output["tool_calls"])


@pytest.mark.asyncio
async def test_classifier_can_list_documents_before_inspection(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Classifier should autonomously list documents before inspection if needed."""
    state = mock_state(query="What types of documents are available?")

    # Mock LLM to call list_available_documents first
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "list_available_documents",
                "input": {}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Available documents: GDPR (regulation), ISO27001 (standard), PCI-DSS (guideline).",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert agent_output["tool_calls"][0]["tool"] == "list_available_documents"


# ============================================================================
# Classification Logic Tests
# ============================================================================

@pytest.mark.asyncio
async def test_classifier_analyzes_query_complexity(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Classifier should analyze and report query complexity."""
    state = mock_state(
        query="Compare GDPR and CCPA data retention requirements, identify gaps, and assess compliance risks for a healthcare organization."
    )

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="High complexity query (score: 85). Requires multi-document comparison, gap analysis, and risk assessment.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert "complexity" in agent_output["final_answer"].lower() or "high" in agent_output["final_answer"].lower()


@pytest.mark.asyncio
async def test_classifier_identifies_document_types(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Classifier should identify document types from metadata."""
    state = mock_state(query="Classify document doc1")

    # Mock get_document_info to return metadata
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=True,
        data={
            "document_id": "doc1",
            "title": "General Data Protection Regulation",
            "metadata": {"type": "regulation", "jurisdiction": "EU"}
        }
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_document_info",
                "input": {"document_id": "doc1"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Document is an EU regulation (GDPR).",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert "regulation" in agent_output["final_answer"].lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_classifier_handles_missing_document(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Classifier should handle missing document gracefully."""
    state = mock_state(query="Classify document nonexistent")

    # Mock tool to return error
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=False,
        error="Document 'nonexistent' not found"
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_document_info",
                "input": {"document_id": "nonexistent"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Document not found in system.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    # Should surface error
    assert "errors" in result
    assert any("get_document_info" in err for err in result["errors"])


@pytest.mark.asyncio
async def test_classifier_handles_empty_document_list(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Classifier should handle empty document list gracefully."""
    state = mock_state(query="What documents are available?")

    # Mock empty document list
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=True,
        data={"documents": []}
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "list_available_documents",
                "input": {}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="No documents are currently indexed in the system.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert "no documents" in agent_output["final_answer"].lower()


# ============================================================================
# State Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_classifier_preserves_previous_agent_outputs(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Classifier should preserve outputs from previous agents."""
    state = mock_state(query="Test query")
    state["agent_outputs"] = {
        "extractor": {"documents_retrieved": 5}
    }

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Classification complete.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    # Should preserve extractor output
    assert "extractor" in result["agent_outputs"]
    assert result["agent_outputs"]["extractor"]["documents_retrieved"] == 5

    # Should add classifier output
    assert "classifier" in result["agent_outputs"]


@pytest.mark.asyncio
async def test_classifier_adds_execution_metadata(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Classifier should add execution metadata to output."""
    state = mock_state(query="Test query")

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Classification complete.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    agent_output = result["agent_outputs"]["classifier"]
    assert "iterations" in agent_output
    assert "tool_calls" in agent_output
    assert "final_answer" in agent_output


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_classifier_full_classification_flow(
    classifier_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """End-to-end test of Classifier agent classification flow."""
    state = mock_state(
        query="Analyze the complexity of comparing GDPR and ISO27001"
    )
    state["agent_outputs"] = {
        "extractor": {"documents": ["gdpr", "iso27001"]}
    }

    # Mock document inspection
    mock_tool_adapter.execute.side_effect = [
        mock_tool_result(
            success=True,
            data={
                "document_id": "gdpr",
                "metadata": {"type": "regulation", "pages": 88}
            }
        ),
        mock_tool_result(
            success=True,
            data={
                "document_id": "iso27001",
                "metadata": {"type": "standard", "pages": 25}
            }
        )
    ]

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_document_info",
                "input": {"document_id": "gdpr"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "get_document_info",
                "input": {"document_id": "iso27001"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Medium-high complexity. Comparing regulation (88 pages) with standard (25 pages). Requires cross-document analysis.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await classifier_agent.execute(state)

    # Verify complete flow
    agent_output = result["agent_outputs"]["classifier"]
    assert len(agent_output["tool_calls"]) == 2
    assert "complexity" in agent_output["final_answer"].lower()
    assert mock_tool_adapter.execute.call_count == 2
