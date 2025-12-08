"""
Tests for Compliance agent.

The Compliance agent is responsible for analyzing compliance requirements,
checking regulatory adherence, and identifying compliance gaps.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from src.multi_agent.agents.compliance import ComplianceAgent
from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def compliance_config():
    """Configuration for Compliance agent."""
    return AgentConfig(
        name="compliance",
        role=AgentRole.VERIFY,
        tier=AgentTier.WORKER,
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=0.0,  # Deterministic compliance checking
        tools={
            "hierarchical_search",
            "similarity_search",
            "graph_search",
            "hybrid_search",
            "get_document_info",
            "assess_confidence"
        },
        enable_prompt_caching=True
    )


@pytest.fixture
def compliance_agent(compliance_config, mock_anthropic_provider):
    """Create Compliance agent instance."""
    with patch("builtins.open", mock_open(read_data="You are a compliance agent.")):
        agent = ComplianceAgent(config=compliance_config)
        # Override provider after initialization
        agent.provider = mock_anthropic_provider
    return agent


# ============================================================================
# Configuration Tests
# ============================================================================

def test_compliance_has_correct_tools(compliance_agent):
    """Compliance should have comprehensive search and verification tools."""
    expected_tools = {
        "hierarchical_search",
        "similarity_search",
        "graph_search",
        "hybrid_search",
        "get_document_info",
        "assess_confidence"
    }
    assert compliance_agent.config.tools == expected_tools


def test_compliance_role_is_verify(compliance_agent):
    """Compliance should have VERIFY role."""
    assert compliance_agent.config.role == AgentRole.VERIFY


def test_compliance_tier_is_worker(compliance_agent):
    """Compliance should be WORKER tier."""
    assert compliance_agent.config.tier == AgentTier.WORKER


def test_compliance_uses_zero_temperature(compliance_config):
    """Compliance should use temperature=0.0 for deterministic checking."""
    assert compliance_config.temperature == 0.0


def test_compliance_loads_prompt_from_file():
    """Compliance should load system prompt from prompts/agents/compliance.txt."""
    mock_prompt = "You are a compliance agent specialized in regulatory analysis."

    with patch("builtins.open", mock_open(read_data=mock_prompt)):
        config = AgentConfig(
            name="compliance",
            role=AgentRole.VERIFY,
            tier=AgentTier.WORKER,
            model="claude-sonnet-4-5-20250929",
            tools={"hierarchical_search"}
        )
        agent = ComplianceAgent(config=config, provider=Mock())

    assert agent.system_prompt == mock_prompt


# ============================================================================
# Autonomous Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_uses_autonomous_tool_loop(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """CRITICAL: Compliance must use autonomous loop, NOT hardcoded flow."""
    state = mock_state(query="Check GDPR compliance for data retention")

    # Mock LLM to autonomously call hierarchical_search → assess_confidence
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR data retention requirements", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "assess_confidence",
                "input": {"query": "data retention", "sources": ["gdpr:sec5:0"]}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="GDPR Article 5 requires data minimization and storage limitation. Confidence: high.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    # Verify autonomous behavior
    agent_output = result["agent_outputs"]["compliance"]
    assert "tool_calls" in agent_output
    assert len(agent_output["tool_calls"]) == 2
    assert agent_output["tool_calls"][0]["tool"] == "hierarchical_search"
    assert agent_output["tool_calls"][1]["tool"] == "assess_confidence"


@pytest.mark.asyncio
async def test_compliance_can_use_multiple_search_strategies(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Compliance should autonomously choose optimal search strategy."""
    state = mock_state(query="Find requirements related to 'data controller' and 'data processor'")

    # Mock LLM to try semantic search → graph search for entities
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "similarity_search",
                "input": {"query": "data controller data processor", "top_k": 5}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "graph_search",
                "input": {"entity_name": "data controller"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Found definitions and relationships between data controllers and processors.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert len(agent_output["tool_calls"]) == 2
    assert agent_output["tool_calls"][0]["tool"] == "similarity_search"
    assert agent_output["tool_calls"][1]["tool"] == "graph_search"


@pytest.mark.asyncio
async def test_compliance_can_verify_with_confidence_assessment(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Compliance should assess confidence in compliance findings."""
    state = mock_state(query="Is our data retention policy GDPR compliant?")

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR data retention policy", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "assess_confidence",
                "input": {
                    "query": "data retention policy compliance",
                    "sources": ["gdpr:sec5:0", "gdpr:sec5:1"]
                }
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="High confidence: GDPR Article 5(1)(e) requires storage limitation. Confidence score: 0.95.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert any(tc["tool"] == "assess_confidence" for tc in agent_output["tool_calls"])
    assert "confidence" in agent_output["final_answer"].lower()


# ============================================================================
# Compliance Checking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_identifies_requirements(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Compliance should identify specific compliance requirements."""
    state = mock_state(query="What are GDPR breach notification requirements?")

    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=True,
        data={
            "layer3": [
                {
                    "chunk_id": "gdpr:sec33:0",
                    "text": "A personal data breach must be notified to the supervisory authority without undue delay and, where feasible, not later than 72 hours.",
                    "relevance_score": 0.98
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
                "input": {"query": "GDPR breach notification", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="GDPR Article 33 requires breach notification within 72 hours.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert "72 hours" in agent_output["final_answer"] or "article 33" in agent_output["final_answer"].lower()


@pytest.mark.asyncio
async def test_compliance_checks_multiple_regulations(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Compliance should check multiple regulations when needed."""
    state = mock_state(
        query="Compare consent requirements in GDPR and CCPA"
    )
    state["agent_outputs"] = {
        "classifier": {"documents": ["gdpr", "ccpa"]}
    }

    # Mock LLM to search both regulations
    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR consent requirements", "top_k": 5}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "hierarchical_search",
                "input": {"query": "CCPA consent requirements", "top_k": 5}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="GDPR requires explicit consent (opt-in), CCPA allows opt-out approach.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert len(agent_output["tool_calls"]) >= 2
    assert "GDPR" in agent_output["final_answer"] and "CCPA" in agent_output["final_answer"]


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_handles_search_failure(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Compliance should handle search failures gracefully."""
    state = mock_state(query="Check compliance requirements")

    # Mock tool failure
    mock_tool_adapter.execute.return_value = mock_tool_result(
        success=False,
        error="Vector store connection timeout"
    )

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "compliance requirements", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="Unable to retrieve compliance requirements due to system error.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    # Should surface critical tool failure
    assert "errors" in result
    assert any("hierarchical_search" in err for err in result["errors"])


@pytest.mark.asyncio
async def test_compliance_handles_low_confidence_results(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """Compliance should flag low-confidence findings."""
    state = mock_state(query="Check ambiguous compliance requirement")

    mock_tool_adapter.execute.side_effect = [
        mock_tool_result(
            success=True,
            data={"layer3": [{"text": "Some text", "relevance_score": 0.45}]}
        ),
        mock_tool_result(
            success=True,
            data={"confidence_score": 0.3, "confidence_level": "low"}
        )
    ]

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "ambiguous requirement", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "assess_confidence",
                "input": {"query": "ambiguous requirement", "sources": []}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="WARNING: Low confidence (0.3) in these findings. Recommend manual review.",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert "low confidence" in agent_output["final_answer"].lower() or "warning" in agent_output["final_answer"].lower()


# ============================================================================
# State Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_uses_previous_agent_context(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Compliance should use context from previous agents (extractor, classifier)."""
    state = mock_state(query="Check compliance")
    state["agent_outputs"] = {
        "extractor": {"documents_retrieved": ["gdpr", "iso27001"]},
        "classifier": {"complexity": "medium", "document_types": ["regulation", "standard"]}
    }

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Checking compliance for regulation and standard documents.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    # Should preserve previous outputs
    assert "extractor" in result["agent_outputs"]
    assert "classifier" in result["agent_outputs"]
    assert "compliance" in result["agent_outputs"]


@pytest.mark.asyncio
async def test_compliance_adds_execution_metadata(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_state
):
    """Compliance should add execution metadata to output."""
    state = mock_state(query="Test query")

    mock_anthropic_provider.create_message.return_value = mock_llm_response(
        text="Compliance check complete.",
        stop_reason="end_turn"
    )

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    agent_output = result["agent_outputs"]["compliance"]
    assert "iterations" in agent_output
    assert "tool_calls" in agent_output
    assert "final_answer" in agent_output


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_full_verification_flow(
    compliance_agent,
    mock_anthropic_provider,
    mock_llm_response,
    mock_tool_adapter,
    mock_tool_result,
    mock_state
):
    """End-to-end test of Compliance agent verification flow."""
    state = mock_state(
        query="Verify GDPR compliance for our cookie consent implementation"
    )
    state["agent_outputs"] = {
        "extractor": {"documents": ["gdpr"]},
        "classifier": {"document_type": "regulation", "complexity": "medium"}
    }

    # Mock comprehensive compliance check
    mock_tool_adapter.execute.side_effect = [
        # hierarchical_search for cookie consent
        mock_tool_result(
            success=True,
            data={
                "layer3": [
                    {
                        "chunk_id": "gdpr:sec4:11",
                        "text": "Consent must be freely given, specific, informed and unambiguous.",
                        "relevance_score": 0.96
                    }
                ]
            }
        ),
        # assess_confidence
        mock_tool_result(
            success=True,
            data={"confidence_score": 0.92, "confidence_level": "high"}
        ),
        # get_document_info for citation
        mock_tool_result(
            success=True,
            data={"document_id": "gdpr", "title": "General Data Protection Regulation"}
        )
    ]

    mock_anthropic_provider.create_message.side_effect = [
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "hierarchical_search",
                "input": {"query": "GDPR cookie consent requirements", "top_k": 10}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_2",
                "name": "assess_confidence",
                "input": {"query": "cookie consent", "sources": ["gdpr:sec4:11"]}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            tool_calls=[{
                "type": "tool_use",
                "id": "call_3",
                "name": "get_document_info",
                "input": {"document_id": "gdpr"}
            }],
            stop_reason="tool_use"
        ),
        mock_llm_response(
            text="GDPR Article 4(11) requires consent to be freely given, specific, informed and unambiguous. Cookie consent must meet these criteria. Confidence: high (0.92).",
            stop_reason="end_turn"
        )
    ]

    with patch("src.multi_agent.tools.adapter.get_tool_adapter", return_value=mock_tool_adapter):
        result = await compliance_agent.execute(state)

    # Verify complete flow
    agent_output = result["agent_outputs"]["compliance"]
    assert len(agent_output["tool_calls"]) == 3
    assert "consent" in agent_output["final_answer"].lower()
    assert "confidence" in agent_output["final_answer"].lower()
    assert mock_tool_adapter.execute.call_count == 3
