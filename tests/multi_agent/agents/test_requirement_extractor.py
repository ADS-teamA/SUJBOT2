"""
Tests for RequirementExtractorAgent.

The RequirementExtractorAgent is responsible for extracting atomic legal requirements
from legal texts and generating structured JSON checklists for compliance verification.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, mock_open
from src.multi_agent.agents.requirement_extractor import RequirementExtractorAgent
from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier
from src.multi_agent.core.agent_registry import get_agent_registry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def requirement_extractor_config():
    """Configuration for RequirementExtractor agent."""
    return AgentConfig(
        name="requirement_extractor",
        role=AgentRole.EXTRACT,  # Extracts requirements from legal texts
        tier=AgentTier.WORKER,
        model="claude-haiku-4-5",
        max_tokens=4096,  # Higher for JSON checklist
        temperature=0.2,  # Low temp for structured output
        tools={
            "hierarchical_search",
            "graph_search",
            "definition_aligner",
            "multi_doc_synthesizer"
        },
        enable_prompt_caching=True
    )


@pytest.fixture
def requirement_extractor_agent(requirement_extractor_config, mock_anthropic_provider):
    """Create RequirementExtractor agent instance."""
    with patch("builtins.open", mock_open(read_data="You are a requirement extractor agent.")):
        agent = RequirementExtractorAgent(config=requirement_extractor_config)
        # Note: Agent uses autonomous pattern, no provider initialization needed
    return agent


@pytest.fixture
def sample_checklist_json():
    """Sample valid JSON checklist output."""
    return {
        "checklist": [
            {
                "requirement_id": "REQ-001",
                "requirement_text": "Temperature monitoring must record readings every 60 seconds",
                "source_citation": "Vyhláška 157/2025 § 5.2",
                "granularity_level": "CONTENT",
                "severity": "CRITICAL",
                "applicability": "MANDATORY",
                "verification_guidance": "Check temperature log timestamps for 60-second intervals"
            },
            {
                "requirement_id": "REQ-002",
                "requirement_text": "Emergency plan must reference nuclear safety authority contact",
                "source_citation": "Vyhláška 157/2025 § 8.1",
                "granularity_level": "REFERENCE",
                "severity": "HIGH",
                "applicability": "MANDATORY",
                "verification_guidance": "Search for 'SÚJB' or 'nuclear safety authority' in emergency plan"
            }
        ]
    }


# ============================================================================
# Configuration Tests
# ============================================================================

def test_requirement_extractor_has_correct_tools(requirement_extractor_agent):
    """RequirementExtractor should have legal text processing tools."""
    expected_tools = {
        "hierarchical_search",
        "graph_search",
        "definition_aligner",
        "multi_doc_synthesizer"
    }
    assert requirement_extractor_agent.config.tools == expected_tools


def test_requirement_extractor_role_is_extract(requirement_extractor_agent):
    """RequirementExtractor should have EXTRACT role."""
    assert requirement_extractor_agent.config.role == AgentRole.EXTRACT


def test_requirement_extractor_tier_is_worker(requirement_extractor_agent):
    """RequirementExtractor should be WORKER tier."""
    assert requirement_extractor_agent.config.tier == AgentTier.WORKER


def test_requirement_extractor_loads_prompt_from_file(requirement_extractor_agent):
    """RequirementExtractor should load system prompt from file and not be empty."""
    # Check that prompt was loaded (not empty)
    assert requirement_extractor_agent.system_prompt is not None
    assert len(requirement_extractor_agent.system_prompt) > 0
    # Check it's the mocked prompt from the fixture
    assert requirement_extractor_agent.system_prompt == "You are a requirement extractor agent."


def test_requirement_extractor_registered_in_agent_registry():
    """RequirementExtractor should be registered with name 'requirement_extractor'."""
    registry = get_agent_registry()
    assert "requirement_extractor" in registry


# ============================================================================
# JSON Parsing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_parses_valid_json_checklist(requirement_extractor_agent, sample_checklist_json):
    """Should successfully parse valid JSON checklist from LLM response."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        # Mock autonomous loop to return JSON checklist
        mock_loop.return_value = {
            "final_answer": json.dumps(sample_checklist_json),
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check parsing succeeded
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["parsed_successfully"] is True
        assert req_output["parse_error"] is None
        assert len(req_output["checklist"]) == 2
        assert req_output["checklist"][0]["requirement_id"] == "REQ-001"


@pytest.mark.asyncio
async def test_parses_json_with_markdown_code_blocks(requirement_extractor_agent, sample_checklist_json):
    """Should handle JSON wrapped in markdown code blocks (```json...```)."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        # LLM wraps JSON in markdown code block
        markdown_wrapped = f"```json\n{json.dumps(sample_checklist_json)}\n```"
        mock_loop.return_value = {
            "final_answer": markdown_wrapped,
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check parsing succeeded despite markdown wrapper
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["parsed_successfully"] is True
        assert len(req_output["checklist"]) == 2


@pytest.mark.asyncio
async def test_handles_invalid_json_gracefully(requirement_extractor_agent):
    """Should handle invalid JSON without crashing, set parse_error."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        # LLM returns invalid JSON
        mock_loop.return_value = {
            "final_answer": "This is not valid JSON { broken",
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check parsing failed gracefully
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["parsed_successfully"] is False
        assert req_output["parse_error"] is not None
        assert "JSON parsing error" in req_output["parse_error"]
        assert req_output["checklist"] == []  # Empty checklist on parse failure
        assert req_output["raw_answer"] == "This is not valid JSON { broken"  # Preserved for debugging


@pytest.mark.asyncio
async def test_handles_missing_checklist_field(requirement_extractor_agent):
    """Should detect when JSON is valid but missing 'checklist' field."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        # Valid JSON but wrong structure
        wrong_structure = json.dumps({"requirements": [], "metadata": {}})
        mock_loop.return_value = {
            "final_answer": wrong_structure,
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check validation failed
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["parsed_successfully"] is False
        assert "missing 'checklist' field" in req_output["parse_error"]
        assert req_output["checklist"] == []


@pytest.mark.asyncio
async def test_preserves_raw_answer_for_debugging(requirement_extractor_agent, sample_checklist_json):
    """Should preserve raw LLM answer in output for debugging purposes."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        raw_json = json.dumps(sample_checklist_json)
        mock_loop.return_value = {
            "final_answer": raw_json,
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check raw answer preserved
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["raw_answer"] == raw_json
        assert "raw_answer" in req_output  # Always present


# ============================================================================
# State Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_updates_state_with_requirement_extractor_output(requirement_extractor_agent, sample_checklist_json):
    """Should add 'requirement_extractor' to state['agent_outputs']."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        mock_loop.return_value = {
            "final_answer": json.dumps(sample_checklist_json),
            "tool_calls": [{"tool": "hierarchical_search", "args": {}}],
            "iterations": 3,
            "total_tool_cost_usd": 0.005
        }

        state = {"query": "Test query", "agent_outputs": {}}
        result = await requirement_extractor_agent.execute_impl(state)

        # Check state structure
        assert "agent_outputs" in result
        assert "requirement_extractor" in result["agent_outputs"]
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert "checklist" in req_output
        assert "tool_calls_made" in req_output
        assert "iterations" in req_output
        assert "total_tool_cost_usd" in req_output


@pytest.mark.asyncio
async def test_does_not_set_final_answer_in_state(requirement_extractor_agent, sample_checklist_json):
    """Should NOT set final_answer in state (intermediate step, ComplianceAgent consumes output)."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        mock_loop.return_value = {
            "final_answer": json.dumps(sample_checklist_json),
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Should NOT have final_answer (intermediate step)
        assert "final_answer" not in result


@pytest.mark.asyncio
async def test_tracks_tool_calls_made(requirement_extractor_agent, sample_checklist_json):
    """Should track which tools were called during autonomous execution."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        mock_loop.return_value = {
            "final_answer": json.dumps(sample_checklist_json),
            "tool_calls": [
                {"tool": "hierarchical_search", "args": {}},
                {"tool": "definition_aligner", "args": {}},
                {"tool": "graph_search", "args": {}}
            ],
            "iterations": 5,
            "total_tool_cost_usd": 0.01
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["tool_calls_made"] == [
            "hierarchical_search",
            "definition_aligner",
            "graph_search"
        ]
        assert req_output["iterations"] == 5


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_handles_autonomous_loop_exception_gracefully(requirement_extractor_agent):
    """Should handle exceptions from autonomous tool loop without crashing."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        # Simulate tool loop failure
        mock_loop.side_effect = Exception("Tool execution failed")

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Should have error in state, not crash
        assert "errors" in result
        assert any("RequirementExtractor error" in err for err in result["errors"])
        # Should still return state (graceful degradation)
        assert "query" in result


@pytest.mark.asyncio
async def test_handles_empty_final_answer(requirement_extractor_agent):
    """Should handle case where LLM returns empty final_answer."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        mock_loop.return_value = {
            "final_answer": "",  # Empty response
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.0
        }

        state = {"query": "Test query"}
        result = await requirement_extractor_agent.execute_impl(state)

        # Should handle gracefully
        req_output = result["agent_outputs"]["requirement_extractor"]
        assert req_output["parsed_successfully"] is False
        assert req_output["checklist"] == []


# ============================================================================
# Integration with Extractor Agent Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reads_extractor_output_from_state(requirement_extractor_agent, sample_checklist_json):
    """Should be able to access previous extractor agent output from state."""
    with patch.object(
        requirement_extractor_agent,
        "_run_autonomous_tool_loop",
        new_callable=AsyncMock
    ) as mock_loop:
        mock_loop.return_value = {
            "final_answer": json.dumps(sample_checklist_json),
            "tool_calls": [],
            "iterations": 1,
            "total_tool_cost_usd": 0.001
        }

        # State includes extractor output (workflow: extractor → requirement_extractor)
        state = {
            "query": "Test query",
            "agent_outputs": {
                "extractor": {
                    "documents_retrieved": ["doc1.pdf"],
                    "chunks_retrieved": 5
                }
            }
        }
        result = await requirement_extractor_agent.execute_impl(state)

        # Should preserve extractor output
        assert "extractor" in result["agent_outputs"]
        assert "requirement_extractor" in result["agent_outputs"]
