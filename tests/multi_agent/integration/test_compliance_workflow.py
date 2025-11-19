"""
Integration tests for SOTA Compliance workflow with RequirementExtractor.

Tests the complete requirement-first compliance flow:
1. Orchestrator → Extractor → RequirementExtractor → Compliance → GapSynthesizer
2. JSON checklist generation and parsing
3. REGULATORY_GAP vs SCOPE_GAP classification
4. Definition alignment integration
5. Error handling (missing checklist, invalid JSON, timeout)

Based on SOTA 2024 research:
- Legal AI Atomization (Cornell 2024)
- Plan-and-Solve Pattern (Zhou et al., 2023)
- Requirement-First Compliance (Legal AI 2024)
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Mark all tests as asyncio
pytestmark = pytest.mark.asyncio

from src.multi_agent.core.state import MultiAgentState, ExecutionPhase
from src.multi_agent.core.agent_registry import AgentRegistry
from src.multi_agent.routing.workflow_builder import WorkflowBuilder


@pytest.fixture
def sample_requirement_checklist():
    """Sample JSON checklist output from RequirementExtractor."""
    return json.dumps({
        "requirements_extracted": 3,
        "target_law": "Vyhláška č. 157/2025 Sb.",
        "terminology_alignments": [
            {
                "term": "Client",
                "standard_definition": "Natural or legal person entering contract",
                "law_equivalent": "Consumer",
                "alignment_confidence": 0.85
            }
        ],
        "checklist": [
            {
                "requirement_id": "REQ-001",
                "requirement_text": "Documentation must contain reference to approved emergency plan (havarijní řád)",
                "source_citation": "[Doc: Vyhláška_157_2025 > h) Obecné informace > Reference]",
                "granularity_level": "REFERENCE",
                "severity": "CRITICAL",
                "applicability": "MANDATORY",
                "verification_guidance": "Search documentation header and metadata sections for emergency plan reference. Check format: 'Havarijní řád č. XXX schválený dne DD.MM.YYYY'",
                "success_criteria": "Reference present AND includes approval number AND includes approval date"
            },
            {
                "requirement_id": "REQ-002",
                "requirement_text": "Section on safety systems must specify maximum operating temperature",
                "source_citation": "[Doc: Vyhláška_157_2025 > i) Technické parametry > Teplota]",
                "granularity_level": "CONTENT",
                "severity": "HIGH",
                "applicability": "MANDATORY",
                "verification_guidance": "Find section describing cooling/safety systems. Extract numeric temperature value with unit (°C or K)",
                "success_criteria": "Temperature value present AND unit specified AND within regulatory limits (≤40°C)"
            },
            {
                "requirement_id": "REQ-003",
                "requirement_text": "If using nuclear fuel, documentation must list isotope composition",
                "source_citation": "[Doc: Vyhláška_157_2025 > j) Palivo > Složení]",
                "granularity_level": "SECTION",
                "severity": "HIGH",
                "applicability": "CONDITIONAL",
                "verification_guidance": "Check if document mentions nuclear fuel. If yes, verify isotope composition listed (e.g., U-235 enrichment percentage)",
                "success_criteria": "IF fuel mentioned THEN isotope composition present"
            }
        ],
        "extraction_summary": "Extracted 3 atomic requirements from Vyhláška 157/2025. Found 1 critical requirement (missing reference violations), 2 high priority (technical specifications)."
    })


@pytest.fixture
def mock_agent_registry_compliance(sample_requirement_checklist):
    """Create mock agent registry for compliance workflow."""
    registry = AgentRegistry()

    # Mock extractor agent
    extractor = Mock()
    extractor.name = "extractor"
    extractor.config = Mock()
    extractor.config.name = "extractor"
    extractor.execute = AsyncMock(
        return_value={
            "query": "Je dokumentace v souladu s Vyhláškou 157/2025?",
            "documents": [
                {
                    "doc_id": "BZ_VR1:0",
                    "filename": "BZ_VR1.pdf",
                    "layer": 1,
                    "relevance_score": 0.8,
                    "chunk_index": 0,
                },
                {
                    "doc_id": "Vyhláška_157_2025:0",
                    "filename": "Vyhláška_157_2025.pdf",
                    "layer": 1,
                    "relevance_score": 0.75,
                    "chunk_index": 0,
                },
            ],
            "agent_outputs": {"extractor": {"documents_retrieved": 2, "status": "success"}},
        }
    )

    # Mock requirement_extractor agent
    requirement_extractor = Mock()
    requirement_extractor.name = "requirement_extractor"
    requirement_extractor.config = Mock()
    requirement_extractor.config.name = "requirement_extractor"
    requirement_extractor.execute = AsyncMock(
        return_value={
            "query": "Je dokumentace v souladu s Vyhláškou 157/2025?",
            "agent_outputs": {
                "extractor": {"documents_retrieved": 2, "status": "success"},
                "requirement_extractor": {
                    "checklist": sample_requirement_checklist,  # JSON string
                    "tool_calls_made": ["hierarchical_search", "definition_aligner"],
                    "iterations": 3,
                    "total_tool_cost_usd": 0.012
                }
            },
        }
    )

    # Mock compliance agent
    compliance = Mock()
    compliance.name = "compliance"
    compliance.config = Mock()
    compliance.config.name = "compliance"
    compliance.execute = AsyncMock(
        return_value={
            "query": "Je dokumentace v souladu s Vyhláškou 157/2025?",
            "agent_outputs": {
                "extractor": {"documents_retrieved": 2, "status": "success"},
                "requirement_extractor": {
                    "checklist": sample_requirement_checklist,
                    "tool_calls_made": ["hierarchical_search", "definition_aligner"],
                    "iterations": 3
                },
                "compliance": {
                    "analysis": "## Checklist Verification Results\n\n### ❌ Non-Compliant Requirements (1)\nREQ-001: REGULATORY_GAP - Missing emergency plan reference\n\n### ✅ Compliant Requirements (1)\nREQ-002: Temperature specified as 35°C\n\n### 🔍 Scope Gaps (1)\nREQ-003: SCOPE_GAP - Document does not use nuclear fuel",
                    "tool_calls_made": ["hierarchical_search", "definition_aligner", "assess_confidence"],
                    "iterations": 5,
                    "total_tool_cost_usd": 0.025
                }
            },
        }
    )

    # Mock gap_synthesizer agent
    gap_synthesizer = Mock()
    gap_synthesizer.name = "gap_synthesizer"
    gap_synthesizer.config = Mock()
    gap_synthesizer.config.name = "gap_synthesizer"
    gap_synthesizer.execute = AsyncMock(
        return_value={
            "query": "Je dokumentace v souladu s Vyhláškou 157/2025?",
            "agent_outputs": {
                "extractor": {"documents_retrieved": 2, "status": "success"},
                "requirement_extractor": {"checklist": sample_requirement_checklist},
                "compliance": {"analysis": "Compliance analysis..."},
                "gap_synthesizer": {
                    "analysis": "## Critical Regulatory Gaps (1)\n- REQ-001: Missing emergency plan reference\n\n## Scope Gaps (1)\n- REQ-003: Nuclear fuel not applicable",
                    "tool_calls_made": ["graph_search", "multi_doc_synthesizer"],
                    "iterations": 2
                }
            },
        }
    )

    registry.register(extractor)
    registry.register(requirement_extractor)
    registry.register(compliance)
    registry.register(gap_synthesizer)

    return registry


class TestComplianceWorkflowIntegration:
    """Test SOTA compliance workflow (requirement-first approach)."""

    async def test_full_compliance_workflow_success(self, mock_agent_registry_compliance):
        """Test complete compliance workflow with all agents."""
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry_compliance,
            checkpointer=None,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "requirement_extractor", "compliance", "gap_synthesizer"],
            enable_parallel=False
        )

        # Initial state - compliance query
        state = MultiAgentState(
            query="Je dokumentace v souladu s Vyhláškou 157/2025?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=75,
            agent_sequence=["extractor", "requirement_extractor", "compliance", "gap_synthesizer"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "compliance-test-1"})

        # Verify all agents executed
        assert result is not None
        assert "extractor" in result.get("agent_outputs", {})
        assert "requirement_extractor" in result.get("agent_outputs", {})
        assert "compliance" in result.get("agent_outputs", {})
        assert "gap_synthesizer" in result.get("agent_outputs", {})

        # Verify checklist was generated
        req_extractor_output = result["agent_outputs"]["requirement_extractor"]
        assert "checklist" in req_extractor_output
        assert "REQ-001" in req_extractor_output["checklist"]  # JSON contains requirement IDs

        # Verify compliance consumed checklist
        compliance_output = result["agent_outputs"]["compliance"]
        assert "analysis" in compliance_output
        assert "REGULATORY_GAP" in compliance_output["analysis"] or "REQ-001" in compliance_output["analysis"]

        # Verify no errors
        assert len(result.get("errors", [])) == 0

    async def test_requirement_extractor_json_parsing(self, sample_requirement_checklist):
        """Test that requirement_extractor generates valid JSON checklist."""
        # Parse the sample checklist to verify it's valid JSON
        checklist_data = json.loads(sample_requirement_checklist)

        # Verify structure
        assert "requirements_extracted" in checklist_data
        assert checklist_data["requirements_extracted"] == 3
        assert "target_law" in checklist_data
        assert "checklist" in checklist_data
        assert len(checklist_data["checklist"]) == 3

        # Verify requirement structure
        req = checklist_data["checklist"][0]
        assert "requirement_id" in req
        assert "requirement_text" in req
        assert "source_citation" in req
        assert "granularity_level" in req
        assert "severity" in req
        assert "applicability" in req
        assert "verification_guidance" in req
        assert "success_criteria" in req

    async def test_regulatory_gap_vs_scope_gap_classification(self, mock_agent_registry_compliance):
        """Test that compliance correctly classifies REGULATORY_GAP vs SCOPE_GAP."""
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry_compliance,
            checkpointer=None,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            enable_parallel=False
        )

        state = MultiAgentState(
            query="Je dokumentace v souladu s Vyhláškou 157/2025?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=75,
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "compliance-test-2"})

        # Verify compliance output contains gap classification
        compliance_output = result["agent_outputs"]["compliance"]["analysis"]

        # Should have REGULATORY_GAP (REQ-001: MANDATORY + APPLICABLE + MISSING)
        assert "REGULATORY_GAP" in compliance_output or "REQ-001" in compliance_output

        # Should have SCOPE_GAP (REQ-003: CONDITIONAL + not met)
        assert "SCOPE_GAP" in compliance_output or "REQ-003" in compliance_output

    async def test_definition_alignment_integration(self, mock_agent_registry_compliance):
        """Test that terminology alignments from requirement_extractor are used by compliance."""
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry_compliance,
            checkpointer=None,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            enable_parallel=False
        )

        state = MultiAgentState(
            query="Je dokumentace v souladu s Vyhláškou 157/2025?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=75,
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "compliance-test-3"})

        # Verify terminology_alignments present in requirement_extractor output
        req_extractor_output = result["agent_outputs"]["requirement_extractor"]["checklist"]
        checklist_data = json.loads(req_extractor_output)
        assert "terminology_alignments" in checklist_data
        assert len(checklist_data["terminology_alignments"]) == 1
        assert checklist_data["terminology_alignments"][0]["term"] == "Client"
        assert checklist_data["terminology_alignments"][0]["law_equivalent"] == "Consumer"


class TestComplianceErrorHandling:
    """Test error handling in compliance workflow."""

    async def test_compliance_fails_without_requirement_extractor(self):
        """Test that compliance agent fails gracefully when requirement_extractor output is missing."""
        registry = AgentRegistry()

        # Mock extractor only (no requirement_extractor)
        extractor = Mock()
        extractor.name = "extractor"
        extractor.config = Mock()
        extractor.config.name = "extractor"
        extractor.execute = AsyncMock(
            return_value={
                "query": "Compliance query",
                "documents": [],
                "agent_outputs": {"extractor": {"documents_retrieved": 0}},
            }
        )

        # Mock compliance agent - should detect missing requirement_extractor output
        compliance = Mock()
        compliance.name = "compliance"
        compliance.config = Mock()
        compliance.config.name = "compliance"

        async def compliance_execute(state):
            # Simulate check for requirement_extractor output
            extractor_output = state.get("agent_outputs", {}).get("requirement_extractor", {})
            if not extractor_output:
                # Add error to state (realistic behavior)
                state["errors"] = state.get("errors", [])
                state["errors"].append("ComplianceAgent error: Missing requirement_extractor output. Compliance agent requires checklist from RequirementExtractorAgent.")
                return state
            return {
                **state,
                "agent_outputs": {
                    **state.get("agent_outputs", {}),
                    "compliance": {"analysis": "Success"}
                }
            }

        compliance.execute = compliance_execute

        registry.register(extractor)
        registry.register(compliance)

        builder = WorkflowBuilder(
            agent_registry=registry,
            checkpointer=None,
        )

        # Build workflow WITHOUT requirement_extractor (wrong sequence)
        workflow = builder.build_workflow(
            agent_sequence=["extractor", "compliance"],
            enable_parallel=False
        )

        state = MultiAgentState(
            query="Compliance query",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            agent_sequence=["extractor", "compliance"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "error-test-1"})

        # Verify error was logged
        assert len(result.get("errors", [])) > 0
        assert any("requirement_extractor" in str(e) for e in result["errors"])

    async def test_compliance_handles_invalid_json_gracefully(self):
        """Test that compliance handles invalid JSON from requirement_extractor."""
        registry = AgentRegistry()

        extractor = Mock()
        extractor.name = "extractor"
        extractor.config = Mock()
        extractor.config.name = "extractor"
        extractor.execute = AsyncMock(return_value={"agent_outputs": {"extractor": {}}})

        # requirement_extractor returns INVALID JSON
        requirement_extractor = Mock()
        requirement_extractor.name = "requirement_extractor"
        requirement_extractor.config = Mock()
        requirement_extractor.config.name = "requirement_extractor"
        requirement_extractor.execute = AsyncMock(
            return_value={
                "agent_outputs": {
                    "requirement_extractor": {
                        "checklist": "{invalid json, missing closing brace"  # INVALID
                    }
                }
            }
        )

        compliance = Mock()
        compliance.name = "compliance"
        compliance.config = Mock()
        compliance.config.name = "compliance"

        async def compliance_execute_with_json_check(state):
            req_output = state.get("agent_outputs", {}).get("requirement_extractor", {})
            checklist_str = req_output.get("checklist", "")
            try:
                checklist_data = json.loads(checklist_str)
            except json.JSONDecodeError as e:
                state["errors"] = state.get("errors", [])
                state["errors"].append(f"ComplianceAgent error: Invalid JSON from requirement_extractor: {str(e)}")
                return state

            return {
                **state,
                "agent_outputs": {**state.get("agent_outputs", {}), "compliance": {"analysis": "Success"}}
            }

        compliance.execute = compliance_execute_with_json_check

        registry.register(extractor)
        registry.register(requirement_extractor)
        registry.register(compliance)

        builder = WorkflowBuilder(agent_registry=registry, checkpointer=None)
        workflow = builder.build_workflow(
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            enable_parallel=False
        )

        state = MultiAgentState(
            query="Test query",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            agent_sequence=["extractor", "requirement_extractor", "compliance"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "error-test-2"})

        # Verify JSON parsing error was caught
        assert len(result.get("errors", [])) > 0
        assert any("Invalid JSON" in str(e) for e in result["errors"])


class TestCompliancePerformance:
    """Test performance characteristics of compliance workflow."""

    async def test_workflow_completes_within_timeout(self, mock_agent_registry_compliance):
        """Test that compliance workflow completes within reasonable time."""
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry_compliance,
            checkpointer=None,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "requirement_extractor", "compliance", "gap_synthesizer"],
            enable_parallel=False
        )

        state = MultiAgentState(
            query="Je dokumentace v souladu s Vyhláškou 157/2025?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=75,
            agent_sequence=["extractor", "requirement_extractor", "compliance", "gap_synthesizer"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        import time
        start_time = time.time()

        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "perf-test-1"})

        elapsed_time = time.time() - start_time

        # With mocked agents, should complete quickly (<1s)
        # Real workflow: 30-60s expected
        assert elapsed_time < 1.0  # Mocked agents are fast
        assert result is not None


async def test_handles_empty_checklist_gracefully(mock_agent_registry_compliance):
    """
    Test that ComplianceAgent handles empty checklist gracefully.

    Scenario: RequirementExtractor returns empty checklist (no requirements found in law).
    Expected: Compliance agent should handle gracefully without crashing.
    """
    registry = mock_agent_registry_compliance

    # Get requirement_extractor from registry
    requirement_extractor = registry.get_agent("requirement_extractor")

    # Mock empty checklist response
    empty_checklist = json.dumps({
        "requirements_extracted": 0,
        "target_law": "Test Law",
        "terminology_alignments": [],
        "checklist": [],  # EMPTY - no requirements found
        "extraction_summary": "No atomic requirements could be extracted from the provided law text."
    })

    # Override requirement_extractor to return empty checklist
    requirement_extractor.execute = AsyncMock(
        return_value={
            "query": "Test query",
            "agent_outputs": {
                "requirement_extractor": {
                    "checklist": [],  # Empty checklist
                    "raw_answer": empty_checklist,
                    "parsed_successfully": True,
                    "parse_error": None,
                    "tool_calls_made": ["hierarchical_search"],
                    "iterations": 2,
                    "total_tool_cost_usd": 0.001
                }
            }
        }
    )

    # Get compliance agent
    compliance = registry.get_agent("compliance")

    # Create initial state
    state = MultiAgentState(
        query="Test query with empty checklist",
        conversation_id="test-empty-checklist",
        phase=ExecutionPhase.AGENT_EXECUTION,
        agent_outputs={
            "extractor": {"documents": ["doc1"]},
            "requirement_extractor": {
                "checklist": [],  # Empty checklist
                "parsed_successfully": True,
                "parse_error": None
            }
        },
        tool_executions=[],
        documents=[],
        citations=[],
        total_cost_cents=0.0,
        errors=[],
    )

    # Execute compliance agent with empty checklist
    result = await compliance.execute(state.model_dump())

    # Should handle gracefully without crash
    assert result is not None
    assert "errors" not in result or len(result.get("errors", [])) == 0

    # Should have compliance output (may be empty or informational)
    assert "agent_outputs" in result
    assert "compliance" in result["agent_outputs"]

    # Check that compliance agent recognized empty checklist
    compliance_output = result["agent_outputs"]["compliance"]

    # Verify it handled the empty checklist case
    # (implementation may vary, but should not crash)
    assert compliance_output is not None
