"""
Integration tests for HITL workflow with LangGraph.

Tests the complete flow:
1. Orchestrator → Extractor → HITL Gate → (interrupt or continue)
2. User clarification → Resume → Continue workflow
3. Edge cases: timeouts, max rounds, errors
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Mark all tests as asyncio
pytestmark = pytest.mark.asyncio

from src.multi_agent.core.state import MultiAgentState, ExecutionPhase
from src.multi_agent.core.agent_registry import AgentRegistry
from src.multi_agent.routing.workflow_builder import WorkflowBuilder
from src.multi_agent.hitl.config import HITLConfig
from src.multi_agent.hitl.quality_detector import QualityDetector
from src.multi_agent.hitl.clarification_generator import ClarificationGenerator


@pytest.fixture
def hitl_config():
    """Create HITL config for testing."""
    config_dict = {
        "enabled": True,
        "policy": {
            "trigger_after_agent": "extractor",
            "quality_threshold": 0.60,
            "min_complexity_score": 40,
            "max_clarifications_per_query": 2,
        },
        "quality_detection": {
            "require_multiple_failures": True,
            "min_failing_metrics": 2,
            "metrics": {
                "retrieval_score": {
                    "enabled": True,
                    "weight": 0.30,
                    "threshold": 0.65,
                },
                "semantic_coherence": {
                    "enabled": True,
                    "weight": 0.25,
                    "threshold": 0.30,
                },
                "query_pattern": {
                    "enabled": True,
                    "weight": 0.25,
                    "threshold": 0.50,
                },
                "document_diversity": {
                    "enabled": True,
                    "weight": 0.20,
                    "threshold": 5.0,
                },
            },
        },
        "question_generation": {
            "model": "claude-haiku-4-5-20251001",
            "min_questions": 2,
            "max_questions": 5,
            "timeout_seconds": 10,
        },
        "user_interaction": {
            "timeout_seconds": 300,
            "max_wait_attempts": 3,
        },
        "query_enrichment": {
            "strategy": "append_with_context",
            "max_enriched_length": 2000,
        },
    }
    return HITLConfig.from_dict(config_dict)


@pytest.fixture
def mock_agent_registry():
    """Create mock agent registry with extractor and classifier."""
    registry = AgentRegistry()

    # Mock extractor agent
    extractor = Mock()
    # Set name as a direct attribute (not nested Mock)
    extractor.name = "extractor"
    extractor.config = Mock()
    extractor.config.name = "extractor"
    extractor.execute = AsyncMock(
        return_value={
            "query": "What is GDPR?",
            "documents": [
                {
                    "doc_id": "doc1:0",
                    "filename": "document1.pdf",
                    "layer": 1,
                    "relevance_score": 0.3,
                    "chunk_index": 0,
                },
                {
                    "doc_id": "doc1:1",
                    "filename": "document1.pdf",
                    "layer": 1,
                    "relevance_score": 0.25,
                    "chunk_index": 1,
                },
            ],
            "agent_outputs": {"extractor": "Retrieved 2 documents"},
        }
    )

    # Mock classifier agent
    classifier = Mock()
    # Set name as a direct attribute (not nested Mock)
    classifier.name = "classifier"
    classifier.config = Mock()
    classifier.config.name = "classifier"
    classifier.execute = AsyncMock(
        return_value={
            "query": "What is GDPR?",
            "agent_outputs": {"classifier": "Classified as regulatory query"},
        }
    )

    registry.register(extractor)
    registry.register(classifier)

    return registry


@pytest.fixture
def mock_quality_detector(hitl_config):
    """Create mock quality detector."""
    detector = QualityDetector(hitl_config)

    # Mock evaluate to return low quality (triggers clarification)
    original_evaluate = detector.evaluate

    def mock_evaluate(query, search_results, complexity_score):
        from src.multi_agent.hitl.quality_detector import QualityMetrics

        # Return low quality metrics
        return (
            True,  # should_clarify = True
            QualityMetrics(
                retrieval_score=0.27,  # Below threshold
                semantic_coherence=0.15,  # Below threshold
                query_pattern_score=0.40,  # Below threshold
                document_diversity=2.0,  # OK
                overall_quality=0.35,  # Low overall
                failing_metrics=["retrieval_score", "semantic_coherence", "query_pattern"],
            ),
        )

    detector.evaluate = mock_evaluate

    return detector


@pytest.fixture
def mock_clarification_generator(hitl_config):
    """Create mock clarification generator."""
    generator = ClarificationGenerator(hitl_config, api_key="test-key")

    # Mock generate to return questions
    async def mock_generate(query, metrics, context):
        from src.multi_agent.hitl.clarification_generator import ClarificationQuestion

        return [
            ClarificationQuestion(
                id="q1",
                text="What specific aspect of GDPR are you interested in?",
                type="scope",  # type is string, not enum
            ),
            ClarificationQuestion(
                id="q2",
                text="Are you looking for compliance requirements or penalties?",
                type="intent",  # type is string, not enum
            ),
        ]

    generator.generate = mock_generate

    return generator


class TestHITLWorkflowIntegration:
    """Test HITL workflow integration with LangGraph."""

    async def test_workflow_triggers_clarification_on_low_quality(
        self,
        hitl_config,
        mock_agent_registry,
        mock_quality_detector,
        mock_clarification_generator,
    ):
        """Test that workflow triggers clarification when quality is low."""
        # Build workflow with HITL components
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=mock_quality_detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # Initial state with low complexity (should trigger HITL)
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,  # Above min threshold
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - HITL gate should be triggered
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-1"})

        # Verify HITL was triggered and questions were generated
        assert result is not None
        assert result.get("quality_check_required") == True, "Quality check should be required"
        assert len(result.get("clarifying_questions", [])) == 2, "Should have 2 questions"
        assert result.get("awaiting_user_input") == True, "Should be awaiting user input"
        assert result.get("clarification_round") == 1, "Should be first clarification round"

        # Verify quality metrics were stored
        quality_metrics = result.get("quality_metrics")
        assert quality_metrics is not None
        assert "overall_quality" in quality_metrics
        assert quality_metrics["overall_quality"] < 0.6, "Quality should be below threshold"

    async def test_workflow_continues_on_acceptable_quality(
        self, hitl_config, mock_agent_registry, mock_clarification_generator
    ):
        """Test that workflow continues when quality is acceptable."""
        # Create quality detector that returns high quality
        detector = QualityDetector(hitl_config)

        def mock_evaluate_high_quality(query, search_results, complexity_score):
            from src.multi_agent.hitl.quality_detector import QualityMetrics

            return (
                False,  # should_clarify = False
                QualityMetrics(
                    retrieval_score=0.85,  # High quality
                    semantic_coherence=0.60,
                    query_pattern_score=0.70,
                    document_diversity=3.0,
                    overall_quality=0.75,
                    failing_metrics=[],
                ),
            )

        detector.evaluate = mock_evaluate_high_quality

        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # Initial state
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should NOT interrupt
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-2"})

        # Verify workflow continued without interruption
        assert result is not None
        assert result.get("quality_check_required") == False
        assert "classifier" in result.get("agent_outputs", {})

    async def test_workflow_skips_hitl_for_low_complexity(
        self,
        hitl_config,
        mock_agent_registry,
        mock_quality_detector,
        mock_clarification_generator,
    ):
        """Test that HITL is skipped when complexity is below threshold."""
        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=mock_quality_detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # State with LOW complexity (below min_complexity_score=40)
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=30,  # Below threshold
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should skip HITL due to low complexity
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-3"})

        # Verify HITL was skipped
        assert result is not None
        assert result.get("quality_check_required") == False
        assert result.get("clarifying_questions") == []

    async def test_workflow_respects_max_clarification_rounds(
        self,
        hitl_config,
        mock_agent_registry,
        mock_quality_detector,
        mock_clarification_generator,
    ):
        """Test that workflow stops clarification after max rounds."""
        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=mock_quality_detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # State with clarification_round = 2 (max is 2)
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            clarification_round=2,  # Already at max
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should NOT trigger clarification (max rounds reached)
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-4"})

        # Verify workflow continued without clarification
        assert result is not None
        assert result.get("quality_check_required") == False
        assert result.get("clarification_round") == 2  # Unchanged

    async def test_hitl_gate_handles_errors_gracefully(
        self, hitl_config, mock_agent_registry, mock_clarification_generator
    ):
        """Test that HITL gate handles errors gracefully and continues workflow."""
        # Create quality detector that raises an error
        detector = QualityDetector(hitl_config)

        def mock_evaluate_error(query, search_results, complexity_score):
            raise ValueError("Simulated quality detection error")

        detector.evaluate = mock_evaluate_error

        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # Initial state
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should handle error and continue
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-5"})

        # Verify workflow continued despite error
        assert result is not None
        assert result.get("quality_check_required") == False
        assert len(result.get("errors", [])) > 0  # Error logged
        assert any("HITL gate error" in e for e in result.get("errors", []))


class TestHITLResume:
    """Test HITL workflow resume after user clarification."""

    async def test_workflow_enriches_query_on_resume(
        self,
        hitl_config,
        mock_agent_registry,
        mock_quality_detector,
        mock_clarification_generator,
    ):
        """Test that workflow enriches query when resuming after user clarification."""
        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=mock_quality_detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # State after user provided clarification
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            original_query="What is GDPR?",
            user_clarification="I need information about GDPR compliance requirements for data processors.",
            clarification_round=1,
            awaiting_user_input=True,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should enrich query and continue
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-6"})

        # Verify query was enriched
        assert result is not None
        assert result.get("enriched_query") is not None
        assert "compliance requirements" in result.get("enriched_query", "")
        assert result.get("awaiting_user_input") == False
        assert result.get("quality_check_required") == False

    async def test_workflow_clears_clarification_state_on_resume(
        self,
        hitl_config,
        mock_agent_registry,
        mock_quality_detector,
        mock_clarification_generator,
    ):
        """Test that clarification state is cleared when resuming."""
        # Build workflow
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=hitl_config,
            quality_detector=mock_quality_detector,
            clarification_generator=mock_clarification_generator,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # State with clarification data
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            original_query="What is GDPR?",
            user_clarification="compliance requirements",
            awaiting_user_input=True,
            quality_check_required=True,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-7"})

        # Verify clarification state was cleared
        assert result.get("awaiting_user_input") == False
        assert result.get("quality_check_required") == False


class TestHITLDisabled:
    """Test workflow behavior when HITL is disabled."""

    async def test_workflow_without_hitl_components(self, mock_agent_registry):
        """Test that workflow works without HITL components (disabled)."""
        # Build workflow WITHOUT HITL components
        builder = WorkflowBuilder(
            agent_registry=mock_agent_registry,
            checkpointer=None,
            hitl_config=None,  # HITL disabled
            quality_detector=None,
            clarification_generator=None,
        )

        workflow = builder.build_workflow(
            agent_sequence=["extractor", "classifier"], enable_parallel=False
        )

        # Initial state
        state = MultiAgentState(
            query="What is GDPR?",
            execution_phase=ExecutionPhase.AGENT_EXECUTION,
            complexity_score=60,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        # Execute workflow - should work normally without HITL
        result = await workflow.ainvoke(state.model_dump(), {"thread_id": "test-8"})

        # Verify workflow completed without HITL
        assert result is not None
        assert "classifier" in result.get("agent_outputs", {})
        assert result.get("clarifying_questions") == []
