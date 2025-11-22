"""
Test state reducer functions for parallel execution.

Verifies that LangGraph state updates work correctly when multiple
agents update the same fields in parallel (fan-out/fan-in pattern).
"""

import pytest
from pydantic import ValidationError
from src.multi_agent.core.state import (
    MultiAgentState,
    QueryType,
    ExecutionPhase,
    keep_first,
    merge_dicts,
    merge_lists_unique,
)


class TestReducerFunctions:
    """Test individual reducer functions."""

    def test_keep_first_with_none(self):
        """Test keep_first returns new value when existing is None."""
        result = keep_first(None, "new_value")
        assert result == "new_value"

    def test_keep_first_with_existing(self):
        """Test keep_first keeps existing value when not None."""
        result = keep_first("existing", "new_value")
        assert result == "existing"

    def test_merge_dicts_both_values(self):
        """Test merge_dicts merges two dictionaries."""
        existing = {"a": 1, "b": 2}
        new = {"b": 3, "c": 4}
        result = merge_dicts(existing, new)
        assert result == {"a": 1, "b": 3, "c": 4}  # New overrides existing

    def test_merge_dicts_with_none(self):
        """Test merge_dicts handles None values."""
        assert merge_dicts(None, {"a": 1}) == {"a": 1}
        assert merge_dicts({"a": 1}, None) == {"a": 1}

    def test_merge_lists_unique_no_duplicates(self):
        """Test merge_lists_unique removes duplicates."""
        existing = ["a", "b", "c"]
        new = ["c", "d", "e"]
        result = merge_lists_unique(existing, new)
        assert result == ["a", "b", "c", "d", "e"]  # No duplicate 'c'

    def test_merge_lists_unique_preserves_order(self):
        """Test merge_lists_unique preserves order of first occurrence."""
        existing = ["agent1", "agent2"]
        new = ["agent2", "agent3"]
        result = merge_lists_unique(existing, new)
        assert result == ["agent1", "agent2", "agent3"]

    def test_keep_first_with_enum_default(self):
        """Test keep_first replaces default Enum values."""
        # Default UNKNOWN should be replaced
        result = keep_first(QueryType.UNKNOWN, QueryType.COMPLIANCE_CHECK)
        assert result == QueryType.COMPLIANCE_CHECK

        # Default ROUTING should be replaced
        result = keep_first(ExecutionPhase.ROUTING, ExecutionPhase.EXTRACTION)
        assert result == ExecutionPhase.EXTRACTION

    def test_keep_first_with_enum_already_set(self):
        """Test keep_first preserves non-default Enum values."""
        # Non-default value should be preserved
        result = keep_first(QueryType.COMPLIANCE_CHECK, QueryType.RISK_ASSESSMENT)
        assert result == QueryType.COMPLIANCE_CHECK

        result = keep_first(ExecutionPhase.EXTRACTION, ExecutionPhase.SYNTHESIS)
        assert result == ExecutionPhase.EXTRACTION


class TestMultiAgentStateCreation:
    """Test MultiAgentState creation and validation."""

    def test_create_minimal_state(self):
        """Test creating state with only required fields."""
        state = MultiAgentState(query="Test query")
        assert state.query == "Test query"
        assert state.complexity_score == 0
        assert state.agent_sequence == []
        assert state.agent_outputs == {}
        assert state.errors == []

    def test_create_full_state(self):
        """Test creating state with all fields populated."""
        state = MultiAgentState(
            query="Test query",
            complexity_score=75,
            agent_sequence=["extractor", "classifier"],
            agent_outputs={
                "extractor": {"docs": ["doc1", "doc2"]},
                "classifier": {"category": "compliance"}
            },
            errors=[]
        )
        assert state.complexity_score == 75
        assert len(state.agent_sequence) == 2
        assert "extractor" in state.agent_outputs

    def test_complexity_score_validation(self):
        """Test complexity_score must be 0-100."""
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            MultiAgentState(query="Test", complexity_score=150)

    def test_confidence_score_validation(self):
        """Test confidence_score must be 0.0-1.0."""
        with pytest.raises(ValidationError, match="Confidence score must be 0.0-1.0"):
            MultiAgentState(query="Test", confidence_score=1.5)


class TestStateHelperMethods:
    """Test MultiAgentState helper methods."""

    def test_add_error(self):
        """Test add_error appends error with timestamp."""
        state = MultiAgentState(query="Test")
        state.add_error("Test error")
        assert len(state.errors) == 1
        assert "Test error" in state.errors[0]

    def test_add_agent_output(self):
        """Test add_agent_output records output and updates sequence."""
        state = MultiAgentState(query="Test")
        state.add_agent_output("extractor", {"docs": ["doc1"]})

        assert "extractor" in state.agent_outputs
        assert "extractor" in state.agent_sequence
        assert state.agent_outputs["extractor"]["docs"] == ["doc1"]

    def test_update_execution_phase(self):
        """Test update_execution_phase changes phase."""
        from src.multi_agent.core.state import ExecutionPhase

        state = MultiAgentState(query="Test")
        state.update_execution_phase(ExecutionPhase.EXTRACTION)
        assert state.execution_phase == ExecutionPhase.EXTRACTION

    def test_is_error_state(self):
        """Test is_error_state detects errors."""
        state = MultiAgentState(query="Test")
        assert not state.is_error_state()

        state.add_error("Test error")
        assert state.is_error_state()


class TestParallelExecutionScenario:
    """
    Test scenarios simulating parallel agent execution.

    These tests verify that state updates from multiple agents
    merge correctly using the reducer functions.
    """

    def test_parallel_query_update(self):
        """Test query field keeps first value (immutable)."""
        # Simulate two agents trying to update query
        state1 = MultiAgentState(query="Original query")
        state2 = MultiAgentState(query="Original query")

        # Agent 1 returns state with query unchanged
        agent1_result = {"query": "Original query"}

        # Agent 2 tries to modify query (should be ignored by keep_first)
        agent2_result = {"query": "Modified query"}

        # LangGraph would merge using keep_first reducer
        merged_query = keep_first(agent1_result["query"], agent2_result["query"])
        assert merged_query == "Original query"

    def test_parallel_agent_sequence_update(self):
        """Test agent_sequence merges uniquely from multiple agents."""
        # Agent 1 adds itself to sequence
        agent1_sequence = ["extractor"]

        # Agent 2 adds itself to sequence
        agent2_sequence = ["classifier"]

        # Merge (both agents ran in parallel)
        merged = merge_lists_unique(agent1_sequence, agent2_sequence)
        assert merged == ["extractor", "classifier"]

    def test_parallel_agent_outputs_merge(self):
        """Test agent_outputs from multiple agents merge correctly."""
        # Agent 1 output
        agent1_output = {"extractor": {"docs": ["doc1", "doc2"]}}

        # Agent 2 output
        agent2_output = {"classifier": {"category": "compliance"}}

        # Merge
        merged = merge_dicts(agent1_output, agent2_output)
        assert merged == {
            "extractor": {"docs": ["doc1", "doc2"]},
            "classifier": {"category": "compliance"}
        }

    def test_parallel_errors_concatenate(self):
        """Test errors from multiple agents concatenate."""
        import operator

        # Agent 1 errors
        agent1_errors = ["Error from extractor"]

        # Agent 2 errors
        agent2_errors = ["Error from classifier"]

        # Merge using operator.add
        merged = operator.add(agent1_errors, agent2_errors)
        assert merged == ["Error from extractor", "Error from classifier"]

    def test_parallel_cost_accumulation(self):
        """Test costs from multiple agents sum correctly."""
        import operator

        # Agent 1 cost
        agent1_cost = 0.05

        # Agent 2 cost
        agent2_cost = 0.03

        # Merge using operator.add
        merged = operator.add(agent1_cost, agent2_cost)
        assert merged == 0.08
