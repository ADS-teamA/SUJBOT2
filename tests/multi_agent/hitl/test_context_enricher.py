"""
Unit tests for ContextEnricher.

Tests query enrichment strategies.
"""

import pytest
from src.multi_agent.hitl.config import HITLConfig
from src.multi_agent.hitl.context_enricher import ContextEnricher


@pytest.fixture
def default_config():
    """Default HITL configuration."""
    return HITLConfig()


@pytest.fixture
def enricher(default_config):
    """ContextEnricher instance."""
    return ContextEnricher(default_config)


class TestAppendWithContext:
    """Test append_with_context strategy."""

    def test_basic_enrichment(self, enricher):
        """Basic append with context."""
        state = {}
        original_query = "What are the rules?"
        user_response = "I'm interested in GDPR regulations for 2024"

        enriched_state = enricher.enrich(original_query, user_response, state)

        assert enriched_state["original_query"] == original_query
        assert enriched_state["user_clarification"] == user_response
        assert original_query in enriched_state["enriched_query"]
        assert user_response in enriched_state["enriched_query"]
        assert enriched_state["query"] == enriched_state["enriched_query"]

    def test_template_formatting(self, enricher):
        """Template should be applied correctly."""
        state = {}
        original = "What are compliance requirements?"
        response = "For healthcare in the US"

        enriched_state = enricher.enrich(original, response, state)
        enriched = enriched_state["enriched_query"]

        # Default template: "{original_query}\n\n[Context]: {user_response}"
        assert "[Context]:" in enriched
        assert original in enriched
        assert response in enriched

    def test_shared_context_updated(self, enricher):
        """Shared context should be updated."""
        state = {}
        enriched_state = enricher.enrich("Original", "Response", state)

        assert "shared_context" in enriched_state
        assert "hitl_clarification" in enriched_state["shared_context"]

        hitl_ctx = enriched_state["shared_context"]["hitl_clarification"]
        assert hitl_ctx["original_query"] == "Original"
        assert hitl_ctx["user_response"] == "Response"
        assert "enriched_query" in hitl_ctx


class TestReplaceStrategy:
    """Test replace strategy."""

    def test_replace_query(self, enricher):
        """Replace strategy should use only user response."""
        enricher.config.enrichment_strategy = "replace"

        state = {}
        original = "Vague query"
        response = "GDPR Article 5 requirements for data processing"

        enriched_state = enricher.enrich(original, response, state)

        # Enriched query should be just the user response
        assert enriched_state["enriched_query"] == response
        assert enriched_state["query"] == response
        assert enriched_state["original_query"] == original


class TestMaxLengthEnforcement:
    """Test max_enriched_length enforcement."""

    def test_truncation_when_too_long(self, enricher):
        """Long enriched queries should be truncated."""
        enricher.config.max_enriched_length = 100

        state = {}
        original = "Short query"
        response = "A" * 200  # Very long response

        enriched_state = enricher.enrich(original, response, state)

        assert len(enriched_state["enriched_query"]) <= 100

    def test_no_truncation_when_within_limit(self, enricher):
        """Short queries should not be truncated."""
        enricher.config.max_enriched_length = 500

        state = {}
        original = "What are GDPR requirements?"
        response = "For healthcare data processing in 2024"

        enriched_state = enricher.enrich(original, response, state)
        enriched = enriched_state["enriched_query"]

        # Should contain both original and response (no truncation)
        assert original in enriched
        assert response in enriched


class TestEmptyResponse:
    """Test handling of empty user responses."""

    def test_empty_response_returns_unchanged(self, enricher):
        """Empty response should not modify query."""
        state = {"query": "Original query"}
        enriched_state = enricher.enrich("Original query", "", state)

        # State should be unchanged (or original query preserved)
        assert "user_clarification" not in enriched_state or enriched_state["user_clarification"] == ""

    def test_whitespace_only_response(self, enricher):
        """Whitespace-only response should be treated as empty."""
        state = {}
        enriched_state = enricher.enrich("Original", "   \n\t  ", state)

        # Should not enrich with empty response
        assert "user_clarification" not in enriched_state or not enriched_state.get("user_clarification", "").strip()


class TestClearClarification:
    """Test clearing clarification fields."""

    def test_clear_all_clarification_fields(self, enricher):
        """All HITL fields should be cleared."""
        state = {
            "quality_check_required": True,
            "quality_issues": ["issue1", "issue2"],
            "clarifying_questions": [{"id": "q1", "text": "Question?"}],
            "user_clarification": "Response",
            "enriched_query": "Enriched",
            "other_field": "preserved"
        }

        cleared_state = enricher.clear_clarification(state)

        # HITL fields should be removed
        assert "quality_check_required" not in cleared_state
        assert "quality_issues" not in cleared_state
        assert "clarifying_questions" not in cleared_state
        assert "user_clarification" not in cleared_state
        assert "enriched_query" not in cleared_state

        # Other fields preserved
        assert cleared_state["other_field"] == "preserved"

    def test_clear_idempotent(self, enricher):
        """Clearing already-clean state should not error."""
        state = {"query": "Test"}
        cleared_state = enricher.clear_clarification(state)

        # Should not crash
        assert "query" in cleared_state


class TestStatePreservation:
    """Test that enrichment preserves other state fields."""

    def test_existing_fields_preserved(self, enricher):
        """Existing state fields should not be overwritten."""
        state = {
            "query": "Original",
            "documents": ["doc1", "doc2"],
            "complexity_score": 75,
            "agent_sequence": ["extractor"]
        }

        enriched_state = enricher.enrich("Original", "Clarification", state)

        # Original fields should be preserved
        assert enriched_state["documents"] == ["doc1", "doc2"]
        assert enriched_state["complexity_score"] == 75
        assert enriched_state["agent_sequence"] == ["extractor"]

        # New fields added
        assert "original_query" in enriched_state
        assert "enriched_query" in enriched_state


class TestCustomTemplate:
    """Test custom enrichment templates."""

    def test_custom_template(self, enricher):
        """Custom template should be respected."""
        enricher.config.enrichment_template = "QUERY: {original_query} | USER: {user_response}"

        state = {}
        enriched_state = enricher.enrich("Test query", "Test response", state)

        enriched = enriched_state["enriched_query"]
        assert "QUERY:" in enriched
        assert "USER:" in enriched
        assert "|" in enriched


class TestEdgeCases:
    """Test edge cases."""

    def test_unicode_characters(self, enricher):
        """Handle unicode characters correctly."""
        state = {}
        original = "Co jsou pravidla GDPR?"  # Czech with diacritics
        response = "Pro zdravotnictví v ČR"

        enriched_state = enricher.enrich(original, response, state)

        assert original in enriched_state["enriched_query"]
        assert response in enriched_state["enriched_query"]

    def test_special_characters_in_template(self, enricher):
        """Handle special characters in template."""
        state = {}
        original = "Query with $pecial ch@racters"
        response = "Response with {braces} and [brackets]"

        enriched_state = enricher.enrich(original, response, state)

        # Should not crash
        assert "enriched_query" in enriched_state

    def test_very_long_original_query(self, enricher):
        """Handle very long original queries."""
        enricher.config.max_enriched_length = 200

        state = {}
        original = "What are the rules? " * 50  # Very long
        response = "GDPR 2024"

        enriched_state = enricher.enrich(original, response, state)

        # Should be truncated
        assert len(enriched_state["enriched_query"]) <= 200

    def test_none_state_values(self, enricher):
        """Handle None values in state."""
        state = {
            "query": None,
            "documents": None
        }

        enriched_state = enricher.enrich("Query", "Response", state)

        # Should not crash
        assert "enriched_query" in enriched_state
