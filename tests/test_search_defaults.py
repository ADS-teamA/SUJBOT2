"""
Tests for search tool default parameter changes

This PR changes several search defaults:
- Default k: 3 → 10
- Max k: 20 → 200
- Default enable_graph_boost: False → True
- RRF sort order: FIXED from ascending to descending

These tests ensure the new defaults work correctly and prevent future regressions.
"""

import pytest
from pydantic import ValidationError
from src.agent.tools.search import SearchInput
from src.agent.tools._utils import validate_k_parameter


class TestSearchDefaultChanges:
    """Regression tests for search default parameter changes."""

    def test_default_k_is_10(self):
        """Default k should be 10 (changed from 3)."""
        input_schema = SearchInput(query="test query")
        assert input_schema.k == 10

    def test_max_k_is_200(self):
        """Maximum k should be 200 (changed from 20)."""
        # Should not raise validation error
        input_schema = SearchInput(query="test", k=200)
        assert input_schema.k == 200

    def test_k_above_200_raises_error(self):
        """k > 200 should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchInput(query="test", k=201)
        assert "less than or equal to 200" in str(exc_info.value)

    def test_k_minimum_is_1(self):
        """k must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            SearchInput(query="test", k=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_default_graph_boost_enabled(self):
        """Graph boost should be enabled by default (changed from False)."""
        input_schema = SearchInput(query="test")
        assert input_schema.enable_graph_boost is True

    def test_graph_boost_can_be_disabled(self):
        """Graph boost should be explicitly disableable."""
        input_schema = SearchInput(query="test", enable_graph_boost=False)
        assert input_schema.enable_graph_boost is False

    def test_default_num_expands_is_zero(self):
        """Default num_expands should be 0 (no change)."""
        input_schema = SearchInput(query="test")
        assert input_schema.num_expands == 0

    def test_default_use_hyde_is_false(self):
        """Default use_hyde should be False (no change)."""
        input_schema = SearchInput(query="test")
        assert input_schema.use_hyde is False

    def test_default_search_method_is_hybrid(self):
        """Default search_method should be 'hybrid' (no change)."""
        input_schema = SearchInput(query="test")
        assert input_schema.search_method == "hybrid"


class TestValidateKParameter:
    """Tests for validate_k_parameter utility function."""

    def test_validate_k_parameter_max_200(self):
        """validate_k_parameter should allow k up to 200."""
        validated_k, reason = validate_k_parameter(k=200, adaptive=False)
        assert validated_k == 200
        assert reason is None

    def test_validate_k_parameter_clamps_above_200(self):
        """validate_k_parameter should clamp k > 200."""
        validated_k, reason = validate_k_parameter(k=300, adaptive=False)
        assert validated_k == 200
        assert reason == "exceeded_maximum"

    def test_validate_k_parameter_clamps_below_1(self):
        """validate_k_parameter should clamp k < 1."""
        validated_k, reason = validate_k_parameter(k=0, adaptive=False)
        assert validated_k == 1
        assert reason == "below_minimum"

    def test_validate_k_parameter_negative(self):
        """validate_k_parameter should handle negative k."""
        validated_k, reason = validate_k_parameter(k=-5, adaptive=False)
        assert validated_k == 1
        assert reason == "below_minimum"

    def test_validate_k_parameter_valid_range(self):
        """validate_k_parameter should not adjust k in valid range."""
        for k in [1, 10, 50, 100, 150, 200]:
            validated_k, reason = validate_k_parameter(k=k, adaptive=False)
            assert validated_k == k
            assert reason is None

    def test_validate_k_adaptive_mode_exists(self):
        """Adaptive mode should be available."""
        # Adaptive mode may reduce k based on token budget
        validated_k, reason = validate_k_parameter(
            k=200, adaptive=True, detail_level="full"
        )
        # Should still return valid k
        assert 1 <= validated_k <= 200
        assert isinstance(reason, (str, type(None)))

    def test_validate_k_adaptive_respects_token_budget(self):
        """Adaptive k should not exceed token budget."""
        # With detail_level="full", k=200 may exceed budget
        validated_k, reason = validate_k_parameter(
            k=200, adaptive=True, detail_level="full"
        )
        # Should clamp to budget limit or keep at 200
        assert validated_k <= 200
        if validated_k < 200:
            # If reduced, should have a reason
            assert reason is not None


class TestSearchMethodValidation:
    """Tests for search_method parameter validation."""

    def test_hybrid_method_allowed(self):
        """'hybrid' search method should be allowed."""
        input_schema = SearchInput(query="test", search_method="hybrid")
        assert input_schema.search_method == "hybrid"

    def test_dense_only_method_allowed(self):
        """'dense_only' search method should be allowed."""
        input_schema = SearchInput(query="test", search_method="dense_only")
        assert input_schema.search_method == "dense_only"

    def test_bm25_only_method_allowed(self):
        """'bm25_only' search method should be allowed."""
        input_schema = SearchInput(query="test", search_method="bm25_only")
        assert input_schema.search_method == "bm25_only"

    # Note: search_method validation is done at runtime, not at model level
    # Invalid search methods will fail during execution, not during initialization


class TestNumExpandsValidation:
    """Tests for num_expands parameter validation."""

    def test_num_expands_zero_allowed(self):
        """num_expands=0 should be allowed."""
        input_schema = SearchInput(query="test", num_expands=0)
        assert input_schema.num_expands == 0

    def test_num_expands_up_to_5_allowed(self):
        """num_expands up to 5 should be allowed."""
        for n in range(6):
            input_schema = SearchInput(query="test", num_expands=n)
            assert input_schema.num_expands == n

    def test_num_expands_above_5_rejected(self):
        """num_expands > 5 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchInput(query="test", num_expands=6)
        assert "less than or equal to 5" in str(exc_info.value)

    def test_num_expands_negative_rejected(self):
        """Negative num_expands should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchInput(query="test", num_expands=-1)
        assert "greater than or equal to 0" in str(exc_info.value)


class TestParameterCombinations:
    """Tests for various parameter combinations."""

    def test_high_k_with_hyde(self):
        """High k with HyDE should work."""
        input_schema = SearchInput(
            query="test",
            k=200,
            use_hyde=True
        )
        assert input_schema.k == 200
        assert input_schema.use_hyde is True

    def test_high_k_with_expansion(self):
        """High k with query expansion should work."""
        input_schema = SearchInput(
            query="test",
            k=200,
            num_expands=2
        )
        assert input_schema.k == 200
        assert input_schema.num_expands == 2

    def test_all_features_enabled(self):
        """All features enabled simultaneously should work."""
        input_schema = SearchInput(
            query="test",
            k=100,
            use_hyde=True,
            num_expands=2,
            enable_graph_boost=True,
            search_method="hybrid"
        )
        assert input_schema.k == 100
        assert input_schema.use_hyde is True
        assert input_schema.num_expands == 2
        assert input_schema.enable_graph_boost is True
        assert input_schema.search_method == "hybrid"

    def test_minimal_configuration(self):
        """Minimal configuration (only query) should use defaults."""
        input_schema = SearchInput(query="test")
        assert input_schema.k == 10
        assert input_schema.use_hyde is False
        assert input_schema.num_expands == 0
        assert input_schema.enable_graph_boost is True
        assert input_schema.search_method == "hybrid"


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility where applicable."""

    def test_small_k_values_still_work(self):
        """Small k values (old default k=3) should still work."""
        input_schema = SearchInput(query="test", k=3)
        assert input_schema.k == 3

    def test_old_max_k_still_works(self):
        """Old max k=20 should still work."""
        input_schema = SearchInput(query="test", k=20)
        assert input_schema.k == 20

    def test_explicit_graph_boost_false(self):
        """Explicitly disabling graph_boost should work."""
        input_schema = SearchInput(query="test", enable_graph_boost=False)
        assert input_schema.enable_graph_boost is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_boundary_values(self):
        """Test k at exact boundary values."""
        # Minimum
        input_schema = SearchInput(query="test", k=1)
        assert input_schema.k == 1

        # Maximum
        input_schema = SearchInput(query="test", k=200)
        assert input_schema.k == 200

    def test_empty_query_string(self):
        """Empty query string should be allowed (validation elsewhere)."""
        input_schema = SearchInput(query="")
        assert input_schema.query == ""

    def test_very_long_query(self):
        """Very long query should be allowed."""
        long_query = "test " * 1000
        input_schema = SearchInput(query=long_query)
        assert input_schema.query == long_query

    def test_unicode_in_query(self):
        """Unicode characters in query should work."""
        query = "Jaké jsou požadavky GDPR?"
        input_schema = SearchInput(query=query)
        assert input_schema.query == query
