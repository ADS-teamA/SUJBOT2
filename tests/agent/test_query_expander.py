"""
Tests for Query Expansion Module

Tests the QueryExpander class for:
- Multi-query generation
- Graceful fallback on errors
- Cost tracking
- Optimization (skip expansion when num_expands=1)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agent.query_expander import QueryExpander, ExpansionResult


class TestQueryExpanderInitialization:
    """Test QueryExpander initialization."""

    def test_init_openai_success(self):
        """Test successful initialization with OpenAI provider."""
        with patch("openai.OpenAI") as mock_openai:
            expander = QueryExpander(
                provider="openai",
                model="gpt-4o-mini",
                openai_api_key="sk-test-key"
            )

            assert expander.provider == "openai"
            assert expander.model == "gpt-4o-mini"
            mock_openai.assert_called_once_with(api_key="sk-test-key")

    def test_init_anthropic_success(self):
        """Test successful initialization with Anthropic provider."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            expander = QueryExpander(
                provider="anthropic",
                model="claude-haiku-4-5",
                anthropic_api_key="sk-ant-test-key"
            )

            assert expander.provider == "anthropic"
            assert expander.model == "claude-haiku-4-5"
            mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_init_missing_openai_key(self):
        """Test initialization fails without OpenAI API key."""
        with pytest.raises(ValueError, match="openai_api_key required"):
            QueryExpander(provider="openai", model="gpt-4o-mini")

    def test_init_missing_anthropic_key(self):
        """Test initialization fails without Anthropic API key."""
        with pytest.raises(ValueError, match="anthropic_api_key required"):
            QueryExpander(provider="anthropic", model="claude-haiku-4-5")

    def test_init_invalid_provider(self):
        """Test initialization fails with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            QueryExpander(
                provider="invalid",
                model="test",
                openai_api_key="sk-test"
            )

    def test_init_missing_openai_package(self):
        """Test initialization fails gracefully when openai package is missing."""
        with patch("openai.OpenAI", side_effect=ImportError):
            with pytest.raises(ImportError, match="openai package required"):
                QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")


class TestQueryExpansionOptimization:
    """Test query expansion optimization (skip LLM when num_expansions=0)."""

    def test_skip_expansion_when_num_expands_0(self):
        """Test that expansion is skipped when num_expansions=0 (optimization)."""
        with patch("openai.OpenAI"):
            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")

            # Mock the LLM call to ensure it's NOT called
            expander._generate_expansions_llm = Mock()

            result = expander.expand("test query", num_expansions=0)

            # Verify no LLM call
            expander._generate_expansions_llm.assert_not_called()

            # Verify result
            assert result.original_query == "test query"
            assert result.expanded_queries == ["test query"]
            assert result.num_expansions == 1  # 1 query total (original only)
            assert result.expansion_method == "none"
            assert result.model_used is None

    def test_expansion_when_num_expands_1(self):
        """Test that expansion DOES happen when num_expansions=1 (generates 1 paraphrase)."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")
            expander.client = mock_client

            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Query variation 1"))]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            mock_client.chat.completions.create.return_value = mock_response

            result = expander.expand("test query", num_expansions=1)

            # Verify LLM WAS called
            mock_client.chat.completions.create.assert_called_once()

            # Verify result: original + 1 paraphrase = 2 queries total
            assert result.original_query == "test query"
            assert len(result.expanded_queries) == 2
            assert "test query" in result.expanded_queries
            assert "Query variation 1" in result.expanded_queries
            assert result.num_expansions == 2  # 2 queries total
            assert result.expansion_method == "llm"
            assert result.model_used == "gpt-4o-mini"

    def test_expansion_filters_duplicate_original(self):
        """Test that expansion filters out original query if LLM repeats it."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")
            expander.client = mock_client

            # Mock OpenAI response with duplicate original query
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="test query\nQuery variation 1"))]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            mock_client.chat.completions.create.return_value = mock_response

            result = expander.expand("test query", num_expansions=2)

            # Should filter out duplicate "test query" from LLM response
            # Expected: original + "Query variation 1" only
            assert result.original_query == "test query"
            assert len(result.expanded_queries) == 2
            assert result.expanded_queries.count("test query") == 1  # Only once (original)


class TestQueryExpansionBasic:
    """Test basic query expansion functionality."""

    @pytest.fixture
    def mock_openai_expander(self):
        """Create QueryExpander with mocked OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(
                provider="openai",
                model="gpt-4o-mini",
                openai_api_key="sk-test"
            )
            expander.client = mock_client

            yield expander, mock_client

    def test_expand_generates_multiple_queries(self, mock_openai_expander):
        """Test expansion generates multiple query variations."""
        expander, mock_client = mock_openai_expander

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query variation 1\nQuery variation 2\nQuery variation 3"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        result = expander.expand("test query", num_expansions=3)

        assert result.original_query == "test query"
        assert len(result.expanded_queries) == 4  # Original + 3 expansions
        assert "test query" in result.expanded_queries
        assert result.expansion_method == "llm"
        assert result.model_used == "gpt-4o-mini"

    def test_expand_strips_numbering(self, mock_openai_expander):
        """Test that LLM numbering is stripped from expansions."""
        expander, mock_client = mock_openai_expander

        # Mock OpenAI response with numbering
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="1. First query\n2. Second query\n3) Third query"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        result = expander.expand("test", num_expansions=3)

        # Check that numbering is stripped
        expansions = [q for q in result.expanded_queries if q != "test"]
        assert "First query" in expansions
        assert "Second query" in expansions
        assert "Third query" in expansions
        assert "1. First query" not in expansions


class TestQueryExpansionWarnings:
    """Test warning behavior for high expansion counts."""

    def test_warning_logged_when_num_expands_exceeds_threshold(self, caplog):
        """Test warning is logged when num_expands > warn_threshold."""
        with patch("openai.OpenAI"):
            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")

            # Mock LLM to return immediately
            expander._generate_expansions_llm = Mock(return_value=["query1", "query2"])

            with caplog.at_level("WARNING"):
                expander.expand("test", num_expansions=6, warn_threshold=5)

            # Check warning was logged
            assert "High expansion count" in caplog.text
            assert "may impact latency" in caplog.text


class TestQueryExpansionErrorHandling:
    """Test error handling and graceful fallback."""

    @pytest.fixture
    def mock_openai_expander(self):
        """Create QueryExpander with mocked OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")
            expander.client = mock_client

            yield expander, mock_client

    def test_fallback_on_llm_error(self, mock_openai_expander, caplog):
        """Test graceful fallback when LLM expansion fails."""
        expander, mock_client = mock_openai_expander

        # Mock LLM to raise error
        mock_client.chat.completions.create.side_effect = Exception("API error")

        with caplog.at_level("WARNING"):
            result = expander.expand("test query", num_expansions=3)

        # Should fall back to original query
        assert result.original_query == "test query"
        assert result.expanded_queries == ["test query"]
        assert result.num_expansions == 1
        assert result.expansion_method == "fallback"
        assert "Query expansion failed" in caplog.text


class TestPromptConstruction:
    """Test prompt construction for LLM."""

    def test_prompt_includes_num_expansions(self):
        """Test that prompt includes correct num_expansions count."""
        with patch("openai.OpenAI"):
            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")

            prompt = expander._build_expansion_prompt("test query", num_expansions=5)

            assert "5 related questions" in prompt or "5 variations" in prompt
            assert "test query" in prompt

    def test_prompt_uses_multi_question_strategy(self):
        """Test that prompt uses multi-question generation strategy."""
        with patch("openai.OpenAI"):
            expander = QueryExpander(provider="openai", model="gpt-4o-mini", openai_api_key="sk-test")

            prompt = expander._build_expansion_prompt("test", num_expansions=3)

            # Check for multi-question strategy keywords
            assert "synonyms" in prompt.lower() or "related" in prompt.lower()
            assert "rephrase" in prompt.lower() or "different" in prompt.lower()


class TestCostTracking:
    """Test cost tracking integration."""

    @pytest.fixture
    def mock_openai_expander(self):
        """Create QueryExpander with mocked OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(
                provider="openai",
                model="gpt-4o-mini",
                openai_api_key="sk-test"
            )
            expander.client = mock_client

            yield expander, mock_client

    def test_cost_tracking_called(self, mock_openai_expander):
        """Test that cost tracking is called with correct parameters."""
        expander, mock_client = mock_openai_expander

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1\nQuery 2"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        # Mock cost tracker
        with patch("src.cost_tracker.get_global_tracker") as mock_tracker_getter:
            mock_tracker = Mock()
            mock_tracker_getter.return_value = mock_tracker

            expander.expand("test", num_expansions=2)

            # Verify cost tracking was called
            mock_tracker.track_llm.assert_called_once_with(
                "openai", "gpt-4o-mini", 100, 50
            )

    def test_cost_tracking_continues_on_error(self, mock_openai_expander, caplog):
        """Test that expansion continues even if cost tracking fails."""
        expander, mock_client = mock_openai_expander

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        # Mock cost tracker to raise error
        with patch("src.cost_tracker.get_global_tracker", side_effect=Exception("Tracker error")):
            # Should not raise, just log
            result = expander.expand("test", num_expansions=2)

            # Expansion should still succeed
            assert result.expansion_method == "llm"


class TestExpansionResult:
    """Test ExpansionResult dataclass."""

    def test_expansion_result_structure(self):
        """Test ExpansionResult has correct structure."""
        result = ExpansionResult(
            original_query="test",
            expanded_queries=["test", "query1", "query2"],
            num_expansions=3,
            expansion_method="llm",
            model_used="gpt-4o-mini",
            cost_estimate=0.001
        )

        assert result.original_query == "test"
        assert len(result.expanded_queries) == 3
        assert result.num_expansions == 3
        assert result.expansion_method == "llm"
        assert result.model_used == "gpt-4o-mini"
        assert result.cost_estimate == 0.001


class TestOSeriesCompatibility:
    """Test O-series (o1, o3, o4) specific compatibility issues."""

    @pytest.fixture
    def mock_openai_o_series_expander(self):
        """Create QueryExpander with mocked OpenAI client for O-series model."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(
                provider="openai",
                model="o1-mini",  # O-series model
                openai_api_key="sk-test"
            )
            expander.client = mock_client

            yield expander, mock_client

    def test_empty_response_handling(self, mock_openai_o_series_expander, caplog):
        """Test handling when LLM returns empty/None content."""
        expander, mock_client = mock_openai_o_series_expander

        # Mock LLM returning empty content
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]  # Empty string
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=0)
        mock_response.model_dump.return_value = {"id": "test", "choices": []}
        mock_client.chat.completions.create.return_value = mock_response

        with caplog.at_level("WARNING"):
            result = expander.expand("test query", num_expansions=2)

        # Should generate warning about empty response
        assert "Generated 0/2 expansions" in caplog.text

        # Should still return original query (graceful degradation)
        assert result.original_query == "test query"
        assert len(result.expanded_queries) == 1  # Only original
        assert result.expanded_queries[0] == "test query"

    def test_o_series_uses_max_completion_tokens(self, mock_openai_o_series_expander):
        """Test that O-series models use max_completion_tokens instead of max_tokens."""
        expander, mock_client = mock_openai_o_series_expander

        # Mock valid response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1\nQuery 2"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        expander.expand("test", num_expansions=2)

        # Verify API call used max_completion_tokens (not max_tokens)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 300

    def test_o_series_no_temperature_parameter(self, mock_openai_o_series_expander):
        """Test that O-series models don't set temperature (only default 1.0 supported)."""
        expander, mock_client = mock_openai_o_series_expander

        # Mock valid response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        expander.expand("test", num_expansions=1)

        # Verify API call doesn't include temperature
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs

    def test_gpt4_uses_max_tokens_and_temperature(self):
        """Test that GPT-4 models use max_tokens and temperature (old API)."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(
                provider="openai",
                model="gpt-4o-mini",  # GPT-4 model
                openai_api_key="sk-test"
            )
            expander.client = mock_client

            # Mock valid response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Query 1"))]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            mock_client.chat.completions.create.return_value = mock_response

            expander.expand("test", num_expansions=1)

            # Verify API call uses old parameters
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "max_tokens" in call_kwargs
            assert "temperature" in call_kwargs
            assert "max_completion_tokens" not in call_kwargs
            assert call_kwargs["max_tokens"] == 300
            assert call_kwargs["temperature"] == 0.7


class TestAnthropicProvider:
    """Test Anthropic provider integration."""

    @pytest.fixture
    def mock_anthropic_expander(self):
        """Create QueryExpander with mocked Anthropic client."""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            expander = QueryExpander(
                provider="anthropic",
                model="claude-haiku-4-5",
                anthropic_api_key="sk-ant-test"
            )
            expander.client = mock_client

            yield expander, mock_client

    def test_anthropic_expansion_success(self, mock_anthropic_expander):
        """Test successful query expansion with Anthropic."""
        expander, mock_client = mock_anthropic_expander

        # Mock Anthropic response
        mock_content = Mock()
        mock_content.text = "Alternative query 1\nAlternative query 2\nAlternative query 3"
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        result = expander.expand("test query", num_expansions=3)

        # Verify result
        assert result.original_query == "test query"
        assert len(result.expanded_queries) == 4  # Original + 3 expansions
        assert "test query" in result.expanded_queries
        assert result.expansion_method == "llm"
        assert result.model_used == "claude-haiku-4-5"

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert call_kwargs["max_tokens"] == 300
        assert call_kwargs["temperature"] == 0.7

    def test_anthropic_cost_tracking(self, mock_anthropic_expander):
        """Test that cost tracking works with Anthropic provider."""
        expander, mock_client = mock_anthropic_expander

        # Mock Anthropic response
        mock_content = Mock()
        mock_content.text = "Query 1\nQuery 2"
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=150, output_tokens=75)
        mock_client.messages.create.return_value = mock_response

        # Mock cost tracker
        with patch("src.cost_tracker.get_global_tracker") as mock_tracker_getter:
            mock_tracker = Mock()
            mock_tracker_getter.return_value = mock_tracker

            expander.expand("test", num_expansions=2)

            # Verify cost tracking was called with Anthropic provider
            mock_tracker.track_llm.assert_called_once_with(
                "anthropic", "claude-haiku-4-5", 150, 75
            )

    def test_anthropic_fallback_on_error(self, mock_anthropic_expander, caplog):
        """Test graceful fallback when Anthropic API fails."""
        expander, mock_client = mock_anthropic_expander

        # Mock Anthropic to raise error
        mock_client.messages.create.side_effect = Exception("API error")

        with caplog.at_level("WARNING"):
            result = expander.expand("test query", num_expansions=2)

        # Should fall back to original query
        assert result.original_query == "test query"
        assert result.expanded_queries == ["test query"]
        assert result.expansion_method == "fallback"
        assert "Query expansion failed" in caplog.text


class TestPartialExpansion:
    """Test handling when LLM generates fewer expansions than requested."""

    @pytest.fixture
    def mock_openai_expander(self):
        """Create QueryExpander with mocked OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            expander = QueryExpander(
                provider="openai",
                model="gpt-4o-mini",
                openai_api_key="sk-test"
            )
            expander.client = mock_client

            yield expander, mock_client

    def test_partial_expansion_warning(self, mock_openai_expander, caplog):
        """Test warning when LLM generates fewer expansions than requested."""
        expander, mock_client = mock_openai_expander

        # Mock response with only 1 expansion (requested 3)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query variation 1"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        with caplog.at_level("WARNING"):
            result = expander.expand("test query", num_expansions=3)

        # Should log warning about incomplete expansion
        assert "Generated 1/3 expansions" in caplog.text
        assert "LLM response may have been incomplete" in caplog.text

        # Should still return what we got
        assert result.original_query == "test query"
        assert len(result.expanded_queries) == 2  # Original + 1 expansion
        assert result.expansion_method == "llm"

    def test_partial_expansion_with_empty_lines(self, mock_openai_expander):
        """Test that empty lines in LLM response are filtered out."""
        expander, mock_client = mock_openai_expander

        # Mock response with empty lines
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1\n\n\nQuery 2\n\n"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        result = expander.expand("test", num_expansions=2)

        # Should filter out empty lines
        assert len(result.expanded_queries) == 3  # Original + 2 valid expansions
        assert "" not in result.expanded_queries
