"""
Tests for HyDE Generator Module

Tests the HyDEGenerator class for:
- Multi-hypothesis generation
- Graceful fallback on errors
- Prompt loading
- Document parsing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agent.hyde_generator import HyDEGenerator, HyDEResult


class TestHyDEGeneratorInitialization:
    """Test HyDEGenerator initialization."""

    def test_init_openai_success(self):
        """Test successful initialization with OpenAI provider."""
        with patch("openai.OpenAI") as mock_openai:
            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test-key"
            )

            assert generator.provider == "openai"
            assert generator.model == "gpt-4o-mini"
            assert generator.num_hypotheses == 3
            mock_openai.assert_called_once_with(api_key="sk-test-key")

    def test_init_anthropic_success(self):
        """Test successful initialization with Anthropic provider."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            generator = HyDEGenerator(
                provider="anthropic",
                model="claude-haiku-4-5",
                anthropic_api_key="sk-ant-test-key",
            )

            assert generator.provider == "anthropic"
            assert generator.model == "claude-haiku-4-5"
            mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_init_missing_openai_key(self):
        """Test initialization fails without OpenAI API key."""
        with pytest.raises(ValueError, match="openai_api_key required"):
            HyDEGenerator(provider="openai", model="gpt-4o-mini")

    def test_init_missing_anthropic_key(self):
        """Test initialization fails without Anthropic API key."""
        with pytest.raises(ValueError, match="anthropic_api_key required"):
            HyDEGenerator(provider="anthropic", model="claude-haiku-4-5")

    def test_init_invalid_provider(self):
        """Test initialization fails with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            HyDEGenerator(provider="invalid", model="test", openai_api_key="sk-test")


class TestHyDEGeneration:
    """Test hypothetical document generation."""

    def test_skip_generation_when_num_docs_0(self):
        """Test that generation is skipped when num_docs=0 (optimization)."""
        with patch("openai.OpenAI"):
            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            result = generator.generate("test query", num_docs=0)

            # Verify result
            assert result.original_query == "test query"
            assert result.hypothetical_docs == []
            assert result.generation_method == "none"
            assert result.model_used is None

    def test_single_doc_generation(self):
        """Test single hypothetical document generation."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock LLM response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = (
                "Radiation dose limits are set at 20 mSv per year for workers."
            )
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response

            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            result = generator.generate("What are radiation dose limits?", num_docs=1)

            # Verify result
            assert result.original_query == "What are radiation dose limits?"
            assert len(result.hypothetical_docs) == 1
            assert "20 mSv" in result.hypothetical_docs[0]
            assert result.generation_method == "llm"
            assert result.model_used == "gpt-4o-mini"

    def test_multi_doc_generation_with_delimiter(self):
        """Test multi-hypothesis generation with '---' delimiter."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock LLM response with delimiter
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = (
                "Doc 1: Radiation limits are 20 mSv/year.\n"
                "---\n"
                "Doc 2: Public exposure is limited to 1 mSv annually.\n"
                "---\n"
                "Doc 3: ICRP recommendations guide occupational limits."
            )
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 60
            mock_client.chat.completions.create.return_value = mock_response

            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            result = generator.generate("What are radiation dose limits?", num_docs=3)

            # Verify result
            assert len(result.hypothetical_docs) == 3
            assert "20 mSv" in result.hypothetical_docs[0]
            assert "1 mSv" in result.hypothetical_docs[1]
            assert "ICRP" in result.hypothetical_docs[2]

    def test_graceful_fallback_on_llm_error(self):
        """Test graceful fallback when LLM call fails."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock LLM error
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            result = generator.generate("test query", num_docs=3)

            # Verify graceful fallback
            assert result.original_query == "test query"
            assert result.hypothetical_docs == []
            assert result.generation_method == "fallback"


class TestPromptLoading:
    """Test prompt template loading."""

    def test_load_prompt_from_file(self):
        """Test loading prompt template from file."""
        with patch("openai.OpenAI"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch(
                    "pathlib.Path.read_text",
                    return_value="Write a passage: {query}\nUse formal language.",
                ):
                    generator = HyDEGenerator(
                        provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
                    )

                    assert "{query}" in generator.prompt_template
                    assert "formal language" in generator.prompt_template

    def test_fallback_prompt_when_file_missing(self):
        """Test fallback to default prompt when file is missing."""
        with patch("openai.OpenAI"):
            with patch("pathlib.Path.exists", return_value=False):
                generator = HyDEGenerator(
                    provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
                )

                # Should use fallback prompt
                assert "{query}" in generator.prompt_template
                assert "factual passage" in generator.prompt_template


class TestDocumentParsing:
    """Test parsing of hypothetical documents from LLM response."""

    def test_parse_single_doc(self):
        """Test parsing single document (no delimiter)."""
        with patch("openai.OpenAI"):
            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            docs = generator._parse_hypothetical_docs(
                "Single hypothetical document text.", num_docs=1
            )

            assert len(docs) == 1
            assert docs[0] == "Single hypothetical document text."

    def test_parse_multi_doc_with_delimiter(self):
        """Test parsing multiple documents with '---' delimiter."""
        with patch("openai.OpenAI"):
            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            text = "Doc 1 text\n---\nDoc 2 text\n---\nDoc 3 text"
            docs = generator._parse_hypothetical_docs(text, num_docs=3)

            assert len(docs) == 3
            assert docs[0] == "Doc 1 text"
            assert docs[1] == "Doc 2 text"
            assert docs[2] == "Doc 3 text"

    def test_parse_fallback_double_newlines(self):
        """Test fallback to double newlines when no delimiter."""
        with patch("openai.OpenAI"):
            generator = HyDEGenerator(
                provider="openai", model="gpt-4o-mini", openai_api_key="sk-test"
            )

            text = "Doc 1 text\n\nDoc 2 text\n\nDoc 3 text"
            docs = generator._parse_hypothetical_docs(text, num_docs=3)

            assert len(docs) == 3
            assert docs[0] == "Doc 1 text"
            assert docs[1] == "Doc 2 text"
            assert docs[2] == "Doc 3 text"
