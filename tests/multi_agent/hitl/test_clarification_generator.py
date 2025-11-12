"""
Unit tests for ClarificationGenerator.

Tests LLM-based question generation and fallback templates.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.multi_agent.hitl.config import HITLConfig
from src.multi_agent.hitl.clarification_generator import ClarificationGenerator, ClarificationQuestion
from src.multi_agent.hitl.quality_detector import QualityMetrics


@pytest.fixture
def default_config():
    """Default HITL configuration."""
    return HITLConfig()


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client."""
    with patch('src.multi_agent.hitl.clarification_generator.Anthropic') as mock:
        yield mock


@pytest.fixture
def generator(default_config, mock_anthropic):
    """ClarificationGenerator with mocked Anthropic."""
    return ClarificationGenerator(default_config, api_key="test-key")


@pytest.fixture
def low_quality_metrics():
    """Quality metrics indicating need for clarification."""
    return QualityMetrics(
        retrieval_score=0.4,
        semantic_coherence=0.3,
        query_pattern_score=0.3,
        document_diversity=8,
        overall_quality=0.35,
        failing_metrics=["retrieval_score", "semantic_coherence", "query_pattern_score"],
        should_clarify=True
    )


class TestQuestionGeneration:
    """Test LLM-based question generation."""

    @pytest.mark.asyncio
    async def test_successful_generation(self, generator, low_quality_metrics, mock_anthropic):
        """LLM successfully generates questions."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = [Mock(text="""1. What time period are you interested in?
2. Should I focus on specific sections or the entire document?
3. Are you looking for a summary or specific details?""")]

        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="What are the rules?",
            metrics=low_quality_metrics,
            context={"complexity_score": 50, "num_results": 3}
        )

        assert len(questions) == 3
        assert all(isinstance(q, ClarificationQuestion) for q in questions)
        assert questions[0].text == "What time period are you interested in?"
        assert questions[0].id == "q1"

    @pytest.mark.asyncio
    async def test_question_type_inference(self, generator, low_quality_metrics, mock_anthropic):
        """Question types should be inferred correctly."""
        mock_response = Mock()
        mock_response.content = [Mock(text="""1. When was this regulation published?
2. Which specific document should I analyze?
3. What is the context for this question?""")]

        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="Analyze compliance",
            metrics=low_quality_metrics,
            context={}
        )

        assert questions[0].type == "temporal"  # "when"
        assert questions[1].type == "entities"  # "which specific"
        assert questions[2].type == "context"  # "context"

    @pytest.mark.asyncio
    async def test_min_questions_enforced(self, generator, low_quality_metrics, mock_anthropic):
        """If LLM generates < min_questions, use fallbacks."""
        # LLM generates only 1 question
        mock_response = Mock()
        mock_response.content = [Mock(text="1. What time period?")]

        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="What are the rules?",
            metrics=low_quality_metrics,
            context={}
        )

        # Should have at least min_questions (default 2)
        assert len(questions) >= generator.config.min_questions

    @pytest.mark.asyncio
    async def test_max_questions_enforced(self, generator, low_quality_metrics, mock_anthropic):
        """If LLM generates > max_questions, truncate."""
        # LLM generates 10 questions
        mock_response = Mock()
        mock_response.content = [Mock(text="\n".join([f"{i}. Question {i}" for i in range(1, 11)]))]

        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="What are the rules?",
            metrics=low_quality_metrics,
            context={}
        )

        # Should be capped at max_questions (default 5)
        assert len(questions) <= generator.config.max_questions


class TestFallbackQuestions:
    """Test template fallback questions."""

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback(self, generator, low_quality_metrics, mock_anthropic):
        """If LLM fails, use fallback templates."""
        # Mock LLM to raise exception
        generator.client.messages.create = Mock(side_effect=Exception("API Error"))

        questions = await generator.generate(
            query="What are the rules?",
            metrics=low_quality_metrics,
            context={}
        )

        # Should still return questions (fallback templates)
        assert len(questions) >= 2
        assert all(isinstance(q, ClarificationQuestion) for q in questions)

    def test_fallback_covers_failing_metrics(self, generator, low_quality_metrics):
        """Fallback questions should address failing metrics."""
        fallbacks = generator._get_fallback_questions("Query", low_quality_metrics)

        # Should have questions for each failing metric type
        question_types = {q.type for q in fallbacks}

        # retrieval_score failure → intent question
        # query_pattern_score failure → scope question
        # semantic_coherence failure → entities question
        assert len(fallbacks) >= 3
        assert any(q.type in ["intent", "scope", "entities"] for q in fallbacks)

    def test_fallback_respects_max_questions(self, generator, low_quality_metrics):
        """Fallback should not exceed max_questions."""
        fallbacks = generator._get_fallback_questions("Query", low_quality_metrics)
        assert len(fallbacks) <= generator.config.max_questions


class TestPromptBuilding:
    """Test user prompt construction."""

    def test_prompt_includes_quality_issues(self, generator, low_quality_metrics):
        """Prompt should describe quality issues."""
        prompt = generator._build_user_prompt(
            query="What are the rules?",
            metrics=low_quality_metrics,
            context={"complexity_score": 50, "num_results": 3}
        )

        assert "What are the rules?" in prompt
        assert "quality" in prompt.lower() or "issues" in prompt.lower()
        assert "retrieval_score" in prompt or "relevance" in prompt.lower()

    def test_prompt_includes_context(self, generator, low_quality_metrics):
        """Prompt should include additional context."""
        prompt = generator._build_user_prompt(
            query="Test query",
            metrics=low_quality_metrics,
            context={"complexity_score": 75, "num_results": 10}
        )

        assert "75" in prompt  # Complexity score
        assert "10" in prompt  # Number of results


class TestQuestionParsing:
    """Test question parsing from LLM response."""

    def test_parse_numbered_list(self, generator):
        """Parse numbered list format."""
        response = """1. What time period are you interested in?
2. Which specific document should I analyze?
3. Should I focus on summaries or details?"""

        questions = generator._parse_questions(response)

        assert len(questions) == 3
        assert questions[0].text == "What time period are you interested in?"
        assert questions[1].text == "Which specific document should I analyze?"
        assert questions[2].text == "Should I focus on summaries or details?"

    def test_parse_bulleted_list(self, generator):
        """Parse bulleted list format."""
        response = """- What time period?
- Which document?
- What level of detail?"""

        questions = generator._parse_questions(response)

        assert len(questions) == 3
        assert questions[0].text == "What time period?"

    def test_parse_mixed_formats(self, generator):
        """Handle mixed numbering formats."""
        response = """1) First question
2. Second question
3- Third question"""

        questions = generator._parse_questions(response)
        assert len(questions) == 3

    def test_skip_empty_lines(self, generator):
        """Empty lines should be ignored."""
        response = """1. Question one

2. Question two


3. Question three"""

        questions = generator._parse_questions(response)
        assert len(questions) == 3


class TestTypeInference:
    """Test question type inference."""

    def test_temporal_inference(self, generator):
        """Detect temporal questions."""
        questions = [
            "What time period are you interested in?",
            "When was this regulation published?",
            "Should I include recent or historical data?"
        ]
        for q in questions:
            assert generator._infer_question_type(q) == "temporal"

    def test_entities_inference(self, generator):
        """Detect entity questions."""
        questions = [
            "Which specific regulation are you referring to?",
            "Who is the responsible party?",
            "Which document should I analyze?"
        ]
        for q in questions:
            assert generator._infer_question_type(q) == "entities"

    def test_scope_inference(self, generator):
        """Detect scope questions."""
        questions = [
            "Should I focus on specific sections?",
            "What is the breadth of analysis needed?",
            "Should I analyze the entire document?"
        ]
        for q in questions:
            assert generator._infer_question_type(q) == "scope"

    def test_context_inference(self, generator):
        """Detect context questions."""
        questions = [
            "What is the context for this question?",
            "What is the background?",
            "Why are you asking this?"
        ]
        for q in questions:
            assert generator._infer_question_type(q) == "context"

    def test_intent_inference(self, generator):
        """Detect intent questions."""
        questions = [
            "Are you looking for a summary or details?",
            "What analysis do you need?",
            "Do you need compliance information?"
        ]
        for q in questions:
            assert generator._infer_question_type(q) == "intent"


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self, generator, low_quality_metrics, mock_anthropic):
        """Handle empty query gracefully."""
        mock_response = Mock()
        mock_response.content = [Mock(text="1. Can you provide more details?")]
        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="",
            metrics=low_quality_metrics,
            context={}
        )

        assert len(questions) >= 1

    @pytest.mark.asyncio
    async def test_non_english_query(self, generator, low_quality_metrics, mock_anthropic):
        """Handle non-English queries."""
        mock_response = Mock()
        mock_response.content = [Mock(text="1. Jaké časové období?")]
        generator.client.messages.create = Mock(return_value=mock_response)

        questions = await generator.generate(
            query="Co jsou pravidla?",
            metrics=low_quality_metrics,
            context={}
        )

        assert len(questions) >= 1

    @pytest.mark.asyncio
    async def test_api_timeout(self, generator, low_quality_metrics, mock_anthropic):
        """Handle API timeout with fallback."""
        generator.client.messages.create = Mock(side_effect=TimeoutError("API timeout"))

        questions = await generator.generate(
            query="Test query",
            metrics=low_quality_metrics,
            context={}
        )

        # Should fall back to templates
        assert len(questions) >= 2
