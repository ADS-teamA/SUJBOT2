"""
Clarification Generator - LLM-based question generation.

Generates 2-5 clarifying questions based on:
- Original user query
- Quality metrics (what failed)
- Retrieved documents (context)

Uses Claude Haiku for fast, cost-effective generation.
"""

import logging
import re
from typing import List, Dict, Any

from anthropic import Anthropic
from pydantic import BaseModel, Field

from .config import HITLConfig
from .quality_detector import QualityMetrics

logger = logging.getLogger(__name__)


class ClarificationQuestion(BaseModel):
    """A single clarification question."""

    id: str = Field(..., description="Unique question ID (q1, q2, ...)")
    text: str = Field(..., description="Question text")
    type: str = Field(..., description="Question type: temporal, scope, entities, context, intent")
    required: bool = Field(default=False, description="Must be answered?")


class ClarificationGenerator:
    """
    Generate clarifying questions using LLM.

    Usage:
        generator = ClarificationGenerator(hitl_config, anthropic_api_key)
        questions = await generator.generate(
            query="What are the rules?",
            metrics=quality_metrics,
            context={"documents": [...]}
        )
    """

    # System prompt for question generation
    SYSTEM_PROMPT = """You are a query clarification assistant. Your role is to help users refine vague or ambiguous questions to get better search results.

When given a user query and quality issues, generate 2-5 specific, actionable clarifying questions.

IMPORTANT GUIDELINES:
1. Questions must be short (1 sentence, max 120 characters)
2. Each question should address a different aspect (time period, scope, entities, context, intent)
3. Use simple language (avoid jargon unless in original query)
4. Make questions specific to the query domain (legal, technical, etc.)
5. Avoid yes/no questions - prefer open-ended questions

QUESTION TYPES:
- temporal: Time period (e.g., "What time period are you interested in?")
- scope: Breadth/depth (e.g., "Should I focus on specific sections or the entire document?")
- entities: Specific items (e.g., "Which regulation are you referring to?")
- context: Background (e.g., "What is the context for this question?")
- intent: Goal (e.g., "Are you looking for a summary or specific details?")

OUTPUT FORMAT:
Return questions as a numbered list, one per line:
1. [Question text]
2. [Question text]
3. [Question text]

Example:
1. What time period should I focus on (recent, historical, specific year)?
2. Are you interested in US, EU, or other jurisdictions?
3. Should the answer include technical details or just a summary?"""

    def __init__(self, config: HITLConfig, api_key: str):
        """
        Initialize clarification generator.

        Args:
            config: HITL configuration
            api_key: Anthropic API key
        """
        self.config = config
        self.client = Anthropic(api_key=api_key)
        logger.info(f"ClarificationGenerator initialized with model={config.question_model}")

    async def generate(
        self,
        query: str,
        metrics: QualityMetrics,
        context: Dict[str, Any]
    ) -> List[ClarificationQuestion]:
        """
        Generate clarifying questions.

        Args:
            query: Original user query
            metrics: Quality assessment metrics
            context: Additional context (documents, complexity, etc.)

        Returns:
            List of 2-5 clarification questions
        """
        try:
            # Build user prompt
            user_prompt = self._build_user_prompt(query, metrics, context)

            # Call LLM
            response = self.client.messages.create(
                model=self.config.question_model,
                max_tokens=self.config.question_max_tokens,
                temperature=self.config.question_temperature,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract questions from response
            response_text = response.content[0].text
            questions = self._parse_questions(response_text)

            # Validate and truncate
            questions = questions[: self.config.max_questions]

            if len(questions) < self.config.min_questions:
                logger.warning(
                    f"LLM generated only {len(questions)} questions, expected {self.config.min_questions}. "
                    "Using fallback templates."
                )
                questions.extend(self._get_fallback_questions(query, metrics))
                questions = questions[: self.config.max_questions]

            logger.info(f"Generated {len(questions)} clarification questions")
            return questions

        except Exception as e:
            logger.error(f"Failed to generate questions via LLM: {e}", exc_info=True)
            # Fallback to template questions
            return self._get_fallback_questions(query, metrics)

    def _build_user_prompt(
        self,
        query: str,
        metrics: QualityMetrics,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM."""
        # Build quality issues description
        issues_desc = []
        if "retrieval_score" in metrics.failing_metrics:
            issues_desc.append(f"- Low relevance score ({metrics.retrieval_score:.2f}): Retrieved documents may not be relevant")

        if "semantic_coherence" in metrics.failing_metrics:
            issues_desc.append(f"- Low semantic coherence ({metrics.semantic_coherence:.2f}): Results are scattered across different topics")

        if "query_pattern_score" in metrics.failing_metrics:
            issues_desc.append(f"- Vague query ({metrics.query_pattern_score:.2f}): Query contains generic terms")

        if "document_diversity" in metrics.failing_metrics:
            issues_desc.append(f"- High document diversity ({metrics.document_diversity} docs): Results span too many documents")

        issues_text = "\n".join(issues_desc) if issues_desc else "General quality concerns"

        # Get complexity if available
        complexity = context.get("complexity_score", "unknown")

        # Get number of results
        num_results = context.get("num_results", metrics.document_diversity)

        prompt = f"""USER QUERY:
"{query}"

QUALITY ISSUES DETECTED:
{issues_text}

ADDITIONAL CONTEXT:
- Query complexity: {complexity}
- Results retrieved: {num_results}
- Overall quality score: {metrics.overall_quality:.2f}

Generate 2-5 clarifying questions to help the user refine this query for better results. Focus on the quality issues identified above."""

        return prompt

    def _parse_questions(self, response_text: str) -> List[ClarificationQuestion]:
        """
        Parse questions from LLM response.

        Expected format:
        1. Question text
        2. Question text
        3. Question text
        """
        questions = []

        # Split by lines
        lines = response_text.strip().split('\n')

        for idx, line in enumerate(lines, start=1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove numbering (1., 2., -, *, etc.)
            cleaned = re.sub(r'^[\d\-\*•]+[\.\)]\s*', '', line)

            if not cleaned:
                continue

            # Infer question type from keywords
            question_type = self._infer_question_type(cleaned)

            questions.append(ClarificationQuestion(
                id=f"q{idx}",
                text=cleaned,
                type=question_type,
                required=False
            ))

        logger.debug(f"Parsed {len(questions)} questions from LLM response")
        return questions

    def _infer_question_type(self, question_text: str) -> str:
        """Infer question type from text content."""
        text_lower = question_text.lower()

        if any(kw in text_lower for kw in ["time", "period", "year", "date", "when", "recent", "historical"]):
            return "temporal"
        elif any(kw in text_lower for kw in ["which", "specific", "particular", "name", "who"]):
            return "entities"
        elif any(kw in text_lower for kw in ["scope", "focus", "breadth", "depth", "entire", "section"]):
            return "scope"
        elif any(kw in text_lower for kw in ["context", "background", "purpose", "why"]):
            return "context"
        elif any(kw in text_lower for kw in ["summary", "details", "analysis", "looking for", "need"]):
            return "intent"
        else:
            return "general"

    def _get_fallback_questions(
        self,
        query: str,
        metrics: QualityMetrics
    ) -> List[ClarificationQuestion]:
        """
        Generate template fallback questions when LLM fails.

        These are generic but always relevant.
        """
        templates = []

        # Always useful: temporal scope
        templates.append(ClarificationQuestion(
            id="q_temporal",
            text="What time period are you interested in? (e.g., recent, specific year)",
            type="temporal",
            required=False
        ))

        # For low retrieval: clarify intent
        if "retrieval_score" in metrics.failing_metrics:
            templates.append(ClarificationQuestion(
                id="q_intent",
                text="Are you looking for a summary, specific details, or compliance information?",
                type="intent",
                required=False
            ))

        # For vague queries: clarify scope
        if "query_pattern_score" in metrics.failing_metrics:
            templates.append(ClarificationQuestion(
                id="q_scope",
                text="Should I focus on specific sections or search the entire document?",
                type="scope",
                required=False
            ))

        # For scattered results: identify entities
        if "semantic_coherence" in metrics.failing_metrics or "document_diversity" in metrics.failing_metrics:
            templates.append(ClarificationQuestion(
                id="q_entities",
                text="Are you looking for information about a specific regulation, standard, or document?",
                type="entities",
                required=False
            ))

        # General context question
        templates.append(ClarificationQuestion(
            id="q_context",
            text="What is the context or purpose for this question?",
            type="context",
            required=False
        ))

        logger.info(f"Using {len(templates)} fallback template questions")
        return templates[: self.config.max_questions]
