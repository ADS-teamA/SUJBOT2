"""
LLM-as-Judge Evaluation System.

Uses a fast LLM (e.g., Claude Haiku) to evaluate agent outputs
against quality criteria: relevance, coherence, correctness.

Based on:
- OpenEvals LLM-as-Judge patterns
- LangSmith evaluation best practices

Usage:
    judge = LLMJudge(config)

    # Evaluate a single response
    scores = await judge.evaluate(
        query="What are the compliance requirements?",
        response="Based on the document...",
        context="Retrieved document chunks...",
        criteria=["relevance", "coherence", "groundedness"]
    )

    # Submit scores to LangSmith
    langsmith.send_multiple_feedback(scores.to_feedback_dict(), run_id=run_id)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationCriteria(str, Enum):
    """Available evaluation criteria for LLM judge."""

    RELEVANCE = "relevance"           # Is response relevant to query?
    COHERENCE = "coherence"           # Is response logically coherent?
    GROUNDEDNESS = "groundedness"     # Is response grounded in context?
    CORRECTNESS = "correctness"       # Is response factually correct?
    COMPLETENESS = "completeness"     # Does response fully answer query?
    CONCISENESS = "conciseness"       # Is response appropriately concise?


# Evaluation prompts for each criterion
EVALUATION_PROMPTS = {
    EvaluationCriteria.RELEVANCE: """
Evaluate how relevant the response is to the user's query.

Query: {query}
Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Completely irrelevant, doesn't address the query at all
- 0.5 = Partially relevant, addresses some aspects but misses key points
- 1.0 = Highly relevant, directly and fully addresses the query

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
    EvaluationCriteria.COHERENCE: """
Evaluate how coherent and well-structured the response is.

Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Incoherent, contradictory, or nonsensical
- 0.5 = Some logical flow but with gaps or unclear sections
- 1.0 = Clear, logical, well-organized response

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
    EvaluationCriteria.GROUNDEDNESS: """
Evaluate how well the response is grounded in the provided context.

Context: {context}
Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Response contains information not in context (hallucination)
- 0.5 = Mix of grounded and ungrounded claims
- 1.0 = All claims are supported by the context

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
    EvaluationCriteria.CORRECTNESS: """
Evaluate the factual correctness of the response based on the context.

Context: {context}
Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Contains significant factual errors
- 0.5 = Mostly correct with minor inaccuracies
- 1.0 = Factually accurate throughout

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
    EvaluationCriteria.COMPLETENESS: """
Evaluate how completely the response answers the query given the context.

Query: {query}
Context: {context}
Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Major aspects of the query left unanswered
- 0.5 = Answers main question but misses secondary aspects
- 1.0 = Comprehensively addresses all aspects of the query

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
    EvaluationCriteria.CONCISENESS: """
Evaluate how concise the response is while still being complete.

Query: {query}
Response: {response}

Score from 0.0 to 1.0 where:
- 0.0 = Extremely verbose, much unnecessary content
- 0.5 = Some unnecessary elaboration
- 1.0 = Appropriately concise, no unnecessary content

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
""",
}


@dataclass
class JudgeScore:
    """Score from a single evaluation criterion."""

    criterion: EvaluationCriteria
    score: float  # 0.0 to 1.0
    reasoning: str
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result from LLM judge."""

    scores: Dict[EvaluationCriteria, JudgeScore] = field(default_factory=dict)
    overall_score: float = 0.0
    model_used: str = ""
    evaluation_duration_ms: float = 0.0

    def to_feedback_dict(self) -> Dict[str, float]:
        """
        Convert to LangSmith feedback format.

        Returns:
            Dict mapping feedback keys to scores
        """
        feedback = {}
        for criterion, score in self.scores.items():
            if score.error is None:
                feedback[f"judge_{criterion.value}"] = score.score
        feedback["judge_overall"] = self.overall_score
        return feedback

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scores": {
                k.value: {
                    "score": v.score,
                    "reasoning": v.reasoning,
                    "error": v.error,
                }
                for k, v in self.scores.items()
            },
            "overall_score": self.overall_score,
            "model_used": self.model_used,
            "evaluation_duration_ms": self.evaluation_duration_ms,
        }


class LLMJudge:
    """
    LLM-as-Judge for evaluating agent outputs.

    Uses a fast, cost-effective model (e.g., Claude Haiku) to score
    agent responses against quality criteria.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM Judge.

        Args:
            config: Evaluation config with:
                - llm_judge_model: Model to use (default: claude-haiku-4-5)
                - llm_judge_criteria: List of criteria to evaluate
                - enabled: Whether judge is enabled
        """
        self.config = config
        self.enabled = config.get("enable_llm_judge", False)
        self.model = config.get("llm_judge_model", "claude-haiku-4-5")
        self.default_criteria = config.get(
            "llm_judge_criteria",
            ["relevance", "groundedness", "coherence"]
        )

        # Lazy-loaded provider
        self._provider = None

        logger.info(
            f"LLMJudge initialized: enabled={self.enabled}, "
            f"model={self.model}, criteria={self.default_criteria}"
        )

    def _get_provider(self):
        """Lazy-load the provider to avoid import cycles."""
        if self._provider is None:
            try:
                from ...agent.providers.factory import create_provider
                self._provider = create_provider(self.model)
                logger.info(f"LLMJudge provider created: {self.model}")
            except Exception as e:
                logger.error(f"Failed to create judge provider: {e}")
                raise
        return self._provider

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a response against criteria.

        Args:
            query: Original user query
            response: Agent's response to evaluate
            context: Retrieved context (for groundedness/correctness)
            criteria: List of criteria to evaluate (uses defaults if None)

        Returns:
            EvaluationResult with scores for each criterion
        """
        import time
        from datetime import datetime

        start_time = time.time()

        if not self.enabled:
            logger.debug("LLMJudge disabled, returning empty result")
            return EvaluationResult()

        # Use default criteria if not specified
        criteria_list = criteria or self.default_criteria
        criteria_enums = [
            EvaluationCriteria(c) if isinstance(c, str) else c
            for c in criteria_list
        ]

        result = EvaluationResult(model_used=self.model)

        # Evaluate each criterion
        for criterion in criteria_enums:
            score = await self._evaluate_criterion(
                criterion=criterion,
                query=query,
                response=response,
                context=context or "",
            )
            result.scores[criterion] = score

        # Calculate overall score (average of non-error scores)
        valid_scores = [s.score for s in result.scores.values() if s.error is None]
        if valid_scores:
            result.overall_score = sum(valid_scores) / len(valid_scores)

        result.evaluation_duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"LLMJudge evaluation complete: overall={result.overall_score:.2f}, "
            f"duration={result.evaluation_duration_ms:.0f}ms"
        )

        return result

    async def _evaluate_criterion(
        self,
        criterion: EvaluationCriteria,
        query: str,
        response: str,
        context: str,
    ) -> JudgeScore:
        """
        Evaluate a single criterion.

        Args:
            criterion: Criterion to evaluate
            query: User query
            response: Agent response
            context: Retrieved context

        Returns:
            JudgeScore with score and reasoning
        """
        import json

        try:
            # Get prompt template for this criterion
            prompt_template = EVALUATION_PROMPTS.get(criterion)
            if not prompt_template:
                return JudgeScore(
                    criterion=criterion,
                    score=0.0,
                    reasoning="",
                    error=f"No prompt template for criterion: {criterion.value}"
                )

            # Format prompt
            prompt = prompt_template.format(
                query=query[:1000],  # Truncate to avoid token limits
                response=response[:2000],
                context=context[:3000],
            )

            # Call LLM
            provider = self._get_provider()
            llm_response = await self._call_llm(provider, prompt)

            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_str = llm_response.strip()
                if json_str.startswith("```"):
                    # Handle markdown code blocks
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]

                parsed = json.loads(json_str)
                score = float(parsed.get("score", 0.0))
                reasoning = parsed.get("reasoning", "")

                # Clamp score to valid range
                score = max(0.0, min(1.0, score))

                return JudgeScore(
                    criterion=criterion,
                    score=score,
                    reasoning=reasoning,
                )

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to parse judge response for {criterion.value}: {e}"
                )
                return JudgeScore(
                    criterion=criterion,
                    score=0.5,  # Default to middle score on parse error
                    reasoning=f"Parse error: {llm_response[:100]}",
                    error=str(e),
                )

        except Exception as e:
            logger.error(f"Judge evaluation failed for {criterion.value}: {e}")
            return JudgeScore(
                criterion=criterion,
                score=0.0,
                reasoning="",
                error=str(e),
            )

    async def _call_llm(self, provider, prompt: str) -> str:
        """
        Call LLM provider for evaluation.

        Args:
            provider: LLM provider instance
            prompt: Evaluation prompt

        Returns:
            LLM response text
        """
        try:
            # Use provider's chat method
            messages = [{"role": "user", "content": prompt}]

            if hasattr(provider, 'chat_async'):
                response = await provider.chat_async(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.0,  # Deterministic for evaluation
                )
            elif hasattr(provider, 'chat'):
                # Sync fallback
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: provider.chat(
                        messages=messages,
                        max_tokens=500,
                        temperature=0.0,
                    )
                )
            else:
                raise AttributeError("Provider has no chat method")

            # Extract text from response
            if isinstance(response, dict):
                return response.get("content", response.get("text", str(response)))
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def is_enabled(self) -> bool:
        """Check if judge is enabled."""
        return self.enabled


def create_llm_judge(config: Dict[str, Any]) -> Optional[LLMJudge]:
    """
    Create LLM Judge from configuration.

    Args:
        config: Multi-agent config dict

    Returns:
        LLMJudge instance or None if disabled
    """
    eval_config = config.get("evaluation", {})

    if not eval_config.get("enable_llm_judge", False):
        logger.info("LLM Judge disabled in configuration")
        return None

    try:
        judge = LLMJudge(eval_config)
        logger.info("LLM Judge created successfully")
        return judge
    except Exception as e:
        logger.error(f"Failed to create LLM Judge: {e}")
        return None
