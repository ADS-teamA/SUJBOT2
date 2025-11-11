"""
HITL Configuration Schema

Config-driven quality detection and clarification policy.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MetricConfig:
    """Configuration for a single quality metric."""

    enabled: bool = True
    weight: float = 0.25
    threshold: float = 0.5
    description: str = ""


@dataclass
class HITLConfig:
    """
    Human-in-the-Loop configuration.

    Loaded from config_multi_agent_extension.json under multi_agent.clarification
    """

    # === POLICY ===
    enabled: bool = True
    trigger_after_agent: str = "extractor"
    quality_threshold: float = 0.60  # Overall quality score threshold
    min_complexity_score: int = 40  # Don't clarify simple queries
    always_ask_if_zero_results: bool = True
    never_ask_for_simple_queries: bool = True
    max_clarifications_per_query: int = 2

    # === QUALITY METRICS ===
    retrieval_score_metric: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            enabled=True,
            weight=0.30,
            threshold=0.65,
            description="Average relevance score of retrieved chunks"
        )
    )

    semantic_coherence_metric: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            enabled=True,
            weight=0.25,
            threshold=0.30,
            description="Variance in chunk embeddings (high variance = low coherence)"
        )
    )

    query_pattern_metric: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            enabled=True,
            weight=0.25,
            threshold=0.50,
            description="Vague keyword detection"
        )
    )

    document_diversity_metric: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            enabled=True,
            weight=0.20,
            threshold=5.0,
            description="Number of distinct documents retrieved"
        )
    )

    require_multiple_failures: bool = True
    min_failing_metrics: int = 2  # At least 2 metrics must fail

    # === QUESTION GENERATION ===
    question_model: str = "claude-haiku-4-5-20251001"
    question_temperature: float = 0.4
    question_max_tokens: int = 512
    min_questions: int = 2
    max_questions: int = 5
    enable_prompt_caching: bool = True

    # === USER INTERACTION ===
    timeout_seconds: int = 300  # 5 minutes
    allow_skip: bool = True
    enable_multi_round: bool = True
    max_rounds: int = 2

    # === QUERY ENRICHMENT ===
    enrichment_strategy: str = "append_with_context"
    enrichment_template: str = "{original_query}\n\n[Context]: {user_response}"
    max_enriched_length: int = 500

    # === MONITORING ===
    log_all_clarifications: bool = True
    track_quality_improvement: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HITLConfig":
        """Load configuration from dict (from config JSON)."""

        # Extract policy
        policy = config_dict.get("policy", {})

        # Extract metrics configs
        metrics_config = config_dict.get("quality_detection", {}).get("metrics", {})

        def load_metric_config(metric_dict: Dict[str, Any]) -> MetricConfig:
            return MetricConfig(
                enabled=metric_dict.get("enabled", True),
                weight=metric_dict.get("weight", 0.25),
                threshold=metric_dict.get("threshold", 0.5),
                description=metric_dict.get("description", "")
            )

        # Extract question generation config
        question_gen = config_dict.get("question_generation", {})

        # Extract user interaction config
        user_interaction = config_dict.get("user_interaction", {})

        # Extract query enrichment config
        query_enrichment = config_dict.get("query_enrichment", {})

        # Extract monitoring config
        monitoring = config_dict.get("monitoring", {})

        return cls(
            # Policy
            enabled=config_dict.get("enabled", True),
            trigger_after_agent=policy.get("trigger_after_agent", "extractor"),
            quality_threshold=policy.get("quality_threshold", 0.60),
            min_complexity_score=policy.get("min_complexity_score", 40),
            always_ask_if_zero_results=policy.get("always_ask_if_zero_results", True),
            never_ask_for_simple_queries=policy.get("never_ask_for_simple_queries", True),
            max_clarifications_per_query=policy.get("max_clarifications_per_query", 2),

            # Metrics
            retrieval_score_metric=load_metric_config(
                metrics_config.get("retrieval_score", {})
            ),
            semantic_coherence_metric=load_metric_config(
                metrics_config.get("semantic_coherence", {})
            ),
            query_pattern_metric=load_metric_config(
                metrics_config.get("query_pattern", {})
            ),
            document_diversity_metric=load_metric_config(
                metrics_config.get("document_diversity", {})
            ),
            require_multiple_failures=config_dict.get("quality_detection", {}).get(
                "require_multiple_failures", True
            ),
            min_failing_metrics=config_dict.get("quality_detection", {}).get(
                "min_failing_metrics", 2
            ),

            # Question generation
            question_model=question_gen.get("model", "claude-haiku-4-5-20251001"),
            question_temperature=question_gen.get("temperature", 0.4),
            question_max_tokens=question_gen.get("max_tokens", 512),
            min_questions=question_gen.get("min_questions", 2),
            max_questions=question_gen.get("max_questions", 5),
            enable_prompt_caching=question_gen.get("enable_prompt_caching", True),

            # User interaction
            timeout_seconds=user_interaction.get("timeout_seconds", 300),
            allow_skip=user_interaction.get("allow_skip", True),
            enable_multi_round=user_interaction.get("enable_multi_round", True),
            max_rounds=user_interaction.get("max_rounds", 2),

            # Query enrichment
            enrichment_strategy=query_enrichment.get("strategy", "append_with_context"),
            enrichment_template=query_enrichment.get(
                "template",
                "{original_query}\n\n[Context]: {user_response}"
            ),
            max_enriched_length=query_enrichment.get("max_enriched_length", 500),

            # Monitoring
            log_all_clarifications=monitoring.get("log_all_clarifications", True),
            track_quality_improvement=monitoring.get("track_quality_improvement", True),
        )

    def is_enabled(self) -> bool:
        """Check if HITL is enabled."""
        return self.enabled

    def should_trigger_for_complexity(self, complexity_score: int) -> bool:
        """Check if query complexity allows clarification."""
        if self.never_ask_for_simple_queries and complexity_score < self.min_complexity_score:
            return False
        return True
