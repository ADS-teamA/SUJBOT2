"""
Human-in-the-Loop (HITL) Clarification System

Implements ChatGPT Deep Research-style clarifications for ambiguous/vague queries.

Flow:
1. Extractor retrieves documents
2. Quality Detector evaluates retrieval quality (4 metrics)
3. If quality < threshold → Generate clarifying questions
4. User responds via chat
5. Enrich query with user response
6. Re-run Extractor → Continue workflow

Components:
- QualityDetector: 4 metrics to detect poor queries
- ClarificationGenerator: LLM-based question generation
- ContextEnricher: Merge user answer into query context
"""

from .config import HITLConfig
from .quality_detector import QualityDetector, QualityMetrics
from .clarification_generator import ClarificationGenerator
from .context_enricher import ContextEnricher

__all__ = [
    "HITLConfig",
    "QualityDetector",
    "QualityMetrics",
    "ClarificationGenerator",
    "ContextEnricher",
]
