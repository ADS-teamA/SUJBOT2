"""
Complexity Analyzer - Query complexity assessment and routing decisions.

Analyzes query characteristics to determine:
1. Complexity score (0-100)
2. Query type (compliance, risk, synthesis, search, reporting)
3. Recommended workflow pattern (simple, standard, complex)
4. Agent sequence for optimal processing
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..core.state import QueryType

logger = logging.getLogger(__name__)


class WorkflowPattern(Enum):
    """Workflow patterns based on complexity."""

    SIMPLE = "simple"  # < 30: Extractor → Report Generator
    STANDARD = "standard"  # 30-70: Extractor → Classifier → Domain Agent → Report Generator
    COMPLEX = "complex"  # > 70: Full pipeline with all agents


class ComplexityAnalyzer:
    """
    Analyzes query complexity and determines routing.

    Uses keyword matching, query structure analysis, and heuristics
    to score complexity and recommend agent sequences.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize complexity analyzer.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Complexity thresholds
        self.threshold_low = self.config.get("complexity_threshold_low", 30)
        self.threshold_high = self.config.get("complexity_threshold_high", 70)

        # Keywords for different complexity levels and query types
        self.compliance_keywords = [
            "gdpr",
            "ccpa",
            "hipaa",
            "sox",
            "compliance",
            "regulation",
            "legal",
            "regulatory",
        ]

        self.risk_keywords = [
            "risk",
            "safety",
            "liability",
            "impact",
            "hazard",
            "danger",
            "assess",
        ]

        self.synthesis_keywords = [
            "gap",
            "missing",
            "complete",
            "comprehensive",
            "coverage",
            "analyze",
        ]

        self.simple_keywords = ["find", "get", "show", "what", "where", "when"]

        logger.info(
            f"ComplexityAnalyzer initialized (thresholds: {self.threshold_low}/{self.threshold_high})"
        )

    def analyze(self, query: str) -> Dict[str, any]:
        """
        Analyze query complexity and determine routing.

        Args:
            query: User query string

        Returns:
            Dict with:
                - complexity_score (int 0-100)
                - query_type (QueryType enum)
                - workflow_pattern (WorkflowPattern enum)
                - agent_sequence (List[str])
                - reasoning (str)
        """
        logger.info(f"Analyzing complexity for query: {query[:100]}...")

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(query)

        # Determine query type
        query_type = self._determine_query_type(query)

        # Determine workflow pattern
        workflow_pattern = self._determine_workflow_pattern(complexity_score)

        # Build agent sequence
        agent_sequence = self._build_agent_sequence(
            complexity_score, query_type, workflow_pattern
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            complexity_score, query_type, workflow_pattern, query
        )

        result = {
            "complexity_score": complexity_score,
            "query_type": query_type,
            "workflow_pattern": workflow_pattern,
            "agent_sequence": agent_sequence,
            "reasoning": reasoning,
        }

        logger.info(
            f"Complexity analysis complete: score={complexity_score}, "
            f"type={query_type.value}, pattern={workflow_pattern.value}"
        )

        return result

    def _calculate_complexity_score(self, query: str) -> int:
        """
        Calculate complexity score (0-100) based on query characteristics.

        Factors:
        - Query length (longer = more complex)
        - Number of clauses (multiple AND/OR = more complex)
        - Specialized keywords (compliance, risk = more complex)
        - Question complexity (multi-part questions = more complex)

        Args:
            query: Query string

        Returns:
            Complexity score (0-100)
        """
        score = 0

        # Base score from query length
        length_score = min(30, len(query) // 10)  # Max 30 points
        score += length_score

        # Score from query structure
        query_lower = query.lower()

        # Multiple clauses (AND, OR, multiple sentences)
        clauses = len(re.split(r"[.?!;]|\band\b|\bor\b", query))
        clause_score = min(20, clauses * 5)  # Max 20 points
        score += clause_score

        # Compliance keywords (high complexity)
        if any(kw in query_lower for kw in self.compliance_keywords):
            score += 25

        # Risk keywords (high complexity)
        if any(kw in query_lower for kw in self.risk_keywords):
            score += 20

        # Synthesis/analysis keywords (medium-high complexity)
        if any(kw in query_lower for kw in self.synthesis_keywords):
            score += 15

        # Simple retrieval keywords (low complexity)
        if any(kw in query_lower for kw in self.simple_keywords):
            score -= 10

        # Question complexity
        question_marks = query.count("?")
        if question_marks > 1:
            score += 10  # Multiple questions increase complexity

        # Normalize to 0-100
        score = max(0, min(100, score))

        return score

    def _determine_query_type(self, query: str) -> QueryType:
        """
        Determine query type from keywords.

        Args:
            query: Query string

        Returns:
            QueryType enum value
        """
        query_lower = query.lower()

        # Check for compliance queries
        if any(kw in query_lower for kw in self.compliance_keywords):
            return QueryType.COMPLIANCE

        # Check for risk queries
        if any(kw in query_lower for kw in self.risk_keywords):
            return QueryType.RISK

        # Check for synthesis queries
        if any(kw in query_lower for kw in self.synthesis_keywords):
            return QueryType.SYNTHESIS

        # Check for reporting queries
        if any(kw in query_lower for kw in ["report", "summary", "overview"]):
            return QueryType.REPORTING

        # Default to search
        return QueryType.SEARCH

    def _determine_workflow_pattern(self, complexity_score: int) -> WorkflowPattern:
        """
        Determine workflow pattern based on complexity score.

        Args:
            complexity_score: Score from 0-100

        Returns:
            WorkflowPattern enum value
        """
        if complexity_score < self.threshold_low:
            return WorkflowPattern.SIMPLE
        elif complexity_score < self.threshold_high:
            return WorkflowPattern.STANDARD
        else:
            return WorkflowPattern.COMPLEX

    def _build_agent_sequence(
        self,
        complexity_score: int,
        query_type: QueryType,
        workflow_pattern: WorkflowPattern,
    ) -> List[str]:
        """
        Build agent sequence based on complexity and query type.

        Args:
            complexity_score: Complexity score (0-100)
            query_type: Query type enum
            workflow_pattern: Workflow pattern enum

        Returns:
            List of agent names in execution order
        """
        # Always start with orchestrator (already ran, but include in sequence)
        sequence = ["extractor"]  # Extractor always first after orchestrator

        # Add agents based on workflow pattern
        if workflow_pattern == WorkflowPattern.SIMPLE:
            # Simple: Just extractor and report generator
            sequence.append("report_generator")

        elif workflow_pattern == WorkflowPattern.STANDARD:
            # Standard: Add classifier and domain-specific agent
            sequence.append("classifier")

            # Add domain-specific agent based on query type
            if query_type == QueryType.COMPLIANCE:
                sequence.append("compliance")
            elif query_type == QueryType.RISK:
                sequence.append("risk_verifier")
            elif query_type == QueryType.SYNTHESIS:
                sequence.append("gap_synthesizer")

            sequence.append("report_generator")

        else:  # COMPLEX
            # Complex: Full pipeline
            sequence.extend(
                [
                    "classifier",
                    "compliance",
                    "risk_verifier",
                    "citation_auditor",
                    "gap_synthesizer",
                    "report_generator",
                ]
            )

        return sequence

    def _generate_reasoning(
        self,
        complexity_score: int,
        query_type: QueryType,
        workflow_pattern: WorkflowPattern,
        query: str,
    ) -> str:
        """
        Generate human-readable reasoning for routing decision.

        Args:
            complexity_score: Complexity score
            query_type: Query type enum
            workflow_pattern: Workflow pattern enum
            query: Original query string

        Returns:
            Reasoning string
        """
        reasoning_parts = []

        # Complexity explanation
        if complexity_score < 30:
            reasoning_parts.append(f"Low complexity ({complexity_score}/100) - simple retrieval")
        elif complexity_score < 70:
            reasoning_parts.append(
                f"Medium complexity ({complexity_score}/100) - requires domain analysis"
            )
        else:
            reasoning_parts.append(
                f"High complexity ({complexity_score}/100) - comprehensive analysis needed"
            )

        # Query type explanation
        reasoning_parts.append(f"Query type: {query_type.value}")

        # Workflow pattern explanation
        if workflow_pattern == WorkflowPattern.SIMPLE:
            reasoning_parts.append("Using simple pattern (extractor → report)")
        elif workflow_pattern == WorkflowPattern.STANDARD:
            reasoning_parts.append(
                "Using standard pattern (extractor → classifier → domain agent → report)"
            )
        else:
            reasoning_parts.append("Using complex pattern (full agent pipeline)")

        # Keyword-based reasoning
        query_lower = query.lower()
        if any(kw in query_lower for kw in self.compliance_keywords):
            reasoning_parts.append("Detected compliance/regulatory focus")

        if any(kw in query_lower for kw in self.risk_keywords):
            reasoning_parts.append("Detected risk assessment focus")

        return ". ".join(reasoning_parts) + "."
