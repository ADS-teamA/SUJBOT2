"""
Risk Scorer for compliance analysis.

Assesses risk severity of compliance issues using multi-factor analysis
and classifies severity levels (CRITICAL/HIGH/MEDIUM/LOW).
"""

import logging
import numpy as np
from typing import List, Dict

from .compliance_types import (
    ComplianceIssue,
    ComplianceStatus,
    IssueSeverity,
    RequirementType
)


class RiskScorer:
    """Assess risk severity of compliance issues."""

    def __init__(self, config: Dict = None):
        """
        Initialize RiskScorer.

        Args:
            config: Configuration dictionary with risk scoring parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Load weights from config or use defaults
        self.weights = self.config.get('risk_scoring', {
            'mandatory_weight': 0.3,
            'issue_type_weight': 0.3,
            'prohibition_weight': 0.2,
            'temporal_weight': 0.1,
            'confidence_weight': 0.1
        })

        # Load severity thresholds from config or use defaults
        self.thresholds = self.config.get('severity_thresholds', {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.0
        })

    def score_issues(self, issues: List[ComplianceIssue]) -> List[ComplianceIssue]:
        """
        Assign risk scores and severity levels to issues.

        Modifies issues in-place and returns them.

        Args:
            issues: List of ComplianceIssue objects

        Returns:
            Same list with updated risk scores and severity levels
        """
        for issue in issues:
            # Compute risk factors
            risk_factors = self._compute_risk_factors(issue)

            # Aggregate into overall risk score
            risk_score = self._aggregate_risk_score(risk_factors)

            # Classify severity
            severity = self._classify_severity(issue.status, risk_score)

            # Update issue
            issue.risk_factors = risk_factors
            issue.risk_score = risk_score
            issue.severity = severity
            issue.priority = self._severity_to_priority(severity)

        self.logger.info(f"Scored {len(issues)} issues")
        return issues

    def _compute_risk_factors(self, issue: ComplianceIssue) -> Dict[str, float]:
        """Compute individual risk factors."""
        factors = {}

        # Factor 1: Mandatory vs. optional requirement
        if any(req.is_mandatory for req in issue.law_requirements):
            factors["mandatory"] = 1.0
        else:
            factors["mandatory"] = 0.3

        # Factor 2: Type of issue
        status_weights = {
            ComplianceStatus.CONFLICT: 1.0,
            ComplianceStatus.MISSING: 0.9,
            ComplianceStatus.DEVIATION: 0.5,
            ComplianceStatus.UNCERTAIN: 0.3,
            ComplianceStatus.NOT_APPLICABLE: 0.0,
            ComplianceStatus.COMPLIANT: 0.0
        }
        factors["issue_type"] = status_weights.get(issue.status, 0.5)

        # Factor 3: Requirement type (prohibition violations are serious)
        has_prohibition = any(
            req.requirement_type == RequirementType.PROHIBITION
            for req in issue.law_requirements
        )
        factors["prohibition_violation"] = 1.0 if has_prohibition else 0.5

        # Factor 4: Temporal constraints (deadlines increase urgency)
        has_temporal = any(
            req.temporal_constraint is not None
            for req in issue.law_requirements
        )
        factors["temporal"] = 0.8 if has_temporal else 0.4

        # Factor 5: Detection confidence
        factors["confidence"] = issue.detection_confidence

        return factors

    def _aggregate_risk_score(self, factors: Dict[str, float]) -> float:
        """
        Aggregate risk factors into single score.

        Args:
            factors: Dictionary of risk factor scores

        Returns:
            Aggregated risk score (0.0 to 1.0)
        """
        # Weighted average
        weights = {
            "mandatory": self.weights.get('mandatory_weight', 0.3),
            "issue_type": self.weights.get('issue_type_weight', 0.3),
            "prohibition_violation": self.weights.get('prohibition_weight', 0.2),
            "temporal": self.weights.get('temporal_weight', 0.1),
            "confidence": self.weights.get('confidence_weight', 0.1)
        }

        risk_score = sum(factors.get(k, 0.5) * weights[k] for k in weights)

        # Clip to valid range
        return float(np.clip(risk_score, 0.0, 1.0))

    def _classify_severity(
        self,
        status: ComplianceStatus,
        risk_score: float
    ) -> IssueSeverity:
        """
        Classify severity based on status and risk score.

        Args:
            status: ComplianceStatus of the issue
            risk_score: Computed risk score

        Returns:
            IssueSeverity classification
        """
        # Conflicts with high risk are critical
        if status == ComplianceStatus.CONFLICT and risk_score >= self.thresholds.get('critical', 0.8):
            return IssueSeverity.CRITICAL

        # Missing mandatory requirements are high severity
        if status == ComplianceStatus.MISSING and risk_score >= 0.7:
            return IssueSeverity.HIGH

        # General risk-based classification
        if risk_score >= self.thresholds.get('critical', 0.8):
            return IssueSeverity.CRITICAL
        elif risk_score >= self.thresholds.get('high', 0.6):
            return IssueSeverity.HIGH
        elif risk_score >= self.thresholds.get('medium', 0.4):
            return IssueSeverity.MEDIUM
        elif risk_score >= self.thresholds.get('low', 0.0):
            return IssueSeverity.LOW
        else:
            return IssueSeverity.INFO

    def _severity_to_priority(self, severity: IssueSeverity) -> int:
        """
        Convert severity to priority number.

        Args:
            severity: IssueSeverity level

        Returns:
            Priority as integer (1=highest, 5=lowest)
        """
        priority_map = {
            IssueSeverity.CRITICAL: 1,
            IssueSeverity.HIGH: 2,
            IssueSeverity.MEDIUM: 3,
            IssueSeverity.LOW: 4,
            IssueSeverity.INFO: 5
        }
        return priority_map.get(severity, 3)
