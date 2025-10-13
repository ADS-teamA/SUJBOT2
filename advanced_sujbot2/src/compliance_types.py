"""
Data structures for compliance analysis system.

This module defines all data types used in the compliance analyzer,
including enums for requirement types, compliance statuses, and severity levels,
as well as dataclasses for requirements, mappings, issues, and reports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class RequirementType(Enum):
    """Type of legal requirement."""
    OBLIGATION = "obligation"       # Must do X
    PROHIBITION = "prohibition"     # Must not do X
    PERMISSION = "permission"       # May do X
    DEFINITION = "definition"       # X is defined as...
    CONDITION = "condition"         # If X then Y


class ComplianceStatus(Enum):
    """Compliance status of a clause."""
    COMPLIANT = "compliant"         # Fully complies
    CONFLICT = "conflict"           # Direct contradiction
    DEVIATION = "deviation"         # Differs but may be acceptable
    MISSING = "missing"             # Required provision absent
    UNCERTAIN = "uncertain"         # Cannot determine
    NOT_APPLICABLE = "not_applicable"  # Requirement doesn't apply


class IssueSeverity(Enum):
    """Severity level for compliance issues."""
    CRITICAL = "critical"   # Showstopper, contract invalid
    HIGH = "high"           # Serious risk, needs immediate fix
    MEDIUM = "medium"       # Moderate risk, should fix
    LOW = "low"             # Minor issue, consider fixing
    INFO = "info"           # Informational, no action needed


@dataclass
class LegalRequirement:
    """A requirement extracted from law."""
    requirement_id: str
    requirement_type: RequirementType

    # Source
    law_reference: str          # §89 odst. 2
    law_document_id: str
    law_chunk_id: str

    # Content
    requirement_text: str       # Full text of requirement
    requirement_summary: str    # Short description

    # Classification
    is_mandatory: bool          # vs. optional
    applies_to: List[str]       # ["contractor", "all_parties"]
    temporal_constraint: Optional[str] = None  # "within 30 days"

    # Metadata
    extraction_confidence: float = 1.0
    related_requirements: List[str] = field(default_factory=list)  # IDs


@dataclass
class ClauseMapping:
    """Mapping between contract clause and law provision."""
    # Contract side
    contract_chunk_id: str
    contract_reference: str     # Článek 5.2
    contract_text: str

    # Law side
    law_requirements: List[LegalRequirement]

    # Matching metadata
    match_score: float          # Confidence in mapping
    match_type: str             # explicit_ref | semantic | structural


@dataclass
class ComplianceIssue:
    """A compliance issue found during analysis."""
    issue_id: str
    status: ComplianceStatus
    severity: IssueSeverity

    # Contract clause
    contract_chunk_id: str
    contract_reference: str
    contract_text: str

    # Related law requirement(s)
    law_requirements: List[LegalRequirement]

    # Issue details
    issue_description: str      # What's wrong
    evidence: str               # Specific text causing issue
    reasoning: str              # Why it's an issue

    # Risk assessment
    risk_score: float           # 0.0 to 1.0
    risk_factors: Dict[str, float]  # {factor: weight}

    # Recommendations
    recommendations: List[str]  # Actions to remediate
    priority: int               # 1 (highest) to 5 (lowest)

    # Metadata
    detected_by: str            # Component that detected issue
    detection_confidence: float = 1.0
    verification_status: Optional[str] = None  # For human review


@dataclass
class ComplianceReport:
    """Final compliance analysis report."""
    # Metadata
    report_id: str
    generated_at: datetime
    contract_id: str
    law_ids: List[str]
    analysis_mode: str          # exhaustive | sample

    # Summary statistics
    total_clauses_checked: int
    total_requirements_checked: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_status: Dict[str, int]

    # Issues
    critical_issues: List[ComplianceIssue]
    high_issues: List[ComplianceIssue]
    medium_issues: List[ComplianceIssue]
    low_issues: List[ComplianceIssue]

    # Overall assessment
    overall_compliance_score: float  # 0.0 to 1.0
    is_compliant: bool              # Overall pass/fail
    risk_level: str                 # critical | high | medium | low

    # Recommendations summary
    top_recommendations: List[str]
    estimated_remediation_effort: str  # hours, simple/moderate/complex

    # Detailed results
    clause_mappings: List[ClauseMapping]
    all_issues: List[ComplianceIssue]

    # Analysis metadata
    processing_time: float
    llm_calls_made: int
    confidence_score: float         # Confidence in analysis
