"""
Compliance Reporter for generating analysis reports.

Generates structured compliance reports in JSON and other formats.
"""

import logging
import json
import dataclasses
from datetime import datetime
from typing import List, Dict
import numpy as np

from .compliance_types import (
    ComplianceReport,
    ComplianceIssue,
    ClauseMapping,
    LegalRequirement,
    IssueSeverity,
    ComplianceStatus
)


class ComplianceReporter:
    """Generate final compliance report."""

    def __init__(self, config: dict = None):
        """
        Initialize ComplianceReporter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self,
        contract_id: str,
        law_ids: List[str],
        mappings: List[ClauseMapping],
        issues: List[ComplianceIssue],
        requirements: List[LegalRequirement],
        processing_time: float,
        llm_calls: int,
        mode: str = "exhaustive"
    ) -> ComplianceReport:
        """
        Generate compliance report from analysis results.

        Args:
            contract_id: Contract document ID
            law_ids: List of law document IDs
            mappings: List of clause-to-law mappings
            issues: List of compliance issues found
            requirements: List of all legal requirements
            processing_time: Total processing time in seconds
            llm_calls: Number of LLM API calls made
            mode: Analysis mode (exhaustive | sample)

        Returns:
            ComplianceReport object
        """
        # Group issues by severity
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high = [i for i in issues if i.severity == IssueSeverity.HIGH]
        medium = [i for i in issues if i.severity == IssueSeverity.MEDIUM]
        low = [i for i in issues if i.severity == IssueSeverity.LOW]

        # Compute statistics
        issues_by_severity = {
            IssueSeverity.CRITICAL.value: len(critical),
            IssueSeverity.HIGH.value: len(high),
            IssueSeverity.MEDIUM.value: len(medium),
            IssueSeverity.LOW.value: len(low)
        }

        issues_by_status = {}
        for status in ComplianceStatus:
            count = len([i for i in issues if i.status == status])
            if count > 0:
                issues_by_status[status.value] = count

        # Overall compliance score (1.0 = perfect, 0.0 = many critical issues)
        compliance_score = self._compute_compliance_score(issues, requirements)

        # Overall pass/fail
        is_compliant = len(critical) == 0 and len(high) <= 2

        # Risk level
        if len(critical) > 0:
            risk_level = "critical"
        elif len(high) > 3:
            risk_level = "high"
        elif len(medium) > 5:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Top recommendations (from critical/high issues)
        top_issues = critical + high
        top_recommendations = []
        for issue in top_issues[:5]:  # Top 5 issues
            if issue.recommendations:
                rec_text = f"{issue.contract_reference}: {issue.recommendations[0]}"
                top_recommendations.append(rec_text)

        # Confidence score
        avg_confidence = float(np.mean([i.detection_confidence for i in issues])) if issues else 1.0

        report = ComplianceReport(
            report_id=f"report_{contract_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            contract_id=contract_id,
            law_ids=law_ids,
            analysis_mode=mode,
            total_clauses_checked=len(mappings),
            total_requirements_checked=len(requirements),
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues_by_status=issues_by_status,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            overall_compliance_score=compliance_score,
            is_compliant=is_compliant,
            risk_level=risk_level,
            top_recommendations=top_recommendations,
            estimated_remediation_effort=self._estimate_effort(issues),
            clause_mappings=mappings,
            all_issues=issues,
            processing_time=processing_time,
            llm_calls_made=llm_calls,
            confidence_score=avg_confidence
        )

        self.logger.info(f"Generated compliance report: {report.report_id}")
        return report

    def _compute_compliance_score(
        self,
        issues: List[ComplianceIssue],
        requirements: List[LegalRequirement]
    ) -> float:
        """
        Compute overall compliance score.
        1.0 = fully compliant, 0.0 = many critical issues.

        Args:
            issues: List of compliance issues
            requirements: List of all legal requirements

        Returns:
            Compliance score (0.0 to 1.0)
        """
        if not requirements:
            return 1.0

        # Penalty for each issue type
        penalties = {
            IssueSeverity.CRITICAL: 0.2,
            IssueSeverity.HIGH: 0.1,
            IssueSeverity.MEDIUM: 0.05,
            IssueSeverity.LOW: 0.02
        }

        total_penalty = sum(penalties.get(i.severity, 0.0) for i in issues)

        # Score = 1.0 - total_penalty (clamped to [0, 1])
        score = max(0.0, 1.0 - total_penalty)

        return float(score)

    def _estimate_effort(self, issues: List[ComplianceIssue]) -> str:
        """
        Estimate remediation effort.

        Args:
            issues: List of compliance issues

        Returns:
            Effort estimate as string
        """
        # Count by severity
        critical = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high = len([i for i in issues if i.severity == IssueSeverity.HIGH])
        medium = len([i for i in issues if i.severity == IssueSeverity.MEDIUM])

        # Rough estimate (hours)
        hours = critical * 4 + high * 2 + medium * 1

        if hours > 40:
            return "complex (40+ hours)"
        elif hours > 16:
            return "moderate (16-40 hours)"
        elif hours > 4:
            return "simple (4-16 hours)"
        else:
            return "minimal (<4 hours)"

    def export_json(self, report: ComplianceReport, output_path: str):
        """
        Export report to JSON file.

        Args:
            report: ComplianceReport object
            output_path: Path to output JSON file
        """
        def serialize_obj(obj):
            """Custom serializer for non-serializable objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if isinstance(obj, (IssueSeverity, ComplianceStatus)):
                return obj.value
            return str(obj)

        report_dict = dataclasses.asdict(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=serialize_obj)

        self.logger.info(f"Exported report to JSON: {output_path}")

    def export_markdown(self, report: ComplianceReport, output_path: str):
        """
        Export report to Markdown file.

        Args:
            report: ComplianceReport object
            output_path: Path to output Markdown file
        """
        markdown = f"""# Compliance Report

**Contract**: {report.contract_id}
**Laws**: {', '.join(report.law_ids)}
**Generated**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Mode**: {report.analysis_mode}

## Summary

- **Compliance Score**: {report.overall_compliance_score:.1%}
- **Risk Level**: {report.risk_level.upper()}
- **Is Compliant**: {"✅ Yes" if report.is_compliant else "❌ No"}
- **Processing Time**: {report.processing_time:.1f}s
- **LLM Calls**: {report.llm_calls_made}

## Issues Found

| Severity | Count |
|----------|-------|
| **Critical** | {len(report.critical_issues)} |
| **High** | {len(report.high_issues)} |
| **Medium** | {len(report.medium_issues)} |
| **Low** | {len(report.low_issues)} |
| **TOTAL** | {report.total_issues} |

## Analysis Statistics

- **Clauses Checked**: {report.total_clauses_checked}
- **Requirements Checked**: {report.total_requirements_checked}
- **Confidence**: {report.confidence_score:.1%}
- **Estimated Remediation Effort**: {report.estimated_remediation_effort}

## Critical Issues

"""
        for i, issue in enumerate(report.critical_issues[:10], 1):
            markdown += f"""### {i}. {issue.contract_reference}

**Status**: {issue.status.value}
**Description**: {issue.issue_description}

**Evidence**: {issue.evidence[:200]}...

**Recommendations**:
{chr(10).join([f"- {rec}" for rec in issue.recommendations[:3]])}

---

"""

        markdown += f"""## Top Recommendations

{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(report.top_recommendations[:10])])}

## Detailed Statistics

**Issues by Status**:
{chr(10).join([f"- {status}: {count}" for status, count in report.issues_by_status.items()])}

---

*Report generated by Compliance Analyzer*
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        self.logger.info(f"Exported report to Markdown: {output_path}")
