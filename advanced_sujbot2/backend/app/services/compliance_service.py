"""Service for compliance checking operations."""
from typing import Optional
import json
import os

from app.core.config import settings
from app.models.compliance import ComplianceReportResponse, ComplianceReport
from app.services.document_service import DocumentService


class ComplianceService:
    """Service for compliance checking operations."""

    def __init__(self):
        """Initialize compliance service."""
        self.document_service = DocumentService()
        self.reports_dir = os.path.join(settings.UPLOAD_DIR, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    async def document_exists(self, document_id: str) -> bool:
        """
        Check if document exists.

        Args:
            document_id: Document ID

        Returns:
            True if exists, False otherwise
        """
        status = await self.document_service.get_document_status(document_id)
        return status is not None

    async def get_report(self, task_id: str) -> Optional[ComplianceReportResponse]:
        """
        Get compliance report by task ID.

        Args:
            task_id: Celery task ID

        Returns:
            Report response or None if not found
        """
        # In production, query Celery result backend
        # For now, load from file
        report_path = os.path.join(self.reports_dir, f"{task_id}.json")

        if not os.path.exists(report_path):
            # Check if task is still running (mock)
            return ComplianceReportResponse(
                task_id=task_id,
                status="processing",
                progress=50,
                report=None
            )

        with open(report_path, 'r') as f:
            data = json.load(f)

        return ComplianceReportResponse(
            task_id=task_id,
            status=data["status"],
            progress=data["progress"],
            report=ComplianceReport(**data["report"]) if data.get("report") else None,
            error_message=data.get("error_message")
        )

    async def export_report(self, task_id: str, format: str) -> Optional[bytes]:
        """
        Export compliance report in specified format.

        Args:
            task_id: Task ID
            format: Export format (json, markdown, pdf)

        Returns:
            Report content as bytes or None if not found
        """
        report = await self.get_report(task_id)

        if not report or not report.report:
            return None

        if format == "json":
            return json.dumps(report.model_dump(), indent=2, default=str).encode()

        elif format == "markdown":
            # Generate markdown report
            md = f"""# Compliance Report

## Summary
- **Overall Compliance Score**: {report.report.overall_compliance_score:.2%}
- **Total Issues**: {report.report.total_issues}

## Critical Issues ({len(report.report.critical_issues)})
"""
            for issue in report.report.critical_issues:
                md += f"\n### {issue.issue_id}\n"
                md += f"- **Severity**: {issue.severity}\n"
                md += f"- **Status**: {issue.status}\n"
                md += f"- **Description**: {issue.issue_description}\n"

            return md.encode()

        elif format == "pdf":
            # PDF generation would require additional library (reportlab, weasyprint)
            # For now, return placeholder
            return b"PDF generation not implemented yet"

        return None
