"""Celery tasks for compliance checking."""
import logging
import json
import os
from datetime import datetime

from app.core.celery_app import celery_app
from app.tasks.indexing import CallbackTask
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=CallbackTask)
def compliance_check_task(
    self,
    contract_id: str,
    law_ids: list,
    mode: str = "exhaustive"
):
    """
    Run compliance check asynchronously.

    Args:
        contract_id: Contract document ID
        law_ids: List of law document IDs
        mode: 'exhaustive' | 'sample'

    Returns:
        Compliance report
    """
    logger.info(f"Starting compliance check for contract {contract_id} against laws {law_ids}")

    try:
        self.update_progress(0, 100, "Loading documents...")

        # In production, this would:
        # 1. Load contract and law chunks from index
        # 2. Run compliance analysis using ComplianceAnalyzer
        # 3. Generate detailed report
        # 4. Save report to storage

        # Mock implementation
        import time
        time.sleep(2)
        self.update_progress(20, 100, f"Loaded contract and {len(law_ids)} law documents")

        time.sleep(3)
        self.update_progress(50, 100, "Analyzing compliance...")

        time.sleep(3)
        self.update_progress(80, 100, "Generating report...")

        # Mock report
        report = {
            "report_id": f"report_{contract_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "overall_compliance_score": 0.75,
            "total_issues": 15,
            "critical_issues": [
                {
                    "issue_id": "CRIT-001",
                    "severity": "CRITICAL",
                    "status": "CONFLICT",
                    "contract_reference": "§3.1",
                    "law_references": ["§89 odst. 2 ZVZ"],
                    "issue_description": "Contract clause conflicts with mandatory procurement law provision",
                    "recommendations": [
                        "Revise contract clause to comply with §89 odst. 2",
                        "Consult with legal counsel"
                    ],
                    "risk_score": 0.95
                }
            ],
            "high_issues": [],
            "medium_issues": [],
            "low_issues": [],
            "top_recommendations": [
                "Address critical compliance issues immediately",
                "Review all contract clauses against procurement law"
            ]
        }

        # Save report
        reports_dir = os.path.join(settings.UPLOAD_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, f"{self.request.id}.json")
        with open(report_path, 'w') as f:
            json.dump({
                "status": "completed",
                "progress": 100,
                "report": report
            }, f, indent=2)

        self.update_progress(100, 100, "Report complete")

        logger.info(f"Compliance check complete for contract {contract_id}")

        return {
            "report_id": report["report_id"],
            "status": "completed",
            "report": report
        }

    except Exception as e:
        logger.error(f"Compliance check failed: {e}")

        # Save error report
        reports_dir = os.path.join(settings.UPLOAD_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, f"{self.request.id}.json")
        with open(report_path, 'w') as f:
            json.dump({
                "status": "failed",
                "progress": 0,
                "report": None,
                "error_message": str(e)
            }, f, indent=2)

        raise
