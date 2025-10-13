"""Compliance checking endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from app.models.compliance import (
    ComplianceCheckRequest,
    ComplianceCheckResponse,
    ComplianceReportResponse
)
from app.services.compliance_service import ComplianceService
from app.core.dependencies import get_compliance_service
from app.tasks.compliance import compliance_check_task

router = APIRouter()


@router.post("/check", response_model=ComplianceCheckResponse, status_code=202)
async def start_compliance_check(
    request: ComplianceCheckRequest,
    service: ComplianceService = Depends(get_compliance_service)
):
    """
    Start compliance check task.

    Compares contract against provided laws and generates compliance report.
    This is an async operation that returns a task ID.
    """
    # Validate documents exist
    contract_exists = await service.document_exists(request.contract_document_id)
    if not contract_exists:
        raise HTTPException(status_code=404, detail="Contract document not found")

    for law_id in request.law_document_ids:
        law_exists = await service.document_exists(law_id)
        if not law_exists:
            raise HTTPException(status_code=404, detail=f"Law document {law_id} not found")

    # Start compliance check task
    task = compliance_check_task.delay(
        contract_id=request.contract_document_id,
        law_ids=request.law_document_ids,
        mode=request.mode
    )

    return ComplianceCheckResponse(
        task_id=task.id,
        status="processing",
        estimated_duration_seconds=180  # Estimated based on document sizes
    )


@router.get("/reports/{task_id}", response_model=ComplianceReportResponse)
async def get_compliance_report(
    task_id: str,
    service: ComplianceService = Depends(get_compliance_service)
):
    """Get compliance check results."""
    report = await service.get_report(task_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report


@router.get("/reports/{task_id}/download")
async def download_compliance_report(
    task_id: str,
    format: str = "json",
    service: ComplianceService = Depends(get_compliance_service)
):
    """Download compliance report in specified format."""
    if format not in ["json", "markdown", "pdf"]:
        raise HTTPException(status_code=400, detail="Invalid format. Allowed: json, markdown, pdf")

    report_content = await service.export_report(task_id, format)

    if not report_content:
        raise HTTPException(status_code=404, detail="Report not found")

    media_types = {
        "json": "application/json",
        "markdown": "text/markdown",
        "pdf": "application/pdf"
    }

    return Response(
        content=report_content,
        media_type=media_types[format],
        headers={
            "Content-Disposition": f"attachment; filename=compliance_report_{task_id}.{format}"
        }
    )
