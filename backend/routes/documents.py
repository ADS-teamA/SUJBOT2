"""
Document REST API

Serves PDF files from the data/ folder with security checks.
All endpoints require JWT authentication.
"""

import logging
import re
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from backend.middleware.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Allowed PDF directory (relative to project root)
# Resolved at module load to absolute path
PDF_BASE_DIR = Path(__file__).parent.parent.parent / "data"
PDF_BASE_DIR = PDF_BASE_DIR.resolve()

# Valid document_id pattern: alphanumeric, underscore, hyphen only
DOCUMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@router.get("/{document_id}/pdf")
async def get_pdf(
    document_id: str,
    user: Dict = Depends(get_current_user)
) -> FileResponse:
    """
    Serve PDF file for a document.

    Args:
        document_id: Document identifier (e.g., "BZ_VR1", "Sb_2016_263_2024-01-01_IZ")

    Returns:
        PDF file stream with inline disposition for browser viewing

    Security:
        - Authentication required (JWT)
        - Strict document_id validation (alphanumeric, underscore, hyphen only)
        - Path traversal prevention via resolve() + prefix check
        - Only files from data/ directory allowed
    """
    # Validate document_id format
    if not DOCUMENT_ID_PATTERN.match(document_id):
        logger.warning(f"Invalid document_id format: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document identifier format. Use only alphanumeric characters, underscores, and hyphens."
        )

    # Construct and validate path
    pdf_filename = f"{document_id}.pdf"
    pdf_path = (PDF_BASE_DIR / pdf_filename).resolve()

    # Security: Verify path is under allowed directory (prevent path traversal)
    if not str(pdf_path).startswith(str(PDF_BASE_DIR)):
        logger.warning(f"Path traversal attempt blocked: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Check file exists
    if not pdf_path.is_file():
        logger.info(f"PDF not found: {document_id} (path: {pdf_path})")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found for document: {document_id}"
        )

    logger.info(f"Serving PDF: {document_id} to user {user.get('id', 'unknown')}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_filename,
        headers={
            "Content-Disposition": f"inline; filename={pdf_filename}",
            "Cache-Control": "public, max-age=3600",  # 1 hour cache
        }
    )
