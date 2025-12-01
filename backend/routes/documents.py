"""
Document REST API

Serves PDF files from the data/ folder with security checks.
All endpoints require JWT authentication.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from backend.config import PDF_BASE_DIR
from backend.middleware.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Valid document_id patterns
# Direct format: alphanumeric, underscore, hyphen (e.g., "BZ_VR1")
DIRECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
# Legal format: "number/year Sb." (e.g., "157/2025 Sb.")
LEGAL_ID_PATTERN = re.compile(r"^(\d+)/(\d{4})\s*Sb\.$")


def _find_pdf_for_document(document_id: str) -> Optional[Path]:
    """
    Find PDF file for a document_id.

    Handles two formats:
    1. Direct match: "BZ_VR1" → "BZ_VR1.pdf"
    2. Legal format: "157/2025 Sb." → searches for "Sb_{year}_{number}_*.pdf"

    Returns:
        Path to PDF file if found, None otherwise
    """
    # Try direct match first (e.g., "BZ_VR1" → "BZ_VR1.pdf")
    if DIRECT_ID_PATTERN.match(document_id):
        direct_path = PDF_BASE_DIR / f"{document_id}.pdf"
        if direct_path.is_file():
            return direct_path

    # Try legal format (e.g., "157/2025 Sb." → "Sb_2025_157_*.pdf")
    legal_match = LEGAL_ID_PATTERN.match(document_id)
    if legal_match:
        number, year = legal_match.groups()
        # Search for matching PDF with pattern Sb_{year}_{number}_*.pdf
        pattern = f"Sb_{year}_{number}_*.pdf"
        matches = list(PDF_BASE_DIR.glob(pattern))
        if matches:
            # Return first match (should be unique per document)
            return matches[0]

    return None


@router.get("/{document_id:path}/pdf")
async def get_pdf(
    document_id: str,
    user: Dict = Depends(get_current_user)
) -> FileResponse:
    """
    Serve PDF file for a document.

    Args:
        document_id: Document identifier in one of these formats:
            - Direct: "BZ_VR1" → "BZ_VR1.pdf"
            - Legal: "157/2025 Sb." → searches for "Sb_2025_157_*.pdf"

    Returns:
        PDF file stream with inline disposition for browser viewing

    Security:
        - Authentication required (JWT)
        - Pattern-based validation (direct or legal format)
        - Path traversal prevention via resolve() + prefix check
        - Only files from data/ directory allowed
    """
    from urllib.parse import unquote
    document_id = unquote(document_id)

    # Find PDF using pattern matching
    pdf_path = _find_pdf_for_document(document_id)

    if pdf_path is None:
        logger.info(f"PDF not found for document_id: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found for document: {document_id}"
        )

    # Resolve to absolute path
    pdf_path = pdf_path.resolve()

    # Security: Verify path is under allowed directory (prevent path traversal)
    if not str(pdf_path).startswith(str(PDF_BASE_DIR.resolve())):
        logger.warning(f"Path traversal attempt blocked: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    pdf_filename = pdf_path.name
    logger.info(f"Serving PDF: {pdf_filename} (doc: {document_id}) to user {user.get('id', 'unknown')}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_filename,
        headers={
            "Content-Disposition": f"inline; filename={pdf_filename}",
            "Cache-Control": "public, max-age=3600",  # 1 hour cache
        }
    )
