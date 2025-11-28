"""
Citation Metadata REST API

Resolves chunk_id to citation metadata for frontend display.
All endpoints require JWT authentication.
"""

import logging
import re
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.config import PDF_BASE_DIR
from backend.middleware.auth import get_current_user
from backend.routes.conversations import get_postgres_adapter
from backend.routes.documents import _find_pdf_for_document
from src.storage.postgres_adapter import PostgreSQLStorageAdapter

router = APIRouter(prefix="/citations", tags=["citations"])
logger = logging.getLogger(__name__)

# Valid chunk_id pattern: alphanumeric, underscore, hyphen, space, slash, dot
# Examples: "157/2025 Sb._L3_1", "BZ_VR1_L3_c5_sec_3"
CHUNK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s/.]+$")


# ============================================================================
# Pydantic Models
# ============================================================================

class CitationMetadata(BaseModel):
    """Response model for citation metadata."""
    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    document_id: str = Field(..., min_length=1, description="Document identifier (matches PDF filename)")
    document_name: str = Field(..., min_length=1, description="Human-readable document name")
    section_title: Optional[str] = Field(None, description="Section title")
    section_path: Optional[str] = Field(None, description="Full section path/breadcrumb")
    hierarchical_path: Optional[str] = Field(None, description="Full hierarchical path including document")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in PDF (1-indexed)")
    pdf_available: bool = Field(..., description="Whether PDF file exists on server")
    content: Optional[str] = Field(None, description="Chunk text content for PDF highlighting")


class BatchCitationRequest(BaseModel):
    """Request model for batch citation lookup."""
    chunk_ids: List[str] = Field(..., min_length=1, max_length=50, description="List of chunk IDs to resolve")


# ============================================================================
# Helper Functions
# ============================================================================

def _format_document_name(document_id: str) -> str:
    """Format document_id to human-readable name."""
    # Replace underscores with spaces for better readability
    # E.g., "Sb_2016_263_2024-01-01_IZ" -> "Sb 2016 263 2024-01-01 IZ"
    return document_id.replace("_", " ")


def _check_pdf_available(document_id: str) -> bool:
    """Check if PDF file exists for document using pattern matching."""
    return _find_pdf_for_document(document_id) is not None


async def _fetch_chunk_metadata(
    adapter: PostgreSQLStorageAdapter,
    chunk_id: str
) -> Optional[Dict]:
    """
    Fetch chunk metadata from database.

    Searches all vector layers (3, 2, 1) to find the chunk.
    Note: layer1 has different schema (title instead of section_title, no section_path).

    Raises:
        HTTPException: If database query fails
    """
    try:
        async with adapter.pool.acquire() as conn:
            # Try layer3 and layer2 first (have section_title, section_path)
            for layer in [3, 2]:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        chunk_id,
                        document_id,
                        section_title,
                        section_path,
                        hierarchical_path,
                        page_number,
                        content
                    FROM vectors.layer{layer}
                    WHERE chunk_id = $1
                    LIMIT 1
                    """,
                    chunk_id
                )
                if row:
                    return dict(row)

            # Try layer1 (different schema: title instead of section_title, no section_path)
            row = await conn.fetchrow(
                """
                SELECT
                    chunk_id,
                    document_id,
                    title AS section_title,
                    NULL AS section_path,
                    hierarchical_path,
                    page_number,
                    content
                FROM vectors.layer1
                WHERE chunk_id = $1
                LIMIT 1
                """,
                chunk_id
            )
            if row:
                return dict(row)

        return None
    except Exception as e:
        logger.error(f"Database query failed for chunk_id={chunk_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable. Please try again."
        )


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/{chunk_id}", response_model=CitationMetadata)
async def get_citation_metadata(
    chunk_id: str,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
) -> CitationMetadata:
    """
    Resolve chunk_id to citation metadata.

    Args:
        chunk_id: Chunk identifier (e.g., "BZ_VR1_L3_c5_sec_3")

    Returns:
        CitationMetadata with document info, section, page, and PDF availability

    Raises:
        HTTPException 400: Invalid chunk_id format
        HTTPException 404: Chunk not found in database
    """
    # Validate chunk_id format
    if not CHUNK_ID_PATTERN.match(chunk_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk_id format. Use only alphanumeric characters, underscores, and hyphens."
        )

    # Fetch from database
    row = await _fetch_chunk_metadata(adapter, chunk_id)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk not found: {chunk_id}"
        )

    document_id = row["document_id"]

    return CitationMetadata(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=_format_document_name(document_id),
        section_title=row.get("section_title"),
        section_path=row.get("section_path"),
        hierarchical_path=row.get("hierarchical_path"),
        page_number=row.get("page_number"),
        pdf_available=_check_pdf_available(document_id),
        content=row.get("content"),
    )


@router.post("/batch", response_model=List[CitationMetadata])
async def get_citations_batch(
    request: BatchCitationRequest,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
) -> List[CitationMetadata]:
    """
    Resolve multiple chunk_ids in one request.

    This is more efficient than making multiple individual requests
    when a message contains many citations.

    Args:
        request: BatchCitationRequest with list of chunk_ids (max 50)

    Returns:
        List of CitationMetadata for found chunks.
        Chunks that don't exist are silently skipped (no error).

    Note:
        Order of results may not match order of input chunk_ids.
        Frontend should match by chunk_id field.
    """
    results = []

    for chunk_id in request.chunk_ids:
        # Skip invalid chunk_ids silently
        if not CHUNK_ID_PATTERN.match(chunk_id):
            logger.warning(f"Skipping invalid chunk_id in batch: {chunk_id}")
            continue

        row = await _fetch_chunk_metadata(adapter, chunk_id)

        if row:
            document_id = row["document_id"]
            # Convert empty strings to None for optional fields
            section_title = row.get("section_title") or None
            section_path = row.get("section_path") or None
            hierarchical_path = row.get("hierarchical_path") or None
            results.append(CitationMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=_format_document_name(document_id),
                section_title=section_title,
                section_path=section_path,
                hierarchical_path=hierarchical_path,
                page_number=row.get("page_number"),
                pdf_available=_check_pdf_available(document_id),
                content=row.get("content"),
            ))

    logger.info(f"Batch citation lookup: {len(request.chunk_ids)} requested, {len(results)} found")
    return results
