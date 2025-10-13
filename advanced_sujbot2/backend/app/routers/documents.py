"""Document management endpoints."""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional

from app.models.document import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentList
)
from app.services.document_service import DocumentService
from app.core.dependencies import get_document_service
from app.core.config import settings
from app.tasks.indexing import index_document_task

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="contract"),
    service: DocumentService = Depends(get_document_service)
):
    """
    Upload and index a document.

    - **file**: Document file (PDF, DOCX, TXT, etc.)
    - **document_type**: Type of document (contract, law_code, regulation)

    Returns document metadata and starts background indexing task.
    """
    # Map frontend document types to database types
    # Frontend uses 'law' for simplicity, but database uses 'law_code'
    type_mapping = {
        'law': 'law_code',
        'contract': 'contract',
        'regulation': 'regulation'
    }
    document_type = type_mapping.get(document_type, document_type)

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file extension
    file_ext = '.' + file.filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB)"
        )

    # Save file first
    document_id = await service.save_uploaded_file(file, document_type)

    # Start indexing task
    task = index_document_task.delay(document_id, document_type)

    # Update metadata with task_id
    service.update_task_id(document_id, task.id)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        filesize=file_size,
        document_type=document_type,
        status="uploaded",
        indexing_task_id=task.id
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get document processing status and metadata."""
    status = await service.get_document_status(document_id)

    if not status:
        raise HTTPException(status_code=404, detail="Document not found")

    return status


@router.get("", response_model=DocumentList)
async def list_documents(
    type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    service: DocumentService = Depends(get_document_service)
):
    """List all documents with optional filters."""
    documents = await service.list_documents(
        document_type=type,
        status=status,
        limit=limit,
        offset=offset
    )

    return documents


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a document and its index."""
    success = await service.delete_document(document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return None
