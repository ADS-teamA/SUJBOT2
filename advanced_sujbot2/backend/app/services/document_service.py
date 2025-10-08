"""Service for document management operations."""
import os
import hashlib
import json
from typing import Optional, List
from datetime import datetime
from fastapi import UploadFile
import aiofiles
import PyPDF2
from io import BytesIO

from app.core.config import settings
from app.models.document import (
    DocumentStatusResponse,
    DocumentList,
    DocumentListItem,
    DocumentMetadata
)


class DocumentService:
    """Service for handling document operations."""

    def __init__(self):
        """Initialize document service."""
        self.upload_dir = settings.UPLOAD_DIR
        self.index_dir = settings.INDEX_DIR

        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        # Metadata storage (in production, use database)
        self.metadata_file = os.path.join(self.upload_dir, "metadata.json")
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        """Ensure metadata file exists."""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)

    def _load_metadata(self) -> dict:
        """Load metadata from file."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _extract_quick_metadata(self, content: bytes, file_ext: str) -> dict:
        """
        Quickly extract basic metadata from file content.

        Args:
            content: File content bytes
            file_ext: File extension (with dot, e.g., '.pdf')

        Returns:
            Dictionary with page_count and word_count (if available)
        """
        metadata = {"page_count": 0, "word_count": 0}

        # Only process PDFs for now
        if file_ext.lower() == '.pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                metadata["page_count"] = len(pdf_reader.pages)
            except Exception as e:
                # If we can't read the PDF, leave at 0
                pass

        return metadata

    async def save_uploaded_file(
        self,
        file: UploadFile,
        document_type: str
    ) -> str:
        """
        Save uploaded file to disk.

        Args:
            file: Uploaded file
            document_type: Type of document

        Returns:
            Document ID (hash of file)
        """
        # Read file content
        content = await file.read()

        # Generate document ID from content hash
        document_id = hashlib.sha256(content).hexdigest()[:16]

        # Get file extension
        file_ext = os.path.splitext(file.filename)[1]

        # Save file
        file_path = os.path.join(self.upload_dir, f"{document_id}{file_ext}")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

        # Extract quick metadata (page count for PDFs)
        quick_metadata = self._extract_quick_metadata(content, file_ext)

        # Save metadata
        metadata = self._load_metadata()
        metadata[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "document_type": document_type,
            "status": "uploaded",
            "filesize": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "file_path": file_path,
            "progress": 0,
            "task_id": None,  # Will be updated after task creation
            "metadata": {
                "format": file_ext.lstrip('.'),
                "page_count": quick_metadata.get("page_count", 0),
                "word_count": quick_metadata.get("word_count", 0),
                "chunk_count": 0
            }
        }
        self._save_metadata(metadata)

        return document_id

    async def get_document_status(self, document_id: str) -> Optional[DocumentStatusResponse]:
        """
        Get document status and metadata.

        Args:
            document_id: Document ID

        Returns:
            Document status or None if not found
        """
        from app.core.celery_app import celery_app

        metadata = self._load_metadata()
        doc_meta = metadata.get(document_id)

        if not doc_meta:
            return None

        # Get current status and progress from metadata
        status = doc_meta.get("status", "uploaded")
        progress = doc_meta.get("progress", 0)

        # If task_id exists and status is processing, check Celery task state
        task_id = doc_meta.get("task_id")
        if task_id and status in ["uploaded", "processing"]:
            try:
                task_result = celery_app.AsyncResult(task_id)
                if task_result.state == "PROGRESS":
                    # Get progress from task state
                    task_info = task_result.info or {}
                    progress = task_info.get("percentage", progress)
                    status = "processing"
                elif task_result.state == "SUCCESS":
                    status = "indexed"
                    progress = 100
                elif task_result.state == "FAILURE":
                    status = "error"
            except Exception:
                # If we can't get task state, use metadata
                pass

        return DocumentStatusResponse(
            document_id=document_id,
            filename=doc_meta["filename"],
            status=status,
            progress=progress,
            metadata=DocumentMetadata(**doc_meta.get("metadata", {})),
            error_message=doc_meta.get("error_message")
        )

    async def list_documents(
        self,
        document_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> DocumentList:
        """
        List documents with optional filtering.

        Args:
            document_type: Filter by document type
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of documents
        """
        metadata = self._load_metadata()
        documents = []

        for doc_id, doc_meta in metadata.items():
            # Apply filters
            if document_type and doc_meta.get("document_type") != document_type:
                continue
            if status and doc_meta.get("status") != status:
                continue

            documents.append(DocumentListItem(
                document_id=doc_id,
                filename=doc_meta["filename"],
                document_type=doc_meta.get("document_type", "unknown"),
                status=doc_meta.get("status", "uploaded"),
                uploaded_at=datetime.fromisoformat(doc_meta["uploaded_at"]),
                metadata=DocumentMetadata(**doc_meta.get("metadata", {}))
            ))

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.uploaded_at, reverse=True)

        # Apply pagination
        total = len(documents)
        documents = documents[offset:offset + limit]

        return DocumentList(documents=documents, total=total)

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document and its index.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        metadata = self._load_metadata()
        doc_meta = metadata.get(document_id)

        if not doc_meta:
            return False

        # Delete file
        file_path = doc_meta.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Delete index directory
        index_path = os.path.join(self.index_dir, document_id)
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)

        # Remove from metadata
        del metadata[document_id]
        self._save_metadata(metadata)

        return True

    def update_task_id(self, document_id: str, task_id: str):
        """
        Update task ID for a document.

        Args:
            document_id: Document ID
            task_id: Celery task ID
        """
        metadata = self._load_metadata()
        if document_id in metadata:
            metadata[document_id]["task_id"] = task_id
            self._save_metadata(metadata)

    def update_document_status(
        self,
        document_id: str,
        status: str,
        progress: int = 0,
        metadata_update: Optional[dict] = None,
        error_message: Optional[str] = None
    ):
        """
        Update document status.

        Args:
            document_id: Document ID
            status: New status
            progress: Progress percentage
            metadata_update: Additional metadata to update
            error_message: Error message if status is error
        """
        metadata = self._load_metadata()

        if document_id not in metadata:
            return

        metadata[document_id]["status"] = status
        metadata[document_id]["progress"] = progress

        if error_message:
            metadata[document_id]["error_message"] = error_message

        if metadata_update:
            metadata[document_id]["metadata"].update(metadata_update)

        self._save_metadata(metadata)

    def get_document_path(self, document_id: str) -> Optional[str]:
        """
        Get file path for document.

        Args:
            document_id: Document ID

        Returns:
            File path or None if not found
        """
        metadata = self._load_metadata()
        doc_meta = metadata.get(document_id)

        if not doc_meta:
            return None

        return doc_meta.get("file_path")
