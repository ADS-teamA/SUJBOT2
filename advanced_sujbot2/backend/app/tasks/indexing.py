"""Celery tasks for document indexing."""
from celery import Task
import logging
from datetime import datetime

from app.core.celery_app import celery_app
from app.services.document_service import DocumentService
from app.services.rag_pipeline import get_rag_pipeline
from app.rag.exceptions import DocumentProcessingError, IndexingError

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Custom task with progress callbacks."""

    def update_progress(self, current, total, message=""):
        """
        Update task progress.

        Args:
            current: Current progress value
            total: Total progress value
            message: Progress message
        """
        # Only update state if we have a valid task_id (bound task context)
        if hasattr(self, 'request') and self.request.id:
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": current,
                    "total": total,
                    "percentage": int((current / total) * 100),
                    "message": message
                }
            )
        else:
            # Log progress without Celery state update (useful for debugging)
            logger.debug(f"Progress: {current}/{total} ({int((current / total) * 100)}%) - {message}")


@celery_app.task(bind=True, base=CallbackTask, acks_late=True, reject_on_worker_lost=True)
def index_document_task(self, document_id: str, document_type: str):
    """
    Index a document asynchronously using the full RAG pipeline.

    This task is interruptible - it checks for revocation signals and
    gracefully terminates if the task is cancelled.

    Args:
        document_id: Document ID
        document_type: 'contract' | 'law_code' | 'regulation'

    Returns:
        Document metadata with indexing info

    Raises:
        Terminated: If task is revoked during execution
    """
    from celery.exceptions import Terminated

    logger.info(f"Starting RAG indexing for document {document_id}")
    document_service = DocumentService()
    rag_pipeline = get_rag_pipeline()

    # Flag to check if task was cancelled
    def check_if_revoked():
        """Check if task has been revoked and raise Terminated if so."""
        if self.request.id:
            from celery.result import AsyncResult
            task_result = AsyncResult(self.request.id, app=celery_app)
            # Check if task was revoked
            if task_result.state == 'REVOKED':
                logger.warning(f"Task {self.request.id} was revoked, terminating gracefully")
                raise Terminated("Task was cancelled by user")

    try:
        # Check for cancellation before starting
        check_if_revoked()

        # Update initial progress
        self.update_progress(0, 100, "Starting indexing...")
        document_service.update_document_status(document_id, "processing", 0)

        # Get document path
        doc_path = document_service.get_document_path(document_id)
        if not doc_path:
            raise ValueError(f"Document {document_id} not found")

        # Define progress callback that syncs with Celery and document service
        # AND checks for task cancellation at each progress update
        def progress_callback(current: int, total: int, message: str):
            # Check if task was cancelled
            check_if_revoked()

            # Update progress
            self.update_progress(current, total, message)
            document_service.update_document_status(
                document_id,
                "processing",
                int((current / total) * 100)
            )

        # Execute full RAG indexing pipeline
        # This includes:
        # 1. DocumentReader - PDF/DOCX parsing
        # 2. LegalChunker - Hierarchical semantic chunking
        # 3. EmbeddingService - BGE-M3 embeddings
        # 4. IndexingPipeline - FAISS indexing
        # 5. KnowledgeGraph - Legal graph construction (for laws)
        metadata = rag_pipeline.index_document(
            document_path=doc_path,
            document_id=document_id,
            document_type=document_type,
            progress_callback=progress_callback
        )

        # Update document status with comprehensive metadata
        document_service.update_document_status(
            document_id,
            "indexed",
            100,
            metadata_update=metadata
        )

        logger.info(
            f"Successfully indexed document {document_id}: "
            f"{metadata.get('chunk_count', 0)} chunks, "
            f"{metadata.get('page_count', 0)} pages"
        )

        return {
            "document_id": document_id,
            "status": "indexed",
            "metadata": metadata
        }

    except Terminated as e:
        # Task was cancelled - this is expected behavior
        logger.info(f"Task cancelled for document {document_id}: {e}")
        document_service.update_document_status(
            document_id,
            "cancelled",
            0,
            error_message="Document processing was cancelled"
        )
        # Don't re-raise - task was cancelled gracefully
        return {
            "document_id": document_id,
            "status": "cancelled",
            "message": "Task was cancelled by user"
        }

    except (DocumentProcessingError, IndexingError) as e:
        logger.error(f"RAG indexing failed for document {document_id}: {e}")
        document_service.update_document_status(
            document_id,
            "error",
            0,
            error_message=f"Indexing error: {str(e)}"
        )
        raise

    except Exception as e:
        logger.error(f"Unexpected error indexing document {document_id}: {e}", exc_info=True)
        document_service.update_document_status(
            document_id,
            "error",
            0,
            error_message=f"Unexpected error: {str(e)}"
        )
        raise
