"""Dependency injection for FastAPI endpoints."""
from typing import Optional
from functools import lru_cache
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from app.services.document_service import DocumentService
from app.services.compliance_service import ComplianceService
from app.services.chat_service import ChatService


# Singleton instances
_document_service: Optional[DocumentService] = None
_compliance_service: Optional[ComplianceService] = None
_chat_service: Optional[ChatService] = None


def get_document_service() -> DocumentService:
    """Get or create document service singleton."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


def get_compliance_service() -> ComplianceService:
    """Get or create compliance service singleton."""
    global _compliance_service
    if _compliance_service is None:
        _compliance_service = ComplianceService()
    return _compliance_service


def get_chat_service() -> ChatService:
    """Get or create chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
