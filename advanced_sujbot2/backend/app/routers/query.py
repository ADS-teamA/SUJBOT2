"""Query/chat REST endpoints."""
from fastapi import APIRouter, Depends

from app.models.query import QueryRequest, QueryResponse
from app.services.chat_service import ChatService
from app.core.dependencies import get_chat_service

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Query documents and get answer with sources.

    - **query**: Question to ask
    - **document_ids**: List of document IDs to search
    - **language**: Query language (cs or en)

    Returns answer with source citations.
    """
    result = await service.process_query(
        query=request.query,
        document_ids=request.document_ids,
        language=request.language
    )

    return result
