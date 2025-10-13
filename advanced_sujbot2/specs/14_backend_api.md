# 14. Backend API Specification

## 1. Purpose

**Objective**: Implement a production-grade FastAPI backend with REST endpoints for document management, WebSocket support for real-time chat, and async task processing for heavy operations.

**Why Backend API?**
- Expose compliance checking engine to frontend
- Handle file uploads efficiently (multipart/form-data)
- Real-time streaming chat responses via WebSocket
- Async background processing for indexing and compliance checks
- Scalable architecture with task queues
- API documentation and validation (OpenAPI/Swagger)

**Key Capabilities**:
1. **REST API** - CRUD operations for documents and compliance reports
2. **WebSocket** - Real-time bidirectional communication for chat
3. **Async Tasks** - Background processing with Celery + Redis
4. **File Upload** - Streaming uploads with progress tracking
5. **Authentication** - JWT-based auth (future)
6. **CORS** - Frontend integration
7. **API Docs** - Auto-generated OpenAPI specification

---

## 2. API Architecture

### 2.1 Overall Architecture

```
┌──────────────────────────────────────────────┐
│         Frontend (React)                     │
└────────────┬────────────────┬────────────────┘
             │                │
    REST API │                │ WebSocket
             ▼                ▼
┌──────────────────────────────────────────────┐
│         FastAPI Application                  │
│  ┌────────────────┬──────────────────────┐  │
│  │  REST Routers  │  WebSocket Handler   │  │
│  └────────┬───────┴──────────┬───────────┘  │
│           │                  │               │
│           ▼                  ▼               │
│  ┌─────────────────────────────────────┐    │
│  │  Business Logic Layer               │    │
│  │  - ComplianceChecker                │    │
│  │  - DocumentManager                  │    │
│  │  - QueryProcessor                   │    │
│  └─────────────┬───────────────────────┘    │
│                │                             │
│                ▼                             │
│  ┌─────────────────────────────────────┐    │
│  │  Task Queue (Celery)                │    │
│  │  - indexing_task                    │    │
│  │  - compliance_check_task            │    │
│  └─────────────┬───────────────────────┘    │
└────────────────┼─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│         Data Layer                           │
│  ┌──────────┬───────────┬──────────────┐    │
│  │ Redis    │ FAISS     │ File Storage │    │
│  │ (Queue)  │ (Indices) │ (Uploads)    │    │
│  └──────────┴───────────┴──────────────┘    │
└──────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.104+ | REST + WebSocket endpoints |
| **ASGI Server** | Uvicorn | High-performance async server |
| **Task Queue** | Celery 5.3+ | Background job processing |
| **Message Broker** | Redis 7.2+ | Task queue + caching |
| **Validation** | Pydantic v2 | Request/response validation |
| **Auth (future)** | python-jose | JWT token management |
| **CORS** | fastapi-cors | Cross-origin support |
| **WebSocket** | FastAPI WebSocket | Real-time chat |
| **File Storage** | Local filesystem | Document uploads (S3 future) |

---

## 3. API Endpoints

### 3.1 REST API Specification

#### Document Management

**Upload Document**
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Parameters:
- file: File (required)
- document_type: string (required) - "contract" | "law_code" | "regulation"
- language: string (optional) - "cs" | "en"

Response: 201 Created
{
  "document_id": "abc123def456",
  "filename": "smlouva.pdf",
  "filesize": 1024000,
  "document_type": "contract",
  "status": "uploaded",
  "indexing_task_id": "task_xyz789"
}
```

**Get Document Status**
```http
GET /api/v1/documents/{document_id}/status

Response: 200 OK
{
  "document_id": "abc123def456",
  "filename": "smlouva.pdf",
  "status": "indexed" | "processing" | "error",
  "progress": 100,
  "metadata": {
    "page_count": 1234,
    "word_count": 567890,
    "format": "pdf",
    "indexed_at": "2025-10-08T10:30:00Z"
  },
  "error_message": null
}
```

**List Documents**
```http
GET /api/v1/documents?type=contract&status=indexed

Response: 200 OK
{
  "documents": [
    {
      "document_id": "abc123",
      "filename": "smlouva.pdf",
      "document_type": "contract",
      "status": "indexed",
      "uploaded_at": "2025-10-08T10:00:00Z",
      "metadata": { ... }
    }
  ],
  "total": 1
}
```

**Delete Document**
```http
DELETE /api/v1/documents/{document_id}

Response: 204 No Content
```

#### Compliance Checking

**Start Compliance Check**
```http
POST /api/v1/compliance/check

Body:
{
  "contract_document_id": "abc123",
  "law_document_ids": ["def456", "ghi789"],
  "mode": "exhaustive" | "sample"
}

Response: 202 Accepted
{
  "task_id": "task_compliance_xyz",
  "status": "processing",
  "estimated_duration_seconds": 180
}
```

**Get Compliance Report**
```http
GET /api/v1/compliance/reports/{task_id}

Response: 200 OK
{
  "task_id": "task_compliance_xyz",
  "status": "completed" | "processing" | "failed",
  "progress": 100,
  "report": {
    "report_id": "report_123",
    "overall_compliance_score": 0.75,
    "total_issues": 15,
    "critical_issues": [...],
    "high_issues": [...],
    "recommendations": [...]
  }
}
```

**Download Compliance Report**
```http
GET /api/v1/compliance/reports/{task_id}/download?format=json

Formats: json | markdown | pdf

Response: 200 OK
Content-Type: application/json | text/markdown | application/pdf
```

#### Query/Chat

**Send Query**
```http
POST /api/v1/query

Body:
{
  "query": "Jaké jsou povinnosti dodavatele?",
  "document_ids": ["abc123", "def456"],
  "language": "cs"
}

Response: 200 OK
{
  "query": "Jaké jsou povinnosti dodavatele?",
  "answer": "Podle §89...",
  "sources": [
    {
      "legal_reference": "§89 odst. 2",
      "content": "...",
      "document_id": "def456",
      "confidence": 0.92
    }
  ],
  "processing_time_ms": 1200
}
```

#### Health & Status

**Health Check**
```http
GET /api/v1/health

Response: 200 OK
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "redis": "connected",
    "celery": "running",
    "faiss": "loaded"
  }
}
```

---

## 4. FastAPI Implementation

### 4.1 Main Application

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.routers import documents, compliance, query, websocket
from app.core.config import settings
from app.core.dependencies import get_compliance_checker
from app.middleware.logging import LoggingMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    # Startup
    print("🚀 Starting SUJBOT2 API...")

    # Initialize ComplianceChecker (singleton)
    checker = get_compliance_checker()
    await checker.initialize()

    # Load FAISS indices
    await checker.load_indices()

    print("✅ API ready!")

    yield

    # Shutdown
    print("🛑 Shutting down...")
    await checker.cleanup()
    print("✅ Shutdown complete")

app = FastAPI(
    title="SUJBOT2 API",
    description="Legal Compliance Checking API with RAG",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Serve uploaded files (if needed)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "SUJBOT2 API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

# Health check
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "redis": check_redis_connection(),
            "celery": check_celery_status(),
            "faiss": check_faiss_loaded()
        }
    }
```

### 4.2 Configuration

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API
    API_TITLE: str = "SUJBOT2 API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Claude API
    CLAUDE_API_KEY: str
    MAIN_AGENT_MODEL: str = "claude-sonnet-4-5-20250929"
    SUBAGENT_MODEL: str = "claude-3-5-haiku-20241022"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # File storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".odt", ".rtf"]

    # Indexing
    INDEX_DIR: str = "indexes"

    # Authentication (future)
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 4.3 Document Router

```python
# app/routers/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import List, Optional

from app.models.document import DocumentUploadResponse, DocumentStatusResponse, DocumentList
from app.services.document_service import DocumentService
from app.core.dependencies import get_document_service
from app.tasks.indexing import index_document_task

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = "contract",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: DocumentService = Depends(get_document_service)
):
    """
    Upload and index a document.

    - **file**: Document file (PDF, DOCX, TXT, etc.)
    - **document_type**: Type of document (contract, law_code, regulation)

    Returns document metadata and starts background indexing task.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file extension
    file_ext = '.' + file.filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported"
        )

    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.MAX_UPLOAD_SIZE} bytes)"
        )

    # Save file
    document_id = await service.save_uploaded_file(file, document_type)

    # Start indexing task
    task = index_document_task.delay(document_id, document_type)

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
```

### 4.4 Compliance Router

```python
# app/routers/compliance.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models.compliance import ComplianceCheckRequest, ComplianceCheckResponse, ComplianceReportResponse
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
        raise HTTPException(status_code=400, detail="Invalid format")

    report_content = await service.export_report(task_id, format)

    if not report_content:
        raise HTTPException(status_code=404, detail="Report not found")

    media_types = {
        "json": "application/json",
        "markdown": "text/markdown",
        "pdf": "application/pdf"
    }

    from fastapi.responses import Response
    return Response(
        content=report_content,
        media_type=media_types[format],
        headers={
            "Content-Disposition": f"attachment; filename=compliance_report.{format}"
        }
    )
```

---

## 5. WebSocket Implementation

### 5.1 WebSocket Router

```python
# app/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict
import json
import asyncio

from app.services.chat_service import ChatService
from app.core.dependencies import get_chat_service

router = APIRouter()

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_message(self, client_id: str, message: dict):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)

    async def stream_response(self, client_id: str, response_stream):
        """Stream chunks to client."""
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return

        try:
            async for chunk in response_stream:
                await websocket.send_json({
                    "type": "stream_chunk",
                    "content": chunk
                })

            # Send completion message
            await websocket.send_json({
                "type": "stream_complete"
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

manager = ConnectionManager()

@router.websocket("/chat")
async def websocket_chat(
    websocket: WebSocket,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    WebSocket endpoint for real-time chat.

    Message format:
    {
        "type": "chat_message",
        "content": "User question...",
        "document_ids": ["doc1", "doc2"]
    }

    Response format:
    {
        "type": "stream_chunk",
        "content": "Response chunk..."
    }
    """
    # Generate client ID
    client_id = str(id(websocket))

    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "chat_message":
                # Send acknowledgment
                await manager.send_message(client_id, {
                    "type": "message_received",
                    "timestamp": datetime.now().isoformat()
                })

                # Process query and stream response
                response_stream = chat_service.process_query_stream(
                    query=message["content"],
                    document_ids=message.get("document_ids", [])
                )

                await manager.stream_response(client_id, response_stream)

            elif message["type"] == "ping":
                # Heartbeat
                await manager.send_message(client_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(client_id)
```

### 5.2 Chat Service with Streaming

```python
# app/services/chat_service.py
from typing import AsyncIterator
from anthropic import AsyncAnthropic

class ChatService:
    def __init__(self, compliance_checker):
        self.checker = compliance_checker
        self.llm = AsyncAnthropic(api_key=settings.CLAUDE_API_KEY)

    async def process_query_stream(
        self,
        query: str,
        document_ids: list[str]
    ) -> AsyncIterator[str]:
        """
        Process query and stream response chunks.

        Yields response chunks as they arrive from Claude API.
        """
        # Process query
        processed_query = await self.checker.query_processor.process(query)

        # Retrieve relevant chunks
        results = await self.checker.hybrid_retriever.search(
            query=query,
            document_ids=document_ids,
            top_k=5
        )

        # Build context
        context = "\n\n".join([
            f"[{r.legal_reference}]\n{r.content}"
            for r in results
        ])

        # Stream response from Claude
        prompt = f"""Answer this question based on the legal provisions below.

Question: {query}

Legal provisions:
{context}

Provide a clear answer with citations.

Answer:"""

        async with self.llm.messages.stream(
            model=settings.MAIN_AGENT_MODEL,
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

        # After streaming complete, yield sources
        sources_json = json.dumps({
            "type": "sources",
            "sources": [
                {
                    "legal_reference": r.legal_reference,
                    "document_id": r.document_id,
                    "confidence": r.confidence
                }
                for r in results
            ]
        })
        yield "\n\n" + sources_json
```

---

## 6. Celery Task Queue

### 6.1 Celery Configuration

```python
# app/core/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "sujbot2_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.indexing", "app.tasks.compliance"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Prague",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)
```

### 6.2 Indexing Task

```python
# app/tasks/indexing.py
from celery import Task
from app.core.celery_app import celery_app
from app.core.dependencies import get_compliance_checker

class CallbackTask(Task):
    """Custom task with progress callbacks."""

    def update_progress(self, current, total, message=""):
        """Update task progress."""
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "percentage": int((current / total) * 100),
                "message": message
            }
        )

@celery_app.task(bind=True, base=CallbackTask)
def index_document_task(self, document_id: str, document_type: str):
    """
    Index a document asynchronously.

    Args:
        document_id: Document ID
        document_type: 'contract' | 'law_code' | 'regulation'

    Returns:
        Document metadata with indexing info
    """
    checker = get_compliance_checker()

    try:
        # Update progress: parsing
        self.update_progress(0, 100, "Parsing document...")

        # Get document path
        doc_path = get_document_path(document_id)

        # Parse document
        document = await checker.document_reader.read_document(doc_path, document_type)
        self.update_progress(20, 100, "Document parsed")

        # Chunk document
        self.update_progress(20, 100, "Chunking document...")
        chunks = await checker.indexing_pipeline.chunk_document(document)
        self.update_progress(50, 100, f"Generated {len(chunks)} chunks")

        # Generate embeddings and index
        self.update_progress(50, 100, "Generating embeddings...")
        indexed_id = await checker.indexing_pipeline.index_chunks(
            chunks,
            document_id=document_id,
            document_type=document_type
        )
        self.update_progress(90, 100, "Embeddings indexed")

        # Update document metadata in database
        metadata = {
            "page_count": len(document.pages) if hasattr(document, 'pages') else 0,
            "word_count": count_words(document),
            "chunk_count": len(chunks),
            "indexed_at": datetime.now().isoformat()
        }

        save_document_metadata(document_id, metadata)

        self.update_progress(100, 100, "Indexing complete")

        return {
            "document_id": document_id,
            "status": "indexed",
            "metadata": metadata
        }

    except Exception as e:
        # Update document status to error
        update_document_status(document_id, "error", error_message=str(e))
        raise
```

### 6.3 Compliance Check Task

```python
# app/tasks/compliance.py
from app.core.celery_app import celery_app
from app.tasks.indexing import CallbackTask

@celery_app.task(bind=True, base=CallbackTask)
def compliance_check_task(
    self,
    contract_id: str,
    law_ids: list[str],
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
    checker = get_compliance_checker()

    try:
        self.update_progress(0, 100, "Loading documents...")

        # Get document chunks
        contract_chunks = await checker.indexing_pipeline.get_chunks(contract_id)

        law_chunks = []
        for law_id in law_ids:
            chunks = await checker.indexing_pipeline.get_chunks(law_id)
            law_chunks.extend(chunks)

        self.update_progress(20, 100, f"Loaded {len(contract_chunks)} contract chunks, {len(law_chunks)} law chunks")

        # Run compliance analysis
        self.update_progress(20, 100, "Running compliance analysis...")

        report = await checker.compliance_analyzer.analyze_compliance(
            contract_chunks=contract_chunks,
            law_chunks=law_chunks,
            contract_id=contract_id,
            law_ids=law_ids,
            mode=mode
        )

        self.update_progress(90, 100, "Analysis complete")

        # Save report
        report_id = save_compliance_report(report)

        self.update_progress(100, 100, "Report saved")

        return {
            "report_id": report_id,
            "status": "completed",
            "report": dataclasses.asdict(report)
        }

    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise
```

---

## 7. Data Models (Pydantic)

### 7.1 Document Models

```python
# app/models/document.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    filesize: int
    document_type: str
    status: str
    indexing_task_id: str

class DocumentMetadata(BaseModel):
    page_count: int
    word_count: int
    format: str
    indexed_at: Optional[datetime] = None

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str  # uploaded | processing | indexed | error
    progress: int = Field(ge=0, le=100)
    metadata: Optional[DocumentMetadata] = None
    error_message: Optional[str] = None

class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    document_type: str
    status: str
    uploaded_at: datetime
    metadata: Optional[DocumentMetadata] = None

class DocumentList(BaseModel):
    documents: List[DocumentListItem]
    total: int
```

### 7.2 Compliance Models

```python
# app/models/compliance.py
from pydantic import BaseModel
from typing import List, Optional

class ComplianceCheckRequest(BaseModel):
    contract_document_id: str
    law_document_ids: List[str]
    mode: str = "exhaustive"  # exhaustive | sample

class ComplianceCheckResponse(BaseModel):
    task_id: str
    status: str
    estimated_duration_seconds: int

class ComplianceIssue(BaseModel):
    issue_id: str
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW
    status: str  # CONFLICT | MISSING | DEVIATION
    contract_reference: str
    law_references: List[str]
    issue_description: str
    recommendations: List[str]
    risk_score: float

class ComplianceReport(BaseModel):
    report_id: str
    overall_compliance_score: float
    total_issues: int
    critical_issues: List[ComplianceIssue]
    high_issues: List[ComplianceIssue]
    medium_issues: List[ComplianceIssue]
    low_issues: List[ComplianceIssue]
    top_recommendations: List[str]

class ComplianceReportResponse(BaseModel):
    task_id: str
    status: str  # processing | completed | failed
    progress: int
    report: Optional[ComplianceReport] = None
```

---

## 8. Deployment

### 8.1 Running the API

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (Gunicorn + Uvicorn workers)
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### 8.2 Running Celery Workers

```bash
# Start Celery worker
celery -A app.core.celery_app worker --loglevel=info --concurrency=4

# Start Flower (monitoring)
celery -A app.core.celery_app flower --port=5555
```

### 8.3 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Redis
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # FastAPI Backend
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./indexes:/app/indexes
    depends_on:
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  # Celery Worker
  celery_worker:
    build: ./backend
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./indexes:/app/indexes
    depends_on:
      - redis
    command: celery -A app.core.celery_app worker --loglevel=info

  # Celery Flower (monitoring)
  flower:
    build: ./backend
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
    command: celery -A app.core.celery_app flower --port=5555

volumes:
  redis_data:
```

---

## 9. Summary

Kompletní backend API pro SUJBOT2:

✅ **FastAPI**: Modern async web framework
✅ **REST endpoints**: CRUD operations for documents, compliance
✅ **WebSocket**: Real-time streaming chat
✅ **Celery + Redis**: Async task processing
✅ **File upload**: Multipart/form-data with validation
✅ **Progress tracking**: Real-time task status updates
✅ **OpenAPI docs**: Auto-generated at /api/docs
✅ **CORS**: Frontend integration
✅ **Error handling**: Comprehensive exception handling
✅ **Docker ready**: Docker Compose configuration

**Next Steps**:
- See [15. Deployment](15_deployment.md) for full stack orchestration

---

**Page Count**: ~20 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
