# 11. API Interfaces Specification

## 1. Purpose

**Objective**: Provide clean, intuitive Python and REST APIs for integrating SUJBOT2 legal compliance checking into applications and workflows.

**Why API Interfaces?**
- Enable easy integration into existing legal workflows
- Support both programmatic (Python) and HTTP-based access
- Provide batch processing for analyzing multiple documents
- Allow asynchronous operations for long-running tasks
- Future-proof with versioning and extensibility

**Key Capabilities**:
1. **Python API** - Async/await interface for direct integration
2. **REST API** - HTTP endpoints for web/mobile applications (future)
3. **Batch Processing** - Analyze multiple documents efficiently
4. **Streaming Results** - Real-time progress updates
5. **Error Handling** - Comprehensive error types and recovery
6. **API Documentation** - OpenAPI spec and interactive docs

---

## 2. API Architecture

### Overall Structure

```
┌─────────────────────────────────────┐
│  Client Application                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  API Layer                          │
│  ├── Python API (ComplianceChecker) │
│  └── REST API (FastAPI) [future]   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Core Components                    │
│  ├── DocumentReader                 │
│  ├── IndexingPipeline               │
│  ├── HybridRetriever                │
│  ├── ComplianceAnalyzer             │
│  └── KnowledgeGraph                 │
└─────────────────────────────────────┘
```

---

## 3. Python API

### 3.1 Main Interface: ComplianceChecker

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, AsyncIterator
from pathlib import Path
import asyncio

@dataclass
class ComplianceCheckRequest:
    """Request for compliance check."""
    contract_path: str          # Path to contract PDF
    law_paths: List[str]        # Paths to law PDFs
    mode: str = "exhaustive"    # exhaustive | sample
    generate_graph: bool = True
    enable_query_interface: bool = True

@dataclass
class IndexingProgress:
    """Progress update during indexing."""
    document_id: str
    document_name: str
    stage: str                  # parsing | chunking | embedding | indexing
    progress: float             # 0.0 to 1.0
    message: str

@dataclass
class AnalysisProgress:
    """Progress update during analysis."""
    stage: str                  # mapping | checking | scoring | reporting
    progress: float
    message: str
    issues_found: int = 0

class ComplianceChecker:
    """
    Main API for legal compliance checking.

    Usage:
        checker = ComplianceChecker(config_path="config.yaml")
        report = await checker.check_compliance(request)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize compliance checker.

        Args:
            config_path: Path to YAML config file
            config: Dict config (overrides file)
        """
        self.config = self._load_config(config_path, config)

        # Initialize components
        self.document_reader = DocumentReader(self.config)
        self.indexing_pipeline = IndexingPipeline(self.config)
        self.hybrid_retriever = HybridRetriever(self.config)
        self.cross_doc_retriever = ComparativeRetriever(self.config)
        self.compliance_analyzer = ComplianceAnalyzer(
            self.config,
            self.cross_doc_retriever,
            llm_client=Anthropic(api_key=self.config["claude_api_key"])
        )
        self.query_processor = QueryProcessor(self.config)
        self.knowledge_graph: Optional[LegalKnowledgeGraph] = None

        self.logger = logging.getLogger(__name__)

    async def check_compliance(
        self,
        request: ComplianceCheckRequest,
        progress_callback: Optional[callable] = None
    ) -> 'ComplianceReport':
        """
        Perform complete compliance check.

        Args:
            request: ComplianceCheckRequest
            progress_callback: Optional async callback(progress: AnalysisProgress)

        Returns:
            ComplianceReport

        Raises:
            DocumentNotFoundError: If file not found
            IndexingError: If indexing fails
            ComplianceCheckError: If analysis fails
        """
        try:
            # Step 1: Index documents
            self.logger.info("Indexing documents...")
            contract_id = await self.index_document(
                request.contract_path,
                document_type="contract",
                progress_callback=progress_callback
            )

            law_ids = []
            for law_path in request.law_paths:
                law_id = await self.index_document(
                    law_path,
                    document_type="law_code",
                    progress_callback=progress_callback
                )
                law_ids.append(law_id)

            # Step 2: Build knowledge graph (optional)
            if request.generate_graph:
                if progress_callback:
                    await progress_callback(AnalysisProgress(
                        stage="graph_building",
                        progress=0.3,
                        message="Building knowledge graph..."
                    ))

                self.knowledge_graph = await self.build_knowledge_graph(
                    [contract_id] + law_ids
                )

            # Step 3: Run compliance analysis
            if progress_callback:
                await progress_callback(AnalysisProgress(
                    stage="analysis",
                    progress=0.5,
                    message="Running compliance analysis..."
                ))

            # Get chunks
            contract_chunks = await self.indexing_pipeline.get_chunks(contract_id)
            law_chunks = []
            for law_id in law_ids:
                law_chunks.extend(await self.indexing_pipeline.get_chunks(law_id))

            # Analyze
            report = await self.compliance_analyzer.analyze_compliance(
                contract_chunks=contract_chunks,
                law_chunks=law_chunks,
                contract_id=contract_id,
                law_ids=law_ids,
                mode=request.mode
            )

            if progress_callback:
                await progress_callback(AnalysisProgress(
                    stage="complete",
                    progress=1.0,
                    message="Compliance check complete",
                    issues_found=report.total_issues
                ))

            return report

        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            raise ComplianceCheckError(f"Compliance check failed: {e}") from e

    async def index_document(
        self,
        document_path: str,
        document_type: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Index a single document.

        Args:
            document_path: Path to PDF
            document_type: 'contract' | 'law_code' | 'regulation'
            progress_callback: Optional callback for progress

        Returns:
            document_id

        Raises:
            DocumentNotFoundError
            IndexingError
        """
        if not Path(document_path).exists():
            raise DocumentNotFoundError(f"File not found: {document_path}")

        # Parse
        if progress_callback:
            await progress_callback(IndexingProgress(
                document_id="",
                document_name=Path(document_path).name,
                stage="parsing",
                progress=0.0,
                message="Parsing document..."
            ))

        document = await self.document_reader.read_document(document_path, document_type)

        # Chunk
        if progress_callback:
            await progress_callback(IndexingProgress(
                document_id=document.document_id,
                document_name=document.title,
                stage="chunking",
                progress=0.25,
                message="Chunking document..."
            ))

        chunks = await self.indexing_pipeline.chunk_document(document)

        # Embed & Index
        if progress_callback:
            await progress_callback(IndexingProgress(
                document_id=document.document_id,
                document_name=document.title,
                stage="embedding",
                progress=0.5,
                message="Generating embeddings..."
            ))

        document_id = await self.indexing_pipeline.index_chunks(
            chunks,
            document_id=document.document_id,
            document_type=document_type
        )

        if progress_callback:
            await progress_callback(IndexingProgress(
                document_id=document_id,
                document_name=document.title,
                stage="complete",
                progress=1.0,
                message="Indexing complete"
            ))

        return document_id

    async def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> 'QueryResponse':
        """
        Ask a question about indexed documents.

        Args:
            query: Natural language question
            document_ids: Filter to specific documents (None = all)
            top_k: Number of results

        Returns:
            QueryResponse with answer and sources
        """
        # Process query
        processed_query = await self.query_processor.process(query)

        # Retrieve relevant chunks
        if processed_query.retrieval_strategy == "cross_document":
            results = await self.cross_doc_retriever.search(
                query=query,
                target_document_types=["contract", "law_code"],
                top_k=top_k
            )
        else:
            results = await self.hybrid_retriever.search(
                query=query,
                document_ids=document_ids,
                top_k=top_k
            )

        # Generate answer (using Claude)
        answer = await self._generate_answer(query, results)

        return QueryResponse(
            query=query,
            answer=answer,
            sources=[self._result_to_source(r) for r in results],
            processed_query=processed_query
        )

    async def build_knowledge_graph(
        self,
        document_ids: List[str]
    ) -> LegalKnowledgeGraph:
        """
        Build knowledge graph for indexed documents.

        Args:
            document_ids: Documents to include in graph

        Returns:
            LegalKnowledgeGraph
        """
        # Get document structures
        documents = []
        for doc_id in document_ids:
            doc = await self.indexing_pipeline.get_document(doc_id)
            documents.append(doc)

        # Build graph
        builder = GraphBuilder()
        kg = builder.build_from_documents(documents)

        # Link references
        ref_linker = ReferenceLinker()
        ref_linker.link_references(kg)

        # Link semantically
        semantic_linker = SemanticLinker(
            self.indexing_pipeline.embedding_model,
            threshold=0.75
        )
        await semantic_linker.link_semantically(kg)

        self.knowledge_graph = kg
        return kg

    async def batch_check_compliance(
        self,
        requests: List[ComplianceCheckRequest],
        max_parallel: int = 3
    ) -> AsyncIterator['ComplianceReport']:
        """
        Process multiple compliance checks in parallel.

        Args:
            requests: List of ComplianceCheckRequest
            max_parallel: Max concurrent checks

        Yields:
            ComplianceReport for each request
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def check_with_semaphore(req):
            async with semaphore:
                return await self.check_compliance(req)

        tasks = [check_with_semaphore(req) for req in requests]

        for task in asyncio.as_completed(tasks):
            report = await task
            yield report

    def export_report(
        self,
        report: 'ComplianceReport',
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export compliance report to file.

        Args:
            report: ComplianceReport
            output_path: Output file path
            format: 'json' | 'markdown' | 'html' | 'pdf'
        """
        exporter = ReportExporter()
        exporter.export(report, output_path, format)

    def _load_config(self, config_path: Optional[str], config: Optional[Dict]) -> Dict:
        """Load configuration."""
        if config:
            return config

        if config_path:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)

        # Default config
        return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "claude_api_key": os.getenv("CLAUDE_API_KEY"),
            "indexing": {
                "chunk_size": 512,
                "chunk_overlap": 0.15
            },
            "retrieval": {
                "hybrid_alpha": 0.7,
                "top_k": 20,
                "rerank_top_k": 5
            },
            "compliance": {
                "default_mode": "exhaustive"
            }
        }

    async def _generate_answer(self, query: str, results: List) -> str:
        """Generate answer using Claude."""
        # Build context from results
        context = "\n\n".join([
            f"[{r.legal_reference}]\n{r.content}"
            for r in results[:5]
        ])

        prompt = f"""Answer this question based on the legal provisions below.

Question: {query}

Legal provisions:
{context}

Provide a clear, concise answer with citations to specific provisions.

Answer:"""

        response = await self.compliance_analyzer.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _result_to_source(self, result) -> 'Source':
        """Convert search result to Source object."""
        return Source(
            legal_reference=result.legal_reference,
            content=result.content,
            document_id=result.document_id,
            confidence=result.confidence
        )

@dataclass
class QueryResponse:
    """Response from query API."""
    query: str
    answer: str
    sources: List['Source']
    processed_query: 'ProcessedQuery'

@dataclass
class Source:
    """Source citation."""
    legal_reference: str
    content: str
    document_id: str
    confidence: float
```

---

## 4. Error Handling

### 4.1 Exception Hierarchy

```python
class SUJBOT2Error(Exception):
    """Base exception for all SUJBOT2 errors."""
    pass

class DocumentNotFoundError(SUJBOT2Error):
    """Document file not found."""
    pass

class DocumentParsingError(SUJBOT2Error):
    """Error parsing document."""
    pass

class IndexingError(SUJBOT2Error):
    """Error during indexing."""
    pass

class RetrievalError(SUJBOT2Error):
    """Error during retrieval."""
    pass

class ComplianceCheckError(SUJBOT2Error):
    """Error during compliance check."""
    pass

class ConfigurationError(SUJBOT2Error):
    """Invalid configuration."""
    pass

class APIKeyError(SUJBOT2Error):
    """Missing or invalid API key."""
    pass
```

### 4.2 Error Handling Example

```python
try:
    report = await checker.check_compliance(request)
except DocumentNotFoundError as e:
    logger.error(f"Document not found: {e}")
    # Handle missing file
except IndexingError as e:
    logger.error(f"Indexing failed: {e}")
    # Retry or notify user
except ComplianceCheckError as e:
    logger.error(f"Compliance check failed: {e}")
    # Generate partial report
except SUJBOT2Error as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

---

## 5. REST API (Future)

### 5.1 API Design

```
POST   /api/v1/documents/index
POST   /api/v1/compliance/check
GET    /api/v1/compliance/reports/{report_id}
POST   /api/v1/query
GET    /api/v1/documents
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST   /api/v1/batch/compliance
GET    /api/v1/batch/{batch_id}/status
```

### 5.2 FastAPI Implementation (Sketch)

```python
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="SUJBOT2 API", version="1.0.0")

class ComplianceCheckPayload(BaseModel):
    contract_document_id: str
    law_document_ids: List[str]
    mode: str = "exhaustive"

@app.post("/api/v1/documents/index")
async def index_document(
    file: UploadFile,
    document_type: str
):
    """Upload and index a document."""
    # Save file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Index
    try:
        document_id = await checker.index_document(temp_path, document_type)
        return {"document_id": document_id, "status": "indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/compliance/check")
async def check_compliance(
    payload: ComplianceCheckPayload,
    background_tasks: BackgroundTasks
):
    """Start compliance check (async)."""
    # Create background task
    task_id = str(uuid.uuid4())

    background_tasks.add_task(
        run_compliance_check,
        task_id,
        payload
    )

    return {"task_id": task_id, "status": "processing"}

@app.get("/api/v1/compliance/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve compliance report."""
    # Load report from storage
    report = load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report

async def run_compliance_check(task_id: str, payload: ComplianceCheckPayload):
    """Background task for compliance check."""
    # Run check
    request = ComplianceCheckRequest(
        contract_path=get_document_path(payload.contract_document_id),
        law_paths=[get_document_path(lid) for lid in payload.law_document_ids],
        mode=payload.mode
    )

    report = await checker.check_compliance(request)

    # Save report
    save_report(task_id, report)
```

---

## 6. Batch Processing

### 6.1 Batch API

```python
class BatchProcessor:
    """Process multiple documents in batch."""

    def __init__(self, checker: ComplianceChecker):
        self.checker = checker

    async def process_batch(
        self,
        batch_requests: List[ComplianceCheckRequest],
        batch_id: Optional[str] = None
    ) -> 'BatchResult':
        """
        Process batch of compliance checks.

        Args:
            batch_requests: List of requests
            batch_id: Optional batch identifier

        Returns:
            BatchResult with all reports
        """
        batch_id = batch_id or str(uuid.uuid4())

        reports = []
        errors = []

        async for report in self.checker.batch_check_compliance(batch_requests):
            reports.append(report)

        return BatchResult(
            batch_id=batch_id,
            total_requests=len(batch_requests),
            successful=len(reports),
            failed=len(errors),
            reports=reports,
            errors=errors
        )

@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    total_requests: int
    successful: int
    failed: int
    reports: List['ComplianceReport']
    errors: List[Dict[str, str]]  # {request_index, error}
```

---

## 7. Configuration

### 7.1 config.yaml

```yaml
# API configuration
api:
  version: "1.0.0"
  environment: "development"  # development | production

  # Python API
  python_api:
    enable_progress_callbacks: true
    max_parallel_tasks: 5

  # REST API (future)
  rest_api:
    host: "0.0.0.0"
    port: 8000
    enable_cors: true
    allowed_origins: ["*"]

  # Rate limiting (future)
  rate_limiting:
    enabled: false
    requests_per_minute: 60

  # Authentication (future)
  authentication:
    enabled: false
    method: "api_key"  # api_key | oauth2

  # Logging
  logging:
    level: "INFO"
    format: "json"
    output: "stdout"

# Document processing
document_processing:
  max_file_size_mb: 500
  supported_formats: ["pdf", "docx"]
  temp_storage_path: "/tmp/sujbot2"

# Indexing
indexing:
  chunk_size: 512
  chunk_overlap: 0.15
  batch_size: 32

# Retrieval
retrieval:
  hybrid_alpha: 0.7
  top_k: 20
  rerank_top_k: 5

# Compliance
compliance:
  default_mode: "exhaustive"
  generate_recommendations: true

# LLM
llm:
  main_model: "claude-sonnet-4-5-20250929"
  sub_model: "claude-3-5-haiku-20241022"
  temperature: 0.1
  max_tokens: 4000
```

---

## 8. Usage Examples

### 8.1 Basic Usage

```python
import asyncio
from advanced_sujbot2 import ComplianceChecker, ComplianceCheckRequest

async def main():
    # Initialize
    checker = ComplianceChecker(config_path="config.yaml")

    # Create request
    request = ComplianceCheckRequest(
        contract_path="documents/smlouva_jaderka.pdf",
        law_paths=["documents/zakon_89_2012.pdf"],
        mode="exhaustive"
    )

    # Run check
    report = await checker.check_compliance(request)

    # Print summary
    print(f"Compliance Score: {report.overall_compliance_score:.2%}")
    print(f"Total Issues: {report.total_issues}")
    print(f"  Critical: {len(report.critical_issues)}")
    print(f"  High: {len(report.high_issues)}")

    # Export report
    checker.export_report(report, "compliance_report.json", format="json")
    checker.export_report(report, "compliance_report.md", format="markdown")

asyncio.run(main())
```

### 8.2 With Progress Callback

```python
async def progress_callback(progress):
    """Handle progress updates."""
    if isinstance(progress, IndexingProgress):
        print(f"[Indexing] {progress.document_name}: {progress.stage} ({progress.progress:.0%})")
    elif isinstance(progress, AnalysisProgress):
        print(f"[Analysis] {progress.stage}: {progress.message} ({progress.progress:.0%})")

async def main():
    checker = ComplianceChecker(config_path="config.yaml")

    request = ComplianceCheckRequest(
        contract_path="smlouva.pdf",
        law_paths=["zakon.pdf"]
    )

    report = await checker.check_compliance(request, progress_callback=progress_callback)

asyncio.run(main())
```

### 8.3 Query Interface

```python
async def main():
    checker = ComplianceChecker(config_path="config.yaml")

    # Index documents first
    contract_id = await checker.index_document("smlouva.pdf", "contract")
    law_id = await checker.index_document("zakon.pdf", "law_code")

    # Ask questions
    response = await checker.query(
        query="Jaké jsou povinnosti dodavatele podle §89?",
        document_ids=[contract_id, law_id]
    )

    print(f"Question: {response.query}")
    print(f"Answer: {response.answer}")
    print("\nSources:")
    for source in response.sources:
        print(f"  - {source.legal_reference}: {source.content[:100]}...")

asyncio.run(main())
```

### 8.4 Batch Processing

```python
async def main():
    checker = ComplianceChecker(config_path="config.yaml")
    batch_processor = BatchProcessor(checker)

    # Create batch requests
    requests = [
        ComplianceCheckRequest(
            contract_path=f"contracts/contract_{i}.pdf",
            law_paths=["laws/zakon_89_2012.pdf"],
            mode="sample"
        )
        for i in range(10)
    ]

    # Process batch
    print("Processing batch of 10 contracts...")
    batch_result = await batch_processor.process_batch(requests)

    print(f"Batch complete: {batch_result.successful}/{batch_result.total_requests} successful")

    # Export all reports
    for i, report in enumerate(batch_result.reports):
        checker.export_report(report, f"reports/report_{i}.json")

asyncio.run(main())
```

---

## 9. Testing

### 9.1 Unit Tests

```python
import pytest
from advanced_sujbot2 import ComplianceChecker, ComplianceCheckRequest

@pytest.mark.asyncio
async def test_document_indexing():
    """Test document indexing."""
    checker = ComplianceChecker()

    document_id = await checker.index_document(
        "test_data/sample_contract.pdf",
        document_type="contract"
    )

    assert document_id is not None
    assert len(document_id) > 0

@pytest.mark.asyncio
async def test_compliance_check():
    """Test compliance check."""
    checker = ComplianceChecker()

    request = ComplianceCheckRequest(
        contract_path="test_data/sample_contract.pdf",
        law_paths=["test_data/sample_law.pdf"],
        mode="sample"
    )

    report = await checker.check_compliance(request)

    assert report is not None
    assert report.total_issues >= 0
    assert 0 <= report.overall_compliance_score <= 1

@pytest.mark.asyncio
async def test_query_interface():
    """Test query interface."""
    checker = ComplianceChecker()

    # Index documents
    await checker.index_document("test_data/sample_contract.pdf", "contract")

    # Query
    response = await checker.query("What are the payment terms?")

    assert response.answer is not None
    assert len(response.sources) > 0
```

### 9.2 Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_compliance_workflow():
    """Test complete compliance workflow."""
    checker = ComplianceChecker(config_path="config.yaml")

    # Step 1: Index
    contract_id = await checker.index_document(
        "test_data/full_contract.pdf",
        "contract"
    )
    law_id = await checker.index_document(
        "test_data/full_law.pdf",
        "law_code"
    )

    # Step 2: Build graph
    kg = await checker.build_knowledge_graph([contract_id, law_id])
    assert kg.graph.number_of_nodes() > 0

    # Step 3: Run compliance check
    request = ComplianceCheckRequest(
        contract_path="test_data/full_contract.pdf",
        law_paths=["test_data/full_law.pdf"],
        mode="exhaustive"
    )
    report = await checker.check_compliance(request)

    assert report.total_issues >= 0

    # Step 4: Export
    checker.export_report(report, "test_output/report.json", "json")

    # Verify export
    import json
    with open("test_output/report.json") as f:
        exported = json.load(f)
        assert exported["total_issues"] == report.total_issues
```

---

## 10. Deployment

### 10.1 Package Structure

```
advanced_sujbot2/
├── __init__.py                 # Public API exports
├── api/
│   ├── __init__.py
│   ├── compliance_checker.py  # Main Python API
│   ├── rest_api.py            # FastAPI endpoints (future)
│   └── batch_processor.py
├── core/
│   ├── document_reader.py
│   ├── indexing_pipeline.py
│   ├── hybrid_retriever.py
│   ├── compliance_analyzer.py
│   └── knowledge_graph.py
├── config/
│   └── default_config.yaml
└── utils/
    ├── error_handling.py
    └── report_exporter.py
```

### 10.2 Installation

```bash
# From PyPI (future)
pip install advanced-sujbot2

# From source
git clone https://github.com/yourusername/advanced-sujbot2.git
cd advanced-sujbot2
pip install -e .
```

### 10.3 Docker Deployment (Future)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "advanced_sujbot2.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. API Documentation

### 11.1 OpenAPI Spec (Future)

```yaml
openapi: 3.0.0
info:
  title: SUJBOT2 API
  version: 1.0.0
  description: Legal compliance checking API

paths:
  /api/v1/documents/index:
    post:
      summary: Index a document
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                document_type:
                  type: string
                  enum: [contract, law_code, regulation]
      responses:
        200:
          description: Document indexed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  document_id:
                    type: string
                  status:
                    type: string

  /api/v1/compliance/check:
    post:
      summary: Check compliance
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ComplianceCheckPayload'
      responses:
        200:
          description: Compliance check started
          content:
            application/json:
              schema:
                type: object
                properties:
                  task_id:
                    type: string
                  status:
                    type: string

components:
  schemas:
    ComplianceCheckPayload:
      type: object
      properties:
        contract_document_id:
          type: string
        law_document_ids:
          type: array
          items:
            type: string
        mode:
          type: string
          enum: [exhaustive, sample]
```

---

## 12. Summary

The API layer provides:

1. **Clean Python API** - Async/await interface for direct integration
2. **Progress Tracking** - Real-time callbacks for long-running operations
3. **Batch Processing** - Efficient multi-document analysis
4. **Error Handling** - Comprehensive exception hierarchy
5. **Extensibility** - Future REST API, authentication, rate limiting
6. **Easy Integration** - Simple, intuitive interface

**Complete Workflow**:
```python
# 1. Initialize
checker = ComplianceChecker(config_path="config.yaml")

# 2. Create request
request = ComplianceCheckRequest(
    contract_path="smlouva.pdf",
    law_paths=["zakon.pdf"]
)

# 3. Run check
report = await checker.check_compliance(request)

# 4. Export results
checker.export_report(report, "report.json")
```

---

**Page Count**: ~16 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅

---

## ALL SPECIFICATIONS NOW COMPLETE!

All 12 specifications have been written:
1. ✅ Architecture Overview
2. ✅ Document Reader
3. ✅ Chunking Strategy
4. ✅ Embedding & Indexing
5. ✅ Hybrid Retrieval
6. ✅ Cross-Document Retrieval
7. ✅ Reranking
8. ✅ Query Processing
9. ✅ Compliance Analyzer
10. ✅ Knowledge Graph
11. ✅ API Interfaces
12. ✅ Implementation Roadmap

**Total: ~200 pages of detailed technical specifications**

The SUJBOT2 system is fully specified and ready for implementation!
