# Architecture Overview - SUJBOT2

## 1. Executive Summary

**Purpose**: Multi-document legal compliance checking system for comparing 1000+ page contracts against Czech legal code using advanced RAG techniques.

**Core Capability**: Automatically identify discrepancies, gaps, and conflicts between contract provisions and legal requirements.

**Key Innovation**: Graph-enhanced multi-document retrieval with compliance-aware reasoning.

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVANCED SUJBOT2 ARCHITECTURE                │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
├── Contract Document (PDF/DOCX)
├── Legal Code (PDF/XML)
└── User Query (Natural Language)

DOCUMENT PROCESSING LAYER
├── LegalDocumentReader
│   ├── Format Detection
│   ├── Structure Parsing (§, článek, odst.)
│   └── Reference Extraction
├── HierarchicalLegalChunker
│   ├── Law Chunking (by paragraph)
│   ├── Contract Chunking (by article)
│   └── Metadata Enrichment
└── EmbeddingGenerator
    ├── Model: BAAI/bge-m3 (1024 dim)
    ├── Contextualized Embeddings
    └── Batch Processing

STORAGE LAYER
├── MultiDocumentVectorStore
│   ├── Contract Index (FAISS)
│   ├── Law Code Index (FAISS)
│   └── Reference Mapping
├── KnowledgeGraph
│   ├── Hierarchical Structure (NetworkX)
│   ├── Cross-References
│   └── Compliance Relationships
└── MetadataStore
    ├── Legal References
    ├── Hierarchy Paths
    └── Content Classifications

RETRIEVAL LAYER
├── TripleHybridRetriever
│   ├── Semantic Search (50%)
│   ├── Keyword Search (30%)
│   └── Structured Search (20%)
├── ComparativeRetriever
│   ├── Cross-Document Matching
│   ├── Explicit Reference Lookup
│   └── Implicit Semantic Matching
└── GraphAwareReranker
    ├── Cross-Encoder Scoring
    ├── Graph Proximity
    └── Legal Precedence Weighting

REASONING LAYER
├── LegalQuestionDecomposer
│   ├── Compliance Query Types
│   ├── Gap Analysis Queries
│   └── Risk Assessment Queries
├── ComplianceAnalyzer
│   ├── Clause-Level Checking
│   ├── Gap Detection
│   └── Conflict Identification
└── RiskScorer
    ├── Severity Assessment
    ├── Impact Analysis
    └── Precedent Lookup

SYNTHESIS LAYER
├── AnswerSynthesizer (Claude Sonnet 4.5)
│   ├── Multi-Source Integration
│   ├── Citation Generation
│   └── Confidence Scoring
└── ComplianceReporter
    ├── Issue Aggregation
    ├── Recommendation Generation
    └── Report Formatting

OUTPUT LAYER
├── Compliance Report (JSON/PDF)
├── Interactive Q&A Interface
└── API Endpoints
```

### 2.2 Data Flow

```
┌──────────────┐  ┌──────────────┐
│  Contract    │  │  Law Code    │
└──────┬───────┘  └──────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────────────────────────┐
│  LegalDocumentReader            │
│  - Parse structure              │
│  - Extract references           │
│  - Classify content             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  HierarchicalLegalChunker       │
│  - Chunk by legal structure     │
│  - Preserve hierarchy           │
│  - Add metadata                 │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Parallel Indexing              │
│  ┌──────────┐  ┌──────────┐    │
│  │Contract  │  │Law Code  │    │
│  │FAISS     │  │FAISS     │    │
│  └──────────┘  └──────────┘    │
│         ┌───────────┐           │
│         │Knowledge  │           │
│         │Graph      │           │
│         └───────────┘           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  User Query                     │
│  "Najdi slabá místa ve smlouvě" │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  LegalQuestionDecomposer        │
│  - Classify query type          │
│  - Generate sub-queries         │
│  - Identify entities            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  TripleHybridRetrieval          │
│  - Semantic (embeddings)        │
│  - Keyword (BM25)               │
│  - Structured (hierarchy)       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ComparativeRetrieval           │
│  - Find matching provisions     │
│  - Cross-document search        │
│  - Reference tracking           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  GraphAwareReranking            │
│  - Cross-encoder scores         │
│  - Graph proximity              │
│  - Legal precedence             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ComplianceAnalyzer             │
│  - Check each clause            │
│  - Identify conflicts           │
│  - Detect gaps                  │
│  - Score risks                  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  AnswerSynthesizer (Claude)     │
│  - Aggregate findings           │
│  - Generate explanations        │
│  - Provide recommendations      │
└──────────────┬──────────────────┘
               │
               ▼
       ┌───────────────┐
       │Compliance     │
       │Report         │
       └───────────────┘
```

---

## 3. Core Design Principles

### 3.1 Multi-Document Architecture

**Problem**: Traditional RAG systems index single documents. Legal compliance requires comparing TWO documents.

**Solution**:
- Separate FAISS indices for contract vs law
- Cross-document retrieval mechanisms
- Knowledge graph connecting both documents

### 3.2 Structure Awareness

**Problem**: Legal documents have hierarchical structure (Část > Hlava > § > odst.) that semantic chunking ignores.

**Solution**:
- Parse document structure during ingestion
- Chunk by legal boundaries (paragraphs, articles)
- Preserve hierarchy in metadata
- Enable structured queries

### 3.3 Compliance-First Design

**Problem**: Generic QA systems don't understand legal compliance requirements.

**Solution**:
- Specialized query types (gap analysis, conflict detection)
- Compliance-aware retrieval (mandatory vs optional provisions)
- Risk scoring based on legal precedence
- Recommendation generation

### 3.4 Graph-Enhanced Retrieval

**Problem**: Legal documents are highly interconnected via references. Flat retrieval misses these connections.

**Solution**:
- Knowledge graph of legal structure
- Reference tracking (§X refers to §Y)
- Graph-aware reranking (proximity scoring)
- Multi-hop reasoning

---

## 4. Technology Stack

### 4.1 Core Technologies

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Embeddings** | BAAI/bge-m3 | latest | SOTA multilingual, 1024 dim, Czech support |
| **Vector DB** | FAISS | 1.7+ | Fast exact search, local deployment |
| **Keyword Search** | BM25Okapi | - | Classic IR, good for exact terms |
| **Reranker** | cross-encoder/mmarco-mMiniLMv2 | - | Multilingual cross-encoder |
| **LLM** | Claude Sonnet 4.5 | latest | Best reasoning for compliance |
| **LLM (decomp)** | Claude Haiku 3.5 | latest | Cost-efficient for sub-tasks |
| **Graph** | NetworkX | 3.0+ | Python-native, flexible |
| **Tokenizer** | tiktoken | latest | GPT-4 tokenizer |
| **OCR/PDF** | pdfplumber | latest | Robust PDF parsing |

### 4.2 Programming Paradigm

- **Language**: Python 3.10+
- **Async**: asyncio for parallel operations
- **Type Hints**: Full typing support
- **Error Handling**: Comprehensive exception hierarchy
- **Logging**: Structured logging with context
- **Testing**: pytest with >80% coverage

---

## 5. Key Interfaces

### 5.1 Main Entry Point

```python
from advanced_sujbot2 import ComplianceChecker

# Initialize
checker = ComplianceChecker(config_path="config.yaml")

# Index documents
contract_id = await checker.index_document(
    "smlouva_jaderny_blok.pdf",
    document_type="contract"
)

law_id = await checker.index_document(
    "zakon_89_2012.pdf",
    document_type="law_code"
)

# Build knowledge graph
await checker.build_knowledge_graph([contract_id, law_id])

# Run compliance check
report = await checker.check_compliance(
    contract_id=contract_id,
    applicable_laws=[law_id],
    mode="exhaustive"
)

# Query interface
answer = await checker.query(
    "Najdi všechna slabá místa ve smlouvě",
    context={'contract': contract_id, 'law': law_id}
)
```

### 5.2 Output Format

```python
@dataclass
class ComplianceReport:
    contract_id: str
    law_id: str
    timestamp: datetime

    summary: ComplianceSummary
    issues: List[ComplianceIssue]
    gaps: List[GapAnalysisResult]
    recommendations: List[Recommendation]

    metadata: Dict[str, Any]

@dataclass
class ComplianceIssue:
    issue_id: str
    contract_section: str  # "Článek 5.2"
    law_references: List[str]  # ["§89 odst. 2"]

    issue_type: str  # CONFLICT | DEVIATION | MISSING
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW

    description: str
    legal_requirement: str
    contract_provision: str

    impact: str
    recommendation: str

    confidence: float
    evidence: List[Citation]
```

---

## 6. Performance Targets

### 6.1 Indexing Performance

| Document Size | Target Time | Memory |
|--------------|-------------|--------|
| 100 pages | <10s | <100 MB |
| 1,000 pages | <1 min | <500 MB |
| 10,000 pages | <5 min | <2 GB |

### 6.2 Query Performance

| Query Type | Target Latency | Accuracy |
|-----------|----------------|----------|
| Simple Q&A | <2s | >85% |
| Compliance check (single clause) | <5s | >90% |
| Full contract scan | <2 min | >90% |
| Gap analysis | <3 min | >85% |

### 6.3 Quality Metrics

- **Recall**: >90% (finds all actual issues)
- **Precision**: >85% (minimal false positives)
- **Critical Miss Rate**: <5% (rarely misses CRITICAL issues)
- **Expert Agreement**: >80% (matches legal expert assessment)

---

## 7. Scalability Considerations

### 7.1 Document Scale

- **Current scope**: 2 documents (contract + law), 10k pages each
- **Future scale**: Multiple contracts vs multiple laws
- **Solution**: Distributed indexing with Ray, vector DB switch to Qdrant

### 7.2 Query Scale

- **Current**: Single-user, interactive queries
- **Future**: Multi-user, batch processing
- **Solution**: Request queuing, caching, horizontal scaling

### 7.3 Memory Management

- **Index caching**: LRU cache for frequently accessed indices
- **Lazy loading**: Load index only when needed
- **Chunked processing**: Process large documents in batches

---

## 8. Security & Compliance

### 8.1 Data Privacy

- All processing local (FAISS) or with trusted LLM provider (Anthropic)
- No data retention in LLM provider
- Encrypted storage for sensitive documents

### 8.2 Audit Trail

- Log all queries and results
- Track all compliance findings
- Version control for indices

---

## 9. Configuration Strategy

### 9.1 Multi-Level Configuration

```yaml
# config.yaml - hierarchical config
system:
  mode: production  # development | testing | production

documents:
  types:
    - contract
    - law_code
    - regulation

indexing:
  chunking_strategy: hierarchical_legal
  contract:
    chunk_by: article
    min_size: 128
    max_size: 1024
  law_code:
    chunk_by: paragraph
    include_context: true

embeddings:
  model: BAAI/bge-m3
  device: mps
  batch_size: 32

retrieval:
  hybrid_weights:
    semantic: 0.5
    keyword: 0.3
    structural: 0.2
  top_k: 20
  rerank_top_k: 5

compliance:
  mode: exhaustive
  check_all_clauses: true
  gap_analysis: true
  risk_scoring: true

llm:
  main_model: claude-sonnet-4-5-20250929
  sub_model: claude-3-5-haiku-20241022
  temperature: 0.1
  max_tokens: 4000
```

---

## 10. Extension Points

### 10.1 Plugin Architecture

- **Custom Chunkers**: Add domain-specific chunking strategies
- **Custom Retrievers**: Add specialized retrieval methods
- **Custom Analyzers**: Add compliance rules for specific domains
- **Custom Scorers**: Add custom risk scoring models

### 10.2 API Extensions

- REST API for external integrations
- Webhook support for async notifications
- Batch processing API for bulk checks

---

## 11. Deployment Architecture

### 11.1 Local Deployment (Current)

```
┌─────────────────────────┐
│   User's Machine        │
│                         │
│  ┌──────────────────┐  │
│  │  Python App      │  │
│  │  - FAISS local   │  │
│  │  - NetworkX      │  │
│  └────────┬─────────┘  │
│           │             │
│           ▼             │
│  ┌──────────────────┐  │
│  │  Claude API      │──┼──> Anthropic
│  └──────────────────┘  │
└─────────────────────────┘
```

### 11.2 Production Deployment (Future)

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│   Load Balancer         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   App Servers (N)       │
│   - Stateless           │
│   - Auto-scaling        │
└──────┬──────────────────┘
       │
       ▼
┌──────────────┬──────────┐
│   Qdrant     │ Claude   │
│   Vector DB  │ API      │
└──────────────┴──────────┘
```

---

## 12. Development Phases

### Phase 1: Foundation (Weeks 1-2)
- Multi-document indexing
- Legal structure parsing
- Hierarchical chunking

### Phase 2: Retrieval (Weeks 3-4)
- Triple hybrid search
- Cross-document retrieval
- Embedding upgrade to BGE-M3

### Phase 3: Compliance (Weeks 5-6)
- Compliance analyzer
- Gap detection
- Risk scoring

### Phase 4: Advanced (Weeks 7-8)
- Knowledge graph
- Graph-aware reranking
- Advanced query processing

### Phase 5: Production (Weeks 9-10)
- Testing & optimization
- Documentation
- Deployment

---

## 13. Success Criteria

### 13.1 Functional Requirements

- ✅ Index 10k page documents in <5 min
- ✅ Identify 90%+ of actual compliance issues
- ✅ <10% false positive rate
- ✅ Generate actionable recommendations
- ✅ Support Czech legal terminology

### 13.2 Non-Functional Requirements

- ✅ Response time <5s for single-clause checks
- ✅ Memory usage <2GB for 10k pages
- ✅ Support concurrent queries
- ✅ Maintain audit trail
- ✅ Handle corrupted PDFs gracefully

---

## 14. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Low accuracy on Czech text | Use multilingual models (BGE-M3), test extensively |
| False positives | Human-in-the-loop validation, confidence thresholds |
| Scalability limits | Design for distributed architecture from start |
| LLM hallucinations | Grounding in retrieved chunks, low temperature |
| Complex legal edge cases | Expert review for high-stakes decisions |

---

## 15. Future Enhancements

### 15.1 Short-term (3-6 months)
- Fine-tune embeddings on Czech legal corpus
- Add precedent database integration
- Multi-language support (SK, PL, EN)

### 15.2 Long-term (6-12 months)
- Automated contract drafting suggestions
- Change detection (law amendments)
- Multi-modal support (tables, diagrams)
- Interactive visual compliance dashboard
