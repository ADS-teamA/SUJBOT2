# SUJBOT2 - Legal Compliance Checker

Multi-document legal compliance checking system for comparing contracts against Czech legal code using advanced RAG techniques.

## 📋 Project Overview

**Purpose**: Automatically identify discrepancies, gaps, and conflicts between 1000+ page contract provisions and Czech legal requirements.

**Key Features**:
- ✅ Multi-document retrieval (contract ↔ law)
- ✅ Hierarchical legal structure parsing (§, články, odstavce)
- ✅ Graph-enhanced reasoning with cross-references
- ✅ Automated compliance checking with risk scoring
- ✅ Czech language support with multilingual embeddings

---

## 📚 Documentation

Complete technical specifications are in the `/specs` directory:

### Core Components
1. **[Architecture Overview](specs/01_architecture_overview.md)** - System design, data flow, technology stack
2. **[Document Reader](specs/02_document_reader.md)** - Legal structure parsing and reference extraction
3. **[Chunking Strategy](specs/03_chunking_strategy.md)** - Hierarchical legal chunking by §, articles
4. **[Embedding & Indexing](specs/04_embedding_indexing.md)** - BGE-M3 embeddings, multi-document vector store

### Advanced Features
5. **Hybrid Retrieval** - Triple hybrid search (semantic + keyword + structural)
6. **Cross-Document Retrieval** - Comparative retrieval across contract and law
7. **Reranking** - Graph-aware reranking with legal precedence
8. **Query Processing** - Legal question decomposition for compliance queries
9. **Compliance Analyzer** - Automated conflict detection and gap analysis
10. **Knowledge Graph** - Legal structure and reference graph with NetworkX
11. **API Interfaces** - Python API and future REST endpoints

### Implementation
12. **[Implementation Roadmap](specs/12_implementation_roadmap.md)** - 10-week development plan

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 16GB+ RAM
- CUDA GPU (optional, for faster embeddings)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd advanced_sujbot2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your CLAUDE_API_KEY
```

### Basic Usage

```python
from advanced_sujbot2 import ComplianceChecker

# Initialize
checker = ComplianceChecker(config_path="config.yaml")

# Index documents
contract_id = await checker.index_document(
    "smlouva.pdf",
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

# View results
print(f"Found {report.total_issues} issues")
for issue in report.critical_issues:
    print(f"- {issue.contract_section}: {issue.description}")
```

### Interactive Query

```python
# Ask questions about the documents
answer = await checker.query(
    "Najdi všechna slabá místa ve smlouvě, která se neshodují se zákonem",
    context={'contract': contract_id, 'law': law_id}
)

print(answer.final_answer)
print(f"Sources: {len(answer.sources)}")
print(f"Confidence: {answer.confidence:.2f}")
```

---

## 🏗️ Architecture

```
┌──────────────┐  ┌──────────────┐
│  Contract    │  │  Law Code    │
└──────┬───────┘  └──────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────────────────────────┐
│  LegalDocumentReader            │
│  - Parse structure (§, články)  │
│  - Extract references           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  HierarchicalLegalChunker       │
│  - Chunk by legal boundaries    │
│  - Preserve hierarchy metadata  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Multi-Document Vector Store    │
│  ┌──────────┐  ┌──────────┐    │
│  │Contract  │  │Law       │    │
│  │Index     │  │Index     │    │
│  └──────────┘  └──────────┘    │
│         ┌───────────┐           │
│         │Knowledge  │           │
│         │Graph      │           │
│         └───────────┘           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Triple Hybrid Retrieval        │
│  - Semantic (BGE-M3)            │
│  - Keyword (BM25)               │
│  - Structured (hierarchy)       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ComplianceAnalyzer             │
│  - Conflict detection           │
│  - Gap analysis                 │
│  - Risk scoring                 │
└──────────────┬──────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Compliance    │
       │ Report        │
       └───────────────┘
```

---

## 🔧 Configuration

Key configuration parameters in `config.yaml`:

```yaml
# Document processing
document_reader:
  pdf:
    primary_reader: pdfplumber
    fallback_reader: pypdf2

# Chunking
chunking:
  strategy: hierarchical_legal
  min_chunk_size: 128
  max_chunk_size: 1024
  law_code:
    chunk_by: paragraph  # §
    include_context: true
  contract:
    chunk_by: article  # Články
    include_context: false

# Embeddings
embedding:
  model: BAAI/bge-m3  # 1024 dim, multilingual
  device: cuda  # cuda | cpu | mps
  add_hierarchical_context: true

# Retrieval
retrieval:
  hybrid_weights:
    semantic: 0.5
    keyword: 0.3
    structural: 0.2
  top_k: 20
  rerank_top_k: 5

# Compliance checking
compliance:
  mode: exhaustive  # Check all clauses
  gap_analysis: true
  risk_scoring: true

# LLM
llm:
  main_model: claude-sonnet-4-5-20250929
  sub_model: claude-3-5-haiku-20241022
  temperature: 0.1
```

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Indexing** | <5 min for 10k pages | Includes parsing + chunking + embedding |
| **Query** | <5s for clause check | Single clause compliance check |
| **Full scan** | <3 min | Complete contract compliance check |
| **Accuracy** | >90% issue detection | Validated against expert |
| **Precision** | >85% | Low false positive rate |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_document_reader.py

# Run with coverage
pytest --cov=advanced_sujbot2 --cov-report=html

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/ -v
```

---

## 📈 Development Roadmap

### Phase 1: Foundation (Weeks 1-2) ✅
- Document parsing with structure extraction
- Hierarchical legal chunking
- Multi-document vector store

### Phase 2: Retrieval (Weeks 3-4) 🔄
- Triple hybrid search
- Cross-document retrieval
- BGE-M3 embeddings

### Phase 3: Compliance (Weeks 5-6) 📋
- Compliance analyzer
- Gap detection
- Risk scoring

### Phase 4: Advanced (Weeks 7-8) 🚀
- Knowledge graph
- Graph-aware reranking
- Multi-hop reasoning

### Phase 5: Production (Weeks 9-10) 🎯
- Testing & QA
- Documentation
- Deployment

See [Implementation Roadmap](specs/12_implementation_roadmap.md) for details.

---

## 🤝 Contributing

1. Read the [Architecture Overview](specs/01_architecture_overview.md)
2. Pick a component from the specs
3. Follow the implementation guidelines
4. Write tests (>80% coverage)
5. Submit PR with documentation

---

## 📄 License

TBD

---

## 🙏 Acknowledgments

- **Anthropic** for Claude API
- **BAAI** for BGE-M3 embeddings
- **Facebook** for FAISS
- Czech legal system for inspiration 😅

---

## 📞 Contact

For questions or feedback: [contact info]

---

## 🔗 Useful Resources

- [FAISS Documentation](https://faiss.ai/)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [Claude API Docs](https://docs.anthropic.com/)
- [Czech Legal Code](https://www.zakonyprolidi.cz/)
