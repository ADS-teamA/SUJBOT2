# SUJBOT2 - Technical Specifications

Complete technical documentation for the SUJBOT2 legal compliance checking system.

---

## 📑 Table of Contents

### 🏛️ Core Architecture

**[01. Architecture Overview](01_architecture_overview.md)**
- System architecture and data flow
- Design principles
- Technology stack
- Deployment architecture
- Performance targets

### 📄 Document Processing

**[02. Document Reader](02_document_reader.md)**
- Legal document structure parsing
- Support for contracts, laws, regulations
- Reference extraction (§, články)
- Hierarchical structure detection
- Content classification

**[03. Chunking Strategy](03_chunking_strategy.md)**
- Hierarchical legal chunking
- Law code chunking (by paragraph §)
- Contract chunking (by article)
- Hybrid semantic chunking
- Metadata enrichment

### 🔍 Retrieval System

**[04. Embedding & Indexing](04_embedding_indexing.md)**
- BGE-M3 multilingual embeddings (1024 dim)
- Multi-document vector store
- FAISS index configuration
- Reference mapping
- Index persistence

**[05. Hybrid Retrieval](05_hybrid_retrieval.md)**
- Triple hybrid search
  - Semantic (embeddings)
  - Keyword (BM25)
  - Structured (hierarchy)
- Score fusion algorithms
- Query expansion

**[06. Cross-Document Retrieval](06_cross_document_retrieval.md)**
- Comparative retrieval
- Explicit reference matching
- Implicit semantic matching
- Structural matching

**[07. Reranking](07_reranking.md)**
- Multilingual cross-encoder
- Graph-aware reranking
- Legal precedence weighting
- Ensemble methods

### 🧠 Reasoning & Analysis

**[08. Query Processing](08_query_processing.md)**
- Legal question decomposition
- Compliance query types
- Gap analysis queries
- Risk assessment queries

**[09. Compliance Analyzer](09_compliance_analyzer.md)**
- Clause-level compliance checking
- Conflict detection
- Gap analysis
- Risk scoring
- Recommendation generation

**[10. Knowledge Graph](10_knowledge_graph.md)**
- Legal structure graph
- Cross-reference tracking
- Multi-hop reasoning
- Graph-based retrieval

### 🔌 Interfaces & Integration

**[11. API Interfaces](11_api_interfaces.md)**
- Python API
- REST API (future)
- Batch processing
- Webhook support

### 📅 Implementation

**[12. Implementation Roadmap](12_implementation_roadmap.md)**
- 10-week development plan
- Sprint breakdown
- Resource requirements
- Risk management
- Success metrics

---

## 🎯 Reading Guide

### For Product Managers
Start with:
1. [Architecture Overview](01_architecture_overview.md) - Understand the system
2. [Implementation Roadmap](12_implementation_roadmap.md) - Timeline and deliverables

### For Developers

**Implementing document processing**:
1. [Document Reader](02_document_reader.md)
2. [Chunking Strategy](03_chunking_strategy.md)

**Implementing retrieval**:
1. [Embedding & Indexing](04_embedding_indexing.md)
2. Hybrid Retrieval (05)
3. Cross-Document Retrieval (06)
4. Reranking (07)

**Implementing compliance features**:
1. Query Processing (08)
2. [Compliance Analyzer](09_compliance_analyzer.md)
3. Knowledge Graph (10)

### For Legal Experts
Focus on:
1. [Document Reader](02_document_reader.md) - How we parse legal structure
2. [Compliance Analyzer](09_compliance_analyzer.md) - How we detect issues
3. [Implementation Roadmap](12_implementation_roadmap.md) - When can I test it?

---

## 📊 Specification Status

| Spec | Status | Completeness |
|------|--------|--------------|
| 01. Architecture | ✅ Complete | 100% |
| 02. Document Reader | ✅ Complete | 100% |
| 03. Chunking | ✅ Complete | 100% |
| 04. Embedding & Indexing | ✅ Complete | 100% |
| 05. Hybrid Retrieval | ✅ Complete | 100% |
| 06. Cross-Document | ✅ Complete | 100% |
| 07. Reranking | ✅ Complete | 100% |
| 08. Query Processing | ✅ Complete | 100% |
| 09. Compliance Analyzer | ✅ Complete | 100% |
| 10. Knowledge Graph | ✅ Complete | 100% |
| 11. API Interfaces | ✅ Complete | 100% |
| 12. Roadmap | ✅ Complete | 100% |

**🎉 All 12 specifications complete! (~210 pages total)**

---

## 🔑 Key Concepts

### Legal Structure Hierarchy

```
Zákon (Law)
├── ČÁST I (Part)
│   ├── HLAVA I (Chapter)
│   │   ├── § 1 (Paragraph)
│   │   │   ├── (1) (Subsection)
│   │   │   ├── (2) (Subsection)
│   │   │   │   ├── a) (Letter)
│   │   │   │   └── b) (Letter)
│   │   └── § 2
│   └── HLAVA II
└── ČÁST II

Smlouva (Contract)
├── Preambule
├── Článek 1 (Article)
│   ├── 1.1 (Point)
│   ├── 1.2 (Point)
│   └── 1.3 (Point)
├── Článek 2
└── Přílohy (Annexes)
```

### Chunking Strategy

**Law**: One paragraph (§) = one chunk (unless too large → split by subsections)
**Contract**: One article = one chunk (unless too large → split by points)

### Multi-Document Retrieval

```
Query: "Najdi konflikty mezi smlouvou a zákonem"

1. Search in Contract Index → Contract chunks
2. Search in Law Index → Law chunks
3. Cross-match based on:
   - Explicit references (smlouva mentions "§89")
   - Semantic similarity
   - Structural patterns
4. Compare and detect conflicts
```

### Compliance Checking

```
For each Contract Clause:
1. Find related Law provisions
2. Extract requirements from Law
3. Check if Contract satisfies requirements
4. If not → classify issue:
   - CONFLICT: Directly contradicts
   - DEVIATION: Differs but may be acceptable
   - MISSING: Required provision absent
5. Score severity: CRITICAL | HIGH | MEDIUM | LOW
```

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **LLM** | Claude Sonnet 4.5 (main), Haiku 3.5 (sub) |
| **Embeddings** | BAAI/bge-m3 (1024 dim) |
| **Vector DB** | FAISS (local), Qdrant (production) |
| **Keyword Search** | BM25Okapi |
| **Reranker** | mmarco-mMiniLMv2-L12-H384-v1 |
| **Graph** | NetworkX |
| **PDF Processing** | pdfplumber, PyPDF2 |

---

## 📐 Design Principles

### 1. Legal Structure First
Don't chunk by tokens. Chunk by legal boundaries (§, články).

### 2. Multi-Document Architecture
Separate indices for contract vs law. Cross-document retrieval.

### 3. Compliance-Aware
Specialized query types and reasoning for legal compliance.

### 4. Graph-Enhanced
Use knowledge graph to track references and improve retrieval.

### 5. Production-Ready
Robust error handling, comprehensive testing, clear documentation.

---

## 🎓 Learning Resources

### RAG Fundamentals
- [Retrieval-Augmented Generation for Large Language Models](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag/)

### Legal AI
- [Legal Document AI: A Survey](https://arxiv.org/abs/2106.09326)
- [Graph RAG for Legal Norms](https://arxiv.org/html/2505.00039v1)

### Embeddings & Retrieval
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [FAISS Documentation](https://faiss.ai/)

---

## 🤔 FAQs

**Q: Why separate indices for contract and law?**
A: Enables filtering, cross-document comparison, and document-type-specific optimizations.

**Q: Why chunk by § rather than tokens?**
A: Legal citations reference §, not arbitrary text spans. Precise chunking = precise citations.

**Q: Can it handle Czech language well?**
A: Yes, BGE-M3 has excellent multilingual support including Czech.

**Q: What about hallucinations?**
A: Low temperature (0.1) + grounding in retrieved chunks + cross-encoder validation.

**Q: Can it scale to multiple contracts vs multiple laws?**
A: Yes, architecture supports N contracts × M laws. Currently optimized for 1×1.

---

## 📝 Notation & Conventions

### File Naming
- Specs: `NN_component_name.md` (e.g., `01_architecture_overview.md`)
- Code: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`

### Legal References
- Paragraph: `§89`, `§89 odst. 2`, `§89 odst. 2 písm. a)`
- Article: `Článek 5`, `Článek 5.2`
- Part: `Část II`
- Chapter: `Hlava III`

### Code Examples
All code examples are in Python 3.10+ with type hints.

---

## 🔄 Updating Specifications

When adding/modifying specs:
1. Follow existing structure and format
2. Include code examples
3. Add to this README's table of contents
4. Update status table
5. Link related specs

---

## 📧 Contact

For questions about specifications: [contact info]

For implementation questions: See [Implementation Roadmap](12_implementation_roadmap.md)

---

**Last Updated**: 2025-10-08
**Version**: 1.0
**Status**: Active Development
