# SUJBOT2 - Production RAG System for Legal/Technical Documents

Research-based RAG system optimized for legal and technical documentation with 7-phase pipeline and **multi-agent AI framework**.

**Status:** PHASE 1-7 COMPLETE + **MULTI-AGENT UPGRADE** ✅ (2025-11-11)

## 🆕 NEW: Multi-Agent System (v2.0)

SUJBOT2 has been upgraded to a **research-backed multi-agent framework** achieving:
- ✅ **90% cost savings** via 3-level prompt caching (Harvey AI case study)
- ✅ **8 specialized agents** for higher quality (Orchestrator, Extractor, Classifier, Compliance, Risk Verifier, Citation Auditor, Gap Synthesizer, Report Generator)
- ✅ **State persistence** with PostgreSQL checkpointing
- ✅ **Full observability** with LangSmith integration

**Quick Start:**
```bash
# New multi-agent command
uv run python -m src.multi_agent.runner --query "Verify GDPR compliance"

# Interactive mode
uv run python -m src.multi_agent.runner --interactive
```

**Migrating from v1.x single-agent?** → See [**MIGRATION_GUIDE.md**](MIGRATION_GUIDE.md)

**Architecture details** → See [**MULTI_AGENT_STATUS.md**](MULTI_AGENT_STATUS.md)

## 📚 Interactive Documentation

**🌐 Live Documentation:** [https://ads-teama.github.io/SUJBOT2/](https://ads-teama.github.io/SUJBOT2/)

Explore our visual, interactive pipeline documentation:
- 📥 **[Indexing Pipeline](https://ads-teama.github.io/SUJBOT2/indexing_pipeline.html)** - Phase 1-5: Document → Vector Store
- 💬 **[User Search Pipeline](https://ads-teama.github.io/SUJBOT2/user_search_pipeline.html)** - Phase 7: Query → AI Answer (14 Tools)
- 🗓️ **[4-Week Roadmap](https://ads-teama.github.io/SUJBOT2/roadmap.html)** - Team plans for pipeline optimization

---

## 🎯 Overview

Production-ready RAG system based on 4 research papers implementing state-of-the-art techniques:
- **LegalBench-RAG** (Pipitone & Alami, 2024)
- **Summary-Augmented Chunking** (Reuter et al., 2024)
- **Multi-Layer Embeddings** (Lima, 2024)
- **NLI for Legal Contracts** (Narendra et al., 2024)

### Key Features

**Pipeline (PHASE 1-6):**
- **PHASE 1:** Smart hierarchy extraction (Docling, font-size classification)
- **PHASE 2:** Generic summary generation (150 chars, proven better than expert summaries)
- **PHASE 3:** RCTS chunking (500 chars) + SAC (58% DRM reduction)
- **PHASE 4:** Multi-layer indexing (3 separate FAISS indexes)
- **PHASE 5:** Hybrid search (BM25+Dense+RRF) + Knowledge graph + Cross-encoder reranking + Query expansion
- **PHASE 6:** Context assembly with citations

**Agent (PHASE 7) - Multi-Agent System:**
- **8 specialized agents** (Orchestrator, Extractor, Classifier, Compliance, Risk Verifier, Citation Auditor, Gap Synthesizer, Report Generator)
- **17 existing tools** integrated via adapter pattern (zero changes needed)
- **3-level prompt caching** for 90% cost savings (regulatory docs + contract templates + system prompts)
- **PostgreSQL checkpointing** for state persistence and recovery
- **LangSmith observability** for full workflow tracing
- **Adaptive routing** with 3 workflow patterns (Simple, Standard, Complex)

---

## 🚀 Quick Start

### ⚠️ Breaking Change (v2.0 - HybridChunker)

**If upgrading from v1.x:**
- Chunking strategy changed to **HybridChunker** (token-aware, 512 tokens ≈ 500 chars)
- Layout model changed to **HERON** (+23.9% accuracy improvement)
- **All existing vector stores must be re-indexed**
- Run: `rm -rf vector_db/* && uv run python run_pipeline.py data/`
- Estimated time: 15-30 minutes for 10k chunks

**Why this change:**
- Token-aware chunking guarantees embedding model compatibility
- HERON provides best accuracy for complex legal layouts
- Preserves research constraints (512 tokens ≈ 500 chars)

---

### Prerequisites

- Python 3.10+
- `uv` package manager ([installation](https://docs.astral.sh/uv/))
- API keys: `ANTHROPIC_API_KEY` and optionally `OPENAI_API_KEY`

### Installation

**macOS/Linux:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys
```

**Windows:**
```bash
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# IMPORTANT: Install PyTorch FIRST (prevents DLL errors)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
uv sync

# Configure (use cloud embeddings for Windows)
copy .env.example .env
# Edit .env and set EMBEDDING_MODEL=text-embedding-3-large
```

**API Keys (.env):**
```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required
OPENAI_API_KEY=sk-...         # Optional (for OpenAI embeddings)
LLM_MODEL=gpt-5-nano          # For summaries & agent
EMBEDDING_MODEL=text-embedding-3-large  # Windows
# EMBEDDING_MODEL=bge-m3      # macOS M1/M2/M3 (local, FREE, GPU-accelerated)
```

**For detailed platform-specific instructions, see [INSTALL.md](INSTALL.md).**

---

## 📖 Usage

### 1. Index Documents

```bash
# Single document
uv run python run_pipeline.py data/document.pdf

# Batch processing
uv run python run_pipeline.py data/regulace/

# Fast mode (default) - 2-3 min, full price
uv run python run_pipeline.py data/document.pdf

# Eco mode - 15-30 min, 50% cheaper (overnight bulk indexing)
# Set SPEED_MODE=eco in .env
```

**Output:** Vector store in `output/<document_name>/phase4_vector_store/`

### 2. Run RAG Agent

```bash
# Launch interactive agent (14 tools)
uv run python -m src.agent.cli

# With specific vector store
uv run python -m src.agent.cli --vector-store output/my_doc/phase4_vector_store

# Debug mode
uv run python -m src.agent.cli --debug
```

**Agent Commands:**
- `/help` - Show available commands and tools
- `/stats` - Show tool usage, conversation stats, session costs
- `/config` - Show current configuration
- `/clear` - Clear conversation history
- `/exit` - Exit agent

**Example Session:**
```
🤖 RAG Agent CLI (14 tools, Claude SDK)
📚 Loaded vector store: output/safety_manual/phase4_vector_store
💰 Session cost: $0.0000 (0 tokens)

You: What are the safety procedures for reactor shutdown?

Agent: [Uses 3 tools: document_search → section_search → extract_text]
Based on the safety manual, reactor shutdown follows these procedures:

1. **Normal Shutdown** (Section 4.2):
   - Reduce power to 50% over 30 minutes
   - Insert control rods gradually...

[Citations: Section 4.2, Page 45-47]

💰 Session cost: $0.0234 (12,450 tokens) | 📦 Cache: 8,500 tokens read (90% saved)
```

### 3. Run Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific phase
uv run pytest tests/test_phase4_indexing.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html
```

---

## 🏗️ Architecture

### Complete Pipeline Flow

```
Document (PDF/DOCX)
    ↓
[PHASE 1] Hierarchy Extraction
    ├─ Docling conversion (OCR: Czech/English)
    ├─ Font-size based classification
    └─ HierarchicalChunker (parent-child relationships)
    ↓
[PHASE 2] Summary Generation
    ├─ gpt-4o-mini or gpt-5-nano (~$0.001 per doc)
    ├─ Generic summaries (150 chars) - NOT expert
    └─ Document + section summaries
    ↓
[PHASE 3] Multi-Layer Chunking + SAC
    ├─ Layer 1: Document (1 chunk, summary)
    ├─ Layer 2: Sections (N chunks, summaries)
    └─ Layer 3: RCTS 500 chars + SAC (PRIMARY)
    ↓
[PHASE 4] Embedding + FAISS Indexing
    ├─ text-embedding-3-large (3072D) or bge-m3 (1024D)
    ├─ 3 separate FAISS indexes (IndexFlatIP)
    └─ Cosine similarity search
    ↓
[PHASE 5] Hybrid Search + Knowledge Graph + Reranking + Query Expansion
    ├─ Query expansion (optional, num_expands=0-5)
    ├─ BM25 + Dense retrieval + RRF fusion
    ├─ Entity/relationship extraction (NetworkX)
    └─ Cross-encoder reranking (NOT Cohere - hurts legal docs)
    ↓
[PHASE 6] Context Assembly
    ├─ Strip SAC summaries
    ├─ Concatenate chunks
    └─ Add citations with section paths
    ↓
[PHASE 7] Agent with 27 Tools
    ├─ Interactive CLI (Claude SDK)
    ├─ 12 basic tools (fast search)
    ├─ 9 advanced tools (quality retrieval)
    ├─ 6 analysis tools (deep understanding)
    └─ Cost tracking + prompt caching
```

### 27 Agent Tools

**Basic Tools (Fast, <1s):**
- `search` - Unified hybrid search with optional query expansion (num_expands parameter)
- `document_search` - Find relevant documents
- `section_search` - Search within sections
- `chunk_search` - Semantic chunk search
- `keyword_search` - Exact keyword matching
- ... (6 more)

**Advanced Tools (Quality, 1-3s):**
- `hybrid_search` - BM25 + Dense + RRF
- `graph_query` - Knowledge graph traversal
- `reranked_search` - Cross-encoder reranking
- `multi_query` - Query decomposition
- ... (5 more)

**Analysis Tools (Deep, 3-10s):**
- `compare_documents` - Cross-document analysis
- `summarize_topic` - Topic-based summarization
- `extract_entities` - Entity recognition
- `trace_relationships` - Relationship mapping
- ... (2 more)

---

## 📊 Performance Metrics

Based on research and testing:

| Metric | Baseline | Our Pipeline | Improvement |
|--------|----------|-------------|-------------|
| **Hierarchy depth** | 1 | 4 | **+300%** |
| **Precision@1** | 2.40% | 6.41% | **+167%** |
| **DRM Rate** | 67% | 28% | **-58%** |
| **Essential chunks** | 16% | 38% | **+131%** |
| **Recall@64** | 35% | 62% | **+77%** |

---

## 🔬 Research Foundation

### Critical Implementation Rules (DO NOT CHANGE)

**Evidence-based decisions:**

1. **RCTS > Fixed-size chunking** (LegalBench-RAG)
   - Chunk size: **500 chars** (optimal, +167% Precision@1)
   - Overlap: 0 (RCTS handles naturally)

2. **Generic > Expert summaries** (Reuter et al.)
   - Summary length: **150 chars**
   - Style: **Generic** (NOT expert - counterintuitive but proven)

3. **SAC reduces DRM by 58%** (Reuter et al.)
   - Prepend document summary to each chunk
   - Baseline DRM: 67% → SAC DRM: 28%

4. **Multi-layer embeddings** (Lima)
   - 3 separate FAISS indexes
   - 2.3x essential chunks

5. **No Cohere reranking** (LegalBench-RAG)
   - Cohere worse than no reranking on legal docs
   - Use cross-encoder instead

6. **Dense > Sparse for legal docs** (Reuter et al.)
   - Better precision/recall than BM25-only
   - Hybrid (BM25+Dense+RRF) best overall

---

## 💻 Configuration

### Load from .env (Recommended)

```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig

# Load all settings from .env
config = IndexingConfig.from_env()
pipeline = IndexingPipeline(config)

# Override specific settings
config = IndexingConfig.from_env(
    enable_knowledge_graph=True,
    enable_hybrid_search=True,
    speed_mode="eco"  # 50% cheaper for bulk indexing
)

# Index document
result = pipeline.index_document("document.pdf")
```

### Key .env Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Models
LLM_MODEL=gpt-5-nano                    # Summaries & agent
EMBEDDING_MODEL=text-embedding-3-large  # Windows
# EMBEDDING_MODEL=bge-m3                # macOS (local, FREE)

# Pipeline
SPEED_MODE=fast                         # fast or eco (50% savings)
ENABLE_HYBRID_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_PROMPT_CACHING=true              # Anthropic only (90% savings)

# OCR
OCR_LANGUAGE=ces,eng                    # Czech + English
```

### Optimal Settings (Research-Based)

```python
IndexingConfig(
    # PHASE 1: Hierarchy
    enable_smart_hierarchy=True,
    ocr_language=["ces", "eng"],

    # PHASE 2: Summaries
    generate_summaries=True,
    summary_model="gpt-5-nano",
    summary_max_chars=150,
    summary_style="generic",  # NOT expert!

    # PHASE 3: Chunking
    chunk_size=500,           # Optimal per research
    enable_sac=True,          # 58% DRM reduction

    # PHASE 4: Embedding
    embedding_model="text-embedding-3-large",

    # PHASE 5: Advanced Features
    enable_hybrid_search=True,
    enable_knowledge_graph=True,

    # Performance
    speed_mode="fast",        # or "eco" for bulk indexing
)
```

---

## 📁 Project Structure

```
MY_SUJBOT/
├── src/
│   ├── indexing_pipeline.py           # Main orchestrator (PHASE 1-6)
│   ├── config.py                      # Central config (load from .env)
│   ├── docling_extractor_v2.py        # PHASE 1: Hierarchy extraction
│   ├── summary_generator.py           # PHASE 2: Generic summaries
│   ├── multi_layer_chunker.py         # PHASE 3: Chunking + SAC
│   ├── embedding_generator.py         # PHASE 4: Embeddings
│   ├── faiss_vector_store.py          # PHASE 4: FAISS indexes
│   ├── hybrid_search.py               # PHASE 5: BM25+Dense+RRF
│   ├── graph/                         # PHASE 5: Knowledge graph
│   │   ├── entity_extractor.py
│   │   └── graph_store.py
│   └── agent/                         # PHASE 7: RAG Agent
│       ├── cli.py                     # Interactive CLI
│       ├── config.py                  # Agent configuration
│       └── tools/                     # 27 specialized tools
├── tests/                             # Comprehensive test suite
├── data/                              # Input documents
├── output/                            # Pipeline outputs
├── vector_db/                         # Central vector database
├── docs/                              # Documentation
├── run_pipeline.py                    # Pipeline entry point
├── CLAUDE.md                          # Development guidelines
├── INSTALL.md                         # Platform-specific installation
├── PIPELINE.md                        # Complete pipeline spec
└── .env.example                       # Environment template
```

---

## 📖 Documentation

### Core Documentation

- **[INSTALL.md](INSTALL.md)** - Platform-specific installation (Windows/macOS/Linux)
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and project instructions
- **[PIPELINE.md](PIPELINE.md)** - Complete pipeline specification with research

### User Guides

- **[Agent CLI Guide](docs/agent/README.md)** - RAG Agent CLI documentation
- **[macOS Quick Start](docs/how-to-run-macos.md)** - Quick start for macOS users
- **[Vector DB Management](docs/vector-db-management.md)** - Central database tools

### Advanced Topics

- **[Cost Tracking](docs/cost-tracking.md)** - API cost monitoring and optimization
- **[Cost Optimization](docs/development/cost-optimization.md)** - Detailed cost analysis
- **[Batching Optimizations](docs/development/batching-optimizations.md)** - Performance guide

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Test specific phase
uv run pytest tests/test_phase4_indexing.py -v

# Test agent
uv run pytest tests/agent/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Single test
uv run pytest tests/agent/test_validation.py::test_api_key_validation -v
```

---

## ⚡ Performance Tips

### Background Processing

```bash
# Run pipeline in background (long-running)
nohup uv run python run_pipeline.py data/ > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log

# Agent in background
nohup uv run python -m src.agent.cli > agent.log 2>&1 &
```

### Cost Optimization

**Speed Modes:**
- `speed_mode="fast"` (default): 2-3 min, full price (ThreadPoolExecutor)
- `speed_mode="eco"`: 15-30 min, 50% cheaper (OpenAI Batch API)

```python
# For overnight bulk indexing
config = IndexingConfig.from_env(speed_mode="eco")
```

**Prompt Caching (Anthropic only):**
```bash
# .env
ENABLE_PROMPT_CACHING=true  # 90% cost reduction on cached tokens
```

**Example savings:**
```
Session cost: $0.0234 (12,450 tokens) | Cache: 8,500 tokens read (90% saved)
```

---

## 🌍 Platform Support

**Tested Platforms:**
- macOS (Apple Silicon M1/M2/M3) - Recommended for local embeddings
- Linux (Ubuntu 20.04+) - Production deployment
- Windows 10/11 - Cloud embeddings recommended

**Embedding Model Selection:**
- **Windows:** `text-embedding-3-large` (cloud) - avoids PyTorch DLL issues
- **macOS M1/M2/M3:** `bge-m3` (local, FREE, GPU-accelerated)
- **Linux GPU:** `bge-m3` (local)
- **Linux CPU:** `text-embedding-3-large` (cloud)

---

## ⚠️ Requirements

- **Python:** >=3.10
- **uv:** Latest version (package manager)
- **Memory:** 8GB+ recommended
- **API Keys:** ANTHROPIC_API_KEY (required), OPENAI_API_KEY (optional)
- **GPU:** Optional (for local embeddings on macOS/Linux)

---

## 🙏 Acknowledgments

Based on research from:
- Pipitone & Alami (LegalBench-RAG, 2024)
- Reuter et al. (Summary-Augmented Chunking, 2024)
- Lima (Multi-Layer Embeddings, 2024)
- Narendra et al. (NLI for Legal Contracts, 2024)

---

## 📄 License

MIT License

---

**Status:** PHASE 1-7 COMPLETE ✅
**Last Updated:** 2025-10-26
