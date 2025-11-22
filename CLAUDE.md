# CLAUDE.md - Development Guide

**SUJBOT2**: Production RAG system for legal/technical documents with multi-agent orchestration.

---

## 📚 Documentation Structure

**For detailed information, read these files:**
- [`README.md`](README.md) - Installation, quick start, user guide
- [`PIPELINE.md`](PIPELINE.md) - Complete 7-phase pipeline specification with research papers
- [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) - Docker configuration, hot reload, deployment
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) - Human-in-the-Loop system
- [`docs/SOTA_COMPLIANCE_IMPLEMENTATION.md`](docs/SOTA_COMPLIANCE_IMPLEMENTATION.md) - Compliance verification workflow
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) - Evaluation system
- [`docs/WEB_INTERFACE.md`](docs/WEB_INTERFACE.md) - Web UI features

---

## 🚀 Quick Start

```bash
# 1. Setup
# CRITICAL: API keys go in .env (NOT in config.json!)
cp .env.example .env
# Edit .env with your API keys (file is gitignored)

# 2. Index documents
uv run python run_pipeline.py data/document.pdf

# 3. Start full stack (Docker)
docker-compose up -d

# 4. Access
# Frontend: http://localhost:5173
# Backend: http://localhost:8000/docs

# See docs/DOCKER_SETUP.md for detailed instructions
```

---

## 🏗️ Architecture Overview

### Why This Architecture?

**Full-stack RAG system:**
- **Dual Storage Backends** - FAISS (fast, in-memory) OR PostgreSQL (production, pgvector + Apache AGE)
- **FastAPI Backend** - Multi-agent orchestration + 7-phase pipeline
- **React Frontend** - Real-time agent progress visualization

**Key Design Decisions:**
1. **Dual Backend Support** - FAISS for development/testing, PostgreSQL for production (user-selectable)
2. **Multi-layer embeddings** - 3 separate indexes (document/section/chunk) for 2.3x better retrieval
3. **Autonomous agents** - LLM-driven tool calling (NOT hardcoded workflows)
4. **Hierarchical summaries** - Document summaries built from section summaries (prevents context overflow)
5. **Hybrid search** - BM25 + dense embeddings + RRF fusion (+23% precision)

**Architecture Layers:**
```
User Query
    ↓
Orchestrator (routing + synthesis)
    ↓
Specialized Agents (extractor, classifier, compliance, etc.)
    ↓
RAG Tools (11 specialized tools for retrieval and analysis)
    ↓
Storage (FAISS or PostgreSQL: vectors, graph, checkpoints)
           ↓                    ↓
    Fast in-memory      Production database
    (development)       (concurrent access, ACID)
```

---

## ⚠️ CRITICAL CONSTRAINTS (NEVER CHANGE)

These are research-backed decisions. **DO NOT modify** without explicit approval.

### 1. SINGLE SOURCE OF TRUTH (SSOT)

**Principles:**
- **One canonical implementation** - Each feature has EXACTLY ONE authoritative version
- **No duplicate code** - Delete obsolete implementations immediately
- **No legacy code** - Remove unused/deprecated code
- **Clean root directory** - Tests in `tests/`, docs in `docs/`, scripts in `scripts/`
- **API keys in `.env` ONLY** - NEVER in config.json, NEVER in code, NEVER in git

**Why:** Prevents confusion, reduces maintenance burden, keeps codebase navigable, protects secrets.

### 2. AUTONOMOUS AGENTIC ARCHITECTURE

**CRITICAL: Agents MUST be LLM-driven, NOT hardcoded workflows!**

```python
# ❌ WRONG (Hardcoded)
def execute():
    step1 = call_tool_a()  # Predefined sequence
    step2 = call_tool_b()
    return synthesize(step1, step2)

# ✅ CORRECT (Autonomous)
def execute():
    return llm.run(
        system_prompt="You are an expert...",
        tools=[search, analyze, verify],  # LLM decides which to call
        messages=[user_query]
    )
```

**Principles:**
- LLM decides tool calling sequence autonomously
- No "step 1, step 2, step 3" logic in code
- System prompts guide behavior (NOT code)
- Exception: Orchestrator has routing logic

**Benefits:**
- 70% code reduction (~200 lines → ~60 lines per agent)
- LLM adapts to query complexity
- Emergent reasoning (discovers optimal strategies)
- Behavior changes via prompts (no code changes)

**Implementation:** All agents inherit from `BaseAgent.run_autonomous_tool_loop()` which handles:
1. LLM receives: system prompt + state + tool schemas
2. LLM decides: call tools OR provide final answer
3. Tool results fed back to LLM
4. Loop continues until final answer or max iterations

### 3. HIERARCHICAL DOCUMENT SUMMARIES

**NEVER pass full document text to LLM for document summary!**

```
Flow: Sections → Section Summaries → Document Summary
```

- **Why:** Prevents context overflow, handles 100+ page documents
- **Implementation:** PHASE 2 generates section summaries, then aggregates to document summary
- **Document summary content:** Describes what the document is about AND provides brief description of sections
- **Length:** 100-1000 chars (adaptive based on document complexity)
- **Fallback:** `"(Document summary unavailable)"` if section summaries fail

### 4. TOKEN-AWARE CHUNKING

- **Max tokens:** 512 (optimal for legal docs)
- **Tokenizer:** tiktoken (OpenAI text-embedding-3-large)
- **Research basis:** 512 tokens ≈ 500 chars (LegalBench-RAG)
- **Why tokens not chars:** Guarantees embedding model compatibility, handles Czech diacritics

**Changing this invalidates ALL vector stores!**

### 5. GENERIC SUMMARIES (Counterintuitive!)

- **Section summaries:** 300 chars
- **Document summaries:** 100-1000 chars (describes document and sections)
- **Style:** GENERIC (NOT expert terminology)
- **Research:** Reuter et al. (2024) - generic summaries improve retrieval
- **Prompts:** Stored in `prompts/` as `.txt` files (loaded as I/O)

### 6. SUMMARY-AUGMENTED CHUNKING (SAC)

- Prepend document summary during embedding
- Strip summaries during retrieval
- **Result:** -58% context drift (Anthropic, 2024)

### 7. MULTI-LAYER EMBEDDINGS

- **3 separate indexes** (document/section/chunk) - NOT merged
- **Result:** 2.3x essential chunks vs single-layer (Lima, 2024)

### 8. HYBRID SEARCH

- BM25 + Dense embeddings + RRF fusion
- **Result:** +23% precision vs dense-only
- **RRF k=60** (optimal)

### 9. NO COHERE RERANKING

- Cohere performs WORSE on legal docs
- **Use:** `ms-marco` or `bge-reranker` instead

### 10. AUTONOMOUS AGENT RESPONSES

**NEVER use hardcoded template responses!**

```python
# ❌ WRONG
if is_greeting(query):
    return "Hello! How can I help?"

# ✅ CORRECT
orchestrator.run(query)  # LLM generates contextual response
```

**Why:** Enables contextual awareness, eliminates brittle templates, allows adaptation.

---

## 🎯 Best Practices

### Agent Development

**Tool selection:**
- Use `search` for most queries (hybrid retrieval with optional expansion)
- Use specialized tools (graph_search, filtered_search) for specific needs
- Check tool documentation with `get_tool_help` when unsure

**Query expansion:**
- `num_expands=0` (default) - Speed
- `num_expands=1-2` - Recall-critical queries

**Graph boost:**
- Enable only for entity-focused queries (organizations, standards, regulations)

**Prompt caching:**
- Enable via `ENABLE_PROMPT_CACHING=true` for 90% cost savings on repeated queries

**Context pruning:**
- Keep conversation history under 50K tokens to prevent quadratic growth

### Pipeline Indexing

**Speed modes:**
- `SPEED_MODE=fast` - Development (fast iteration)
- `SPEED_MODE=eco` - Production (50% cheaper, overnight jobs)

**Knowledge graph:**
- `KG_BACKEND=neo4j` - Production
- `KG_BACKEND=simple` - Dev/testing

**Entity deduplication:**
- Use Layer 1 + Layer 3 (production balanced mode) for legal docs

**Validation:**
- Always run `pytest tests/` before committing pipeline changes

### Code Quality

**Type hints:**
- Required for all public APIs
- Verify with `mypy src/`

**Error handling:**
- Use graceful degradation (e.g., reranker unavailable → fall back to RRF)

**Logging:**
- Use appropriate levels (debug/info/warning/error)
- Avoid print statements

**Testing:**
- Write tests BEFORE implementing new features (TDD approach)

**Documentation:**
- Update PIPELINE.md if research constraints change

**Shared Utilities:**
- Extract common helper functions to `src/utils/` modules
- Examples: `faiss_utils.py` for FAISS operations, `api_clients.py` for API wrappers
- NEVER duplicate helper functions across scripts - use shared utilities
- DRY principle: Each function should have exactly ONE implementation

**Git workflow:**
- **ALWAYS use `gh` CLI** for pull requests (NOT curl or web interface)
- Command: `gh pr create --title "..." --body "..."`
- Benefits: Faster, scriptable, consistent formatting

**Model selection:**
- **Production:** `claude-sonnet-4-5` (highest quality)
- **Development:** `gpt-4o-mini` (best cost/performance: $0.15/$0.60 per 1M tokens)
- **Budget:** `claude-haiku-4-5` (fastest, cheapest)

### Performance

**Embedding cache:**
- Monitor hit rate with `embedder.get_cache_stats()` (target >80%)

**FAISS indexes:**
- Keep layer separation (DO NOT merge L1/L2/L3)

**Reranker loading:**
- Lazy load to reduce startup time (~2s savings)

**Token limits:**
- Use `max_total_tokens` parameter to prevent context overflow

### RAG-Specific Best Practices

**Document preprocessing:**
- Use Docling (yolox layout model) for accurate structure extraction
- Preserve hierarchy (document → sections → chunks)

**Chunking strategy:**
- Respect document structure (don't split mid-sentence)
- 512 tokens with 0 overlap (hierarchical overlap handles naturally)

**Embedding strategy:**
- Multi-layer: separate indexes for documents, sections, chunks
- SAC: prepend document summary during embedding

**Retrieval strategy:**
- Hybrid search (BM25 + dense + RRF) for best precision
- Graph boost for entity queries
- Reranking with ms-marco or bge-reranker

**Context assembly:**
- Include hierarchical metadata (document title, section title)
- Limit total tokens to prevent overflow
- Deduplicate chunks from same section

---

## 🔧 Configuration

**CRITICAL: Two-File Configuration System**

1. **`.env`** - Secrets (gitignored, NEVER commit!)
   - API keys (Anthropic, OpenAI, Voyage, Google)
   - Database passwords
   - JWT secret keys
   - Copy from `.env.example` and fill in your values

2. **`config.json`** - Settings (version-controlled, public)
   - Model selection
   - Pipeline parameters
   - Feature flags
   - NO secrets allowed!

```bash
# Setup API keys (REQUIRED)
cp .env.example .env
# Edit .env with your keys

# Configuration is already in config.json (no copy needed)
```

**Key decisions:**
- **API keys:** In `.env` file (ANTHROPIC_API_KEY or OPENAI_API_KEY)
- **Storage backend:** `faiss` (development, fast) or `postgresql` (production, ACID)
- **Embedding model:** `bge-m3` (free, local) or `text-embedding-3-large` (cloud)
- **Knowledge graph:** `neo4j` (production) or `simple` (dev)
- **Speed mode:** `fast` (dev) or `eco` (production)

### Storage Backend Selection

**Two backends available:**

1. **FAISS** (Default for development)
   - Fast in-memory vector search
   - Zero setup required
   - Perfect for testing and iteration
   - Files saved to `vector_db/` directory

2. **PostgreSQL** (Recommended for production)
   - Persistent database storage (pgvector extension)
   - Concurrent access from multiple agents
   - ACID transactions
   - Standard database backups
   - Requires PostgreSQL with pgvector

**How to choose:**

```json
// config.json
{
  "storage": {
    "backend": "faiss"       // Development: fast, no setup
    // OR
    "backend": "postgresql"  // Production: persistent, scalable
  }
}
```

**CLI override:**
```bash
# Index to PostgreSQL (override config.json)
python run_pipeline.py document.pdf --backend postgresql

# Index to FAISS (override config.json)
python run_pipeline.py document.pdf --backend faiss
```

**PostgreSQL setup:**
```bash
# 1. Set DATABASE_URL in .env
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"

# 2. Configure in config.json
{
  "storage": {
    "backend": "postgresql"
  }
}

# 3. Index documents
python run_pipeline.py document.pdf
```

**See `config.json` for all options.**

### Migrating from Previous Versions

If upgrading from a version that used `config.json` for API keys:

1. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

2. **Move API keys** from `config.json` to `.env`:
   ```bash
   # Old location (config.json) - REMOVE THIS SECTION
   {
     "api_keys": {
       "anthropic_api_key": "sk-ant-...",  # ❌ REMOVE
       "openai_api_key": "sk-..."          # ❌ REMOVE
     }
   }

   # New location (.env) - ADD YOUR KEYS HERE
   ANTHROPIC_API_KEY=sk-ant-...  # ✅ CORRECT
   OPENAI_API_KEY=sk-...          # ✅ CORRECT
   ```

3. **Remove `api_keys` section** from `config.json` (now ignored):
   - The old `api_keys: {}` field is no longer used
   - It's safe to remove it entirely from your config.json

4. **Verify:**
   ```bash
   # .env should NOT be tracked by git
   git status  # Should NOT show .env

   # config.json should have NO secrets
   grep -i "api_key\|password\|secret" config.json  # Should return nothing
   ```

5. **Test:**
   ```bash
   # Should fail with clear error if API keys missing
   python run_pipeline.py --help

   # Should work after setting keys in .env
   python run_pipeline.py document.pdf
   ```

**Why this change?**
- **Security:** API keys were in version-controlled files (bad practice)
- **Best practice:** Secrets in `.env` (gitignored), settings in `config.json`
- **Standards:** Follows 12-factor app methodology

---

## 📖 Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS, reranking, 500-char chunks
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending (-58% drift)
5. **HybridRAG** (2024) - Graph boosting (+8% factual correctness)
6. **HyDE** (Gao et al., 2022) - Hypothetical Document Embeddings (+15-30% recall for zero-shot queries)

---

## 📚 Code Style

```bash
# Formatting
uv run black src/ tests/ --line-length 100
uv run isort src/ tests/ --profile black
```

**Conventions:**
- Classes: `PascalCase`
- Functions/vars: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Docstrings: Google style
- Type hints: Required for public APIs

---

## 🔍 Where to Find Things

**See README.md for:**
- File locations (backend, frontend, agents, tools)
- Installation instructions
- Testing commands

**See docs/DOCKER_SETUP.md for:**
- Docker architecture
- Hot reload workflow
- Debugging commands
- Common operations

**See PIPELINE.md for:**
- 7-phase pipeline details
- Research paper references
- Implementation specifics

**See docs/SOTA_COMPLIANCE_IMPLEMENTATION.md for:**
- Requirement-first compliance approach
- Plan-and-Solve pattern
- Gap classification

---

**Last Updated:** 2025-11-22
**Version:** PHASE 1-7 COMPLETE + Dual Backend Support + Multi-Agent + HITL + Docker Web UI

**Notes:**
- Configuration: SSOT in `config.json` (strict validation, no defaults)
- Storage: **Dual backend support** - FAISS (development) OR PostgreSQL (production, user-selectable)
- Agents: 7 autonomous agents (orchestrator + 6 specialized)
- Tools: 11 RAG tools (search, retrieval, analysis, and metadata tools)
- Hybrid Search: Works seamlessly with both FAISS and PostgreSQL backends
