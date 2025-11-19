# CLAUDE.md - Navigation Guide

**SUJBOT2**: Production RAG system for legal/technical docs. 7-phase pipeline + multi-agent orchestration + Human-in-the-Loop clarifications + Docker-based web interface.

**Status:** ORCHESTRATOR-CENTRIC ARCHITECTURE ✅ (2025-11-13)

---

## 🚀 Quick Start

**Architecture:** Full-stack application running in Docker (PostgreSQL + FastAPI Backend + React Frontend)

**Read these files for detailed information:**
- [`README.md`](README.md) - User guide, installation, quick start
- [`PIPELINE.md`](PIPELINE.md) - Complete pipeline specification with research papers
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) - Human-in-the-Loop clarification system
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) - Benchmark evaluation system
- [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) - Docker configuration and deployment
- [`docs/WEB_INTERFACE.md`](docs/WEB_INTERFACE.md) - Web UI features and usage
- [`docs/LANGUAGE_SUPPORT.md`](docs/LANGUAGE_SUPPORT.md) - Multilingual BM25 support (Czech + 24 languages)
- Visual docs: [`indexing_pipeline.html`](indexing_pipeline.html), [`user_search_pipeline.html`](user_search_pipeline.html)

**Docker Commands (primary interface):**
```bash
# 1. Setup (first time only)
cp config.json.example config.json
# Edit config.json with your API keys

# 2. Index documents (before first use)
uv run python run_pipeline.py data/document.pdf

# 3. Start full stack (backend + frontend + PostgreSQL)
docker-compose up -d

# OR use convenience script
./start_web.sh

# 4. Access web interface
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# 5. View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# 6. Stop services
docker-compose down

# Development: Run tests
uv run pytest tests/ -v
```

---

## 🔄 Storage Migration: FAISS → PostgreSQL

**Current Storage Backend:** PostgreSQL with pgvector (`config.json`: `"storage.backend": "postgresql"`)

**Why PostgreSQL?**
- ✅ ACID transactions and atomic operations
- ✅ Concurrent access without file locking issues
- ✅ Standard backup/recovery (pg_dump, WAL archiving)
- ✅ Production-ready: No shared file mounts across containers
- ✅ Integrated hybrid search: pgvector (dense) + tsvector (sparse, replaces BM25)

### Migration from Legacy FAISS (✅ COMPLETED 2025-11-12)

**Migration Status:** All FAISS data successfully migrated to PostgreSQL

**Migrated Data:**
- ✅ Layer 1: **5 documents**
- ✅ Layer 2: **4,213 sections**
- ✅ Layer 3: **5,650 chunks**
- ✅ **Total: 9,868 vectors**
- ✅ Vector similarity search verified working

**If you need to re-run migration** (e.g., after adding new documents to FAISS):

```bash
# 1. Ensure PostgreSQL is running
docker-compose up -d postgres

# 2. Run migration from backend container (required for Docker networking)
docker-compose exec backend uv run python scripts/migrate_faiss_to_postgres.py \
  --faiss-dir /app/vector_db/ \
  --db-url postgresql://postgres:PASSWORD@postgres:5432/sujbot \
  --batch-size 500 \
  --verify

# 3. Verify migration
docker-compose exec postgres psql -U postgres -d sujbot -c \
  "SELECT 'L1:', COUNT(*) FROM vectors.layer1 UNION ALL
   SELECT 'L2:', COUNT(*) FROM vectors.layer2 UNION ALL
   SELECT 'L3:', COUNT(*) FROM vectors.layer3;"

# 5. Optional: Backup old FAISS data
mv vector_db vector_db.backup.$(date +%Y%m%d)

# 6. Start application (will use PostgreSQL)
docker-compose up -d
```

**Post-Migration:**
- Application automatically uses PostgreSQL (configured in `config.json`)
- Old `vector_db/` directory no longer needed (but keep as backup until verified)
- All searches now use: pgvector (cosine similarity) + tsvector (full-text search) + RRF fusion

**Troubleshooting:**
- If migration fails: Check PostgreSQL logs (`docker-compose logs postgres`)
- If "pgvector extension not found": Ensure Docker PostgreSQL image includes pgvector (see `docker/postgres/Dockerfile`)
- If connection errors: Verify `DATABASE_URL` in `.env` matches PostgreSQL container

---

## ⚠️ CRITICAL CONSTRAINTS (NEVER CHANGE)

These are research-backed decisions. **DO NOT modify** without explicit approval:

### -1. SINGLE SOURCE OF TRUTH (SSOT) & CODE HYGIENE (MANDATORY) 🧹

**CRITICAL: Maintain SSOT principles and eliminate duplicate/legacy implementations!**

**SSOT Principles:**
1. **One canonical implementation** - Each feature/component has EXACTLY ONE authoritative implementation
2. **No duplicate code** - If you find two implementations of the same functionality, remove the obsolete one
3. **No legacy code** - If you encounter unused/deprecated code, delete it immediately
4. **No test files in root** - All tests belong in `tests/` directory
5. **No documentation in root** - Detailed docs belong in `docs/` directory
6. **No utility scripts in root** - Migration/utility scripts belong in `scripts/` directory

**Root Directory Rules:**
```
✅ ALLOWED in root:
- Config files (.env, .gitignore, config.json, docker-compose.yml)
- Main entry points (run_pipeline.py, run_benchmark.py, start_web.sh)
- Core documentation (README.md, CLAUDE.md, PIPELINE.md)
- Python packaging (pyproject.toml, uv.lock, pytest.ini)

❌ FORBIDDEN in root:
- Test files (test_*.py, *_test.sh, test_*.sh)
- Test results (RESULTS.md, TEST_OUTPUT.md)
- Detailed documentation (BENCHMARK.md, DOCKER_SETUP.md, etc.) → move to docs/
- Utility scripts (rebuild_*.py, verify_*.py, migrate_*.py) → move to scripts/
- Temporary files, logs, debug outputs
```

**Duplicate Detection & Removal Protocol:**
When you encounter ANY of these patterns:
1. **Two implementations of same feature** - Compare timestamps, choose newer, delete older
2. **Legacy tool/agent** - If replaced by new version, delete old immediately
3. **Unused imports/functions** - Remove if not referenced anywhere
4. **Commented-out code blocks** - Delete (git history preserves it)
5. **Multiple config files for same thing** - Consolidate to SSOT (config.json)

**Examples of violations to fix immediately:**
```
❌ BAD: src/agent/tools/old_search.py + src/agent/tools/tier1_basic.py both have search
→ FIX: Keep tier1_basic.py (newer), delete old_search.py

❌ BAD: test_fixes.py in root
→ FIX: Move to tests/ or delete if temporary

❌ BAD: BENCHMARK.md in root
→ FIX: Move to docs/BENCHMARK.md

❌ BAD: rebuild_vector_db.py in root
→ FIX: Move to scripts/rebuild_vector_db.py
```

**Why this matters:**
- Prevents confusion about which implementation to use
- Reduces maintenance burden (update one place, not three)
- Keeps codebase navigable and professional
- Eliminates "zombie code" that wastes context window

**Enforcement:** Whenever you read/modify code and spot duplicate/legacy implementations, **DELETE THEM IMMEDIATELY**. Do not wait for explicit user request.

---

### 0. AUTONOMOUS AGENTIC ARCHITECTURE (MANDATORY) 🤖

**CRITICAL: Agents MUST be autonomous LLM-driven, NOT hardcoded workflows!**

```
❌ WRONG (Hardcoded):
class ComplianceAgent:
    def execute():
        step1 = call_tool_a()  # Hardcoded sequence
        step2 = call_tool_b()
        return synthesize(step1, step2)

✅ CORRECT (Autonomous):
class ComplianceAgent:
    def execute():
        # LLM autonomously decides which tools to call and when
        return llm.run(
            system_prompt="You are compliance expert...",
            tools=[search, graph_search, assess_confidence, ...],
            messages=[user_query]
        )
```

**Principles:**
1. **LLM decides tool calling** - Agent provides system prompt + tools, LLM autonomously calls them
2. **No hardcoded flows** - No predefined "step 1, step 2, step 3" logic
3. **Tools are capabilities** - Agent defines WHAT tools are available, LLM decides HOW to use them
4. **Orchestrator exception** - ONLY Orchestrator has hardcoded logic (routing), all other agents are autonomous
5. **System prompts guide behavior** - Control agent behavior via prompts, NOT code

**Technical Implementation:**

All agents inherit from `BaseAgent` which provides the autonomous tool calling loop:

```python
# src/multi_agent/core/agent_base.py
async def _run_autonomous_tool_loop(
    self,
    system_prompt: str,
    state: Dict[str, Any],
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Core autonomous agentic behavior:
    1. LLM sees state/query + available tools
    2. LLM decides to call tools or provide final answer
    3. Tool results fed back to LLM
    4. Loop continues until LLM provides final answer
    """
```

**How it works:**
1. Agent calls `_run_autonomous_tool_loop()` with system prompt and state
2. Loop builds context from state (query + previous agent outputs)
3. LLM receives: system prompt + context + tool schemas
4. LLM decides: call tools (→ execute tools → feed results back) OR provide final answer
5. Loop continues until LLM returns final answer or max_iterations reached
6. Result contains: final_answer, tool_calls history, iterations count

**Tool Schema Conversion:**
- `ToolAdapter.get_tool_schema()` converts Pydantic schemas to LLM format (Anthropic/OpenAI)
- Agents automatically get schemas for their configured tools from `config.json`
- LLM sees tool name, description, and input schema for each available tool

**Prompt Design Pattern:**
Every agent prompt (in `prompts/agents/`) follows this structure:
1. **ROLE** - Agent's responsibility
2. **DOMAIN KNOWLEDGE** - Risk categories, compliance frameworks, etc.
3. **AVAILABLE TOOLS (use autonomously as needed)** - Explicit tool listing
4. **AUTONOMOUS WORKFLOW** - "Typical approach" guidance (NOT prescription)
5. **INTERNAL TODO LIST (LangGraph Pattern)** - Internal task tracking for long-running operations
6. **IMPORTANT** - Reinforces autonomy: "YOU decide which tools to use"
7. **FINAL ANSWER FORMAT** - Expected output structure

**Internal Task Tracking (LangGraph Pattern):**
Agents with multi-step workflows maintain internal todo lists to track progress during autonomous execution:
- **ComplianceAgent**: Tracks sequential requirement verification (10-20+ requirements)
- **RequirementExtractorAgent**: Tracks 6-phase extraction workflow (identification → retrieval → alignment → atomization → classification → checklist generation)
- **GapSynthesizerAgent**: Tracks 8-phase gap analysis (context → expected content → coverage mapping → classification → comparison → relationship analysis → prioritization → recommendations)

**Pattern structure:**
```
INTERNAL TODO LIST (LangGraph Pattern - Track Your Progress):

Phase 1: Setup
- [ ] Task 1
- [ ] Task 2

Phase 2: Processing (repeat for each item)
- [ ] Item 1: Subtask A
- [ ] Item 1: Subtask B
- [ ] Item 2: Subtask A
...

Phase 3: Aggregation
- [ ] Final task

COMPLETE: [X/Total] items processed
```

**Benefits:**
- Maintains context during 30-60s operations with 10+ sequential steps
- Prevents skipping requirements/phases accidentally
- Debugging: Internal checkboxes reveal WHERE agent got stuck if execution fails
- LLM marks [x] completed tasks in reasoning, tracks progress naturally

**Benefits vs Hardcoded:**
- ✅ ~70% code reduction per agent (~200 lines → ~60 lines)
- ✅ LLM adapts to query complexity (simple → calls 1-2 tools, complex → calls 5+ tools)
- ✅ Emergent reasoning (LLM discovers tool combinations we didn't explicitly program)
- ✅ Behavior changes via prompts (no code changes needed)
- ✅ Single implementation in BaseAgent (changes propagate automatically)
- ❌ Requires proper tool schemas and clear prompts
- ❌ LLM cost per agent execution (tool calling loop)

**Files:**
- `src/multi_agent/core/agent_base.py` - Autonomous agent base class with `_run_autonomous_tool_loop()`
- `src/multi_agent/tools/adapter.py` - Tool schema conversion (`get_tool_schema()`)
- `src/multi_agent/agents/*.py` - All 7 agents use autonomous pattern (extractor, classifier, compliance, risk_verifier, citation_auditor, gap_synthesizer, report_generator)
- `prompts/agents/*.txt` - System prompts guide autonomous behavior

**Testing:**
```bash
# Test single autonomous agent (local dev)
uv run python test_autonomous.py

# Test full multi-agent workflow via web interface
# 1. Start services: docker-compose up -d
# 2. Open http://localhost:5173
# 3. Send query: "What are GDPR compliance requirements?"
# 4. Observe agent progress in real-time

# Test backend API directly
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Test query", "conversation_id": "test", "model": null}'
```

**Why:** True agentic system enables emergent behavior, better reasoning, and flexibility without code changes. LLM can discover optimal tool calling strategies we didn't explicitly program.

### 1. Hierarchical Document Summary (MANDATORY)
```
Flow: Sections → Section Summaries (PHASE 3B) → Document Summary
```
- **NEVER pass full document text to LLM** for document summary
- **ALWAYS generate from section summaries** (hierarchical aggregation)
- Exception: Fallback `"(Document summary unavailable)"` if section summaries fail
- Files: `src/docling_extractor_v2.py`, `src/summary_generator.py`

### 2. Token-Aware Chunking (HybridChunker)
- **Max tokens:** 512 (optimal for legal docs)
- **Tokenizer:** tiktoken (OpenAI text-embedding-3-large)
- **Overlap:** 0 (hierarchical chunking handles naturally)
- **Research basis:** 512 tokens ≈ 500 chars (LegalBench-RAG constraint preserved)
- **Why tokens not chars:** Guarantees embedding model compatibility, handles Czech diacritics correctly
- Changing this invalidates ALL vector stores

### 3. Generic Summaries (Reuter et al., counterintuitive!)
- Length: **150 chars**
- Style: **GENERIC** (NOT expert terminology)

### 4. Summary-Augmented Chunking (SAC)
- Prepend document summary during embedding
- Strip summaries during retrieval
- **-58% context drift** (proven by research)

### 5. Multi-Layer Embeddings (Lima 2024)
- **3 separate FAISS indexes** (NOT merged)
- 2.3x essential chunks vs single-layer

### 6. No Cohere Reranking
- Cohere performs WORSE on legal docs
- Use: `ms-marco`, `bge-reranker` instead

### 7. Hybrid Search (Industry 2025)
- BM25 + Dense + RRF fusion
- **+23% precision** vs dense-only
- RRF k=60 (optimal)

### 8. **AUTONOMOUS AGENT RESPONSES (CRITICAL - NO HARDCODED TEMPLATES)**
- **NEVER use rule-based conditional responses** (if greeting → template)
- **ALL communication generated by LLM agents** - no hardcoded strings
- **Orchestrator returns `final_answer` directly** when no specialized agents needed (greetings, chitchat, meta queries)
- **Why this matters:**
  - Enables contextual awareness and natural conversation flow
  - Eliminates brittle template logic that fails on edge cases
  - Allows agent to adapt responses based on conversation history
  - Modern LLM-based architecture principle
- **Implementation:**
  - Orchestrator: Returns `{"agent_sequence": [], "final_answer": "<LLM-generated response>"}` for greetings
  - Runner: Checks for `final_answer` in orchestrator output, returns directly without building workflow
  - NO if/else conditional logic for response generation anywhere in codebase
- **Files:** `src/multi_agent/runner.py`, `src/multi_agent/agents/orchestrator.py`, `prompts/agents/orchestrator.txt`

---

## 🐳 Docker Architecture

**Current Setup:** 3-tier containerized application with persistent storage

### Services

**1. PostgreSQL (`sujbot_postgres`)**
- **Purpose:** Vector storage (pgvector), knowledge graph (Apache AGE), checkpointing
- **Extensions:** pgvector (vector similarity search), Apache AGE (graph queries)
- **Volumes:**
  - `postgres_data:/var/lib/postgresql/data` (persistent, CRITICAL)
  - `./docker/postgres/init:/docker-entrypoint-initdb.d` (initialization scripts)
- **Ports:** 5432 (⚠️ exposed for development, remove in production)
- **Health Check:** `pg_isready` every 5s
- **Resources:** 2-4 CPU cores, 4-8GB RAM

**2. Backend (`sujbot_backend`)**
- **Purpose:** FastAPI server + multi-agent system + RAG pipeline
- **Base:** Python 3.10 + uv package manager
- **Volumes:**
  - `model_cache:/root/.cache` (sentence-transformers, ~2-5GB, persistent)
  - `./data:/app/data:ro` (documents, read-only)
  - `./vector_db:/app/vector_db:ro` (FAISS indexes + BM25, read-only, REQUIRED)
  - `./logs:/app/logs` (optional debugging)
- **Environment:**
  - `DATABASE_URL` (PostgreSQL connection)
  - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
  - `STORAGE_BACKEND=postgresql`
- **Ports:** 8000 (FastAPI)
- **Health Check:** `curl http://localhost:8000/health` every 30s
- **Resources:** 1-2 CPU cores, 2-4GB RAM

**3. Frontend (`sujbot_frontend`)**
- **Purpose:** React SPA with real-time agent progress visualization
- **Base:** Node 22 + Vite (dev) or Nginx (production)
- **Build Targets:**
  - `development`: Hot reload with Vite dev server (port 5173)
  - `production`: Optimized build served by Nginx (port 80)
- **Environment:**
  - `VITE_API_BASE_URL=http://localhost:8000` (backend connection)
- **Ports:** 5173 (dev), 80 (prod)
- **Resources:** 0.5-1 CPU core, 512MB-1GB RAM

### Volumes (Persistent Data)

**CRITICAL: Do NOT delete these without backup!**
- `postgres_data`: All vectors, graphs, checkpoints (~5-10GB)
- `model_cache`: Downloaded models (~2-5GB, speeds up startup)

### Network

- Bridge network `sujbot_net` (172.20.0.0/16)
- All services communicate via service names (e.g., `backend:8000`)

### Build Targets

```bash
# Development (hot reload, verbose logging)
BUILD_TARGET=development docker-compose up

# Production (optimized, nginx serving frontend)
BUILD_TARGET=production docker-compose up
```

### Common Operations

```bash
# Rebuild after code changes
docker-compose build backend
docker-compose up -d backend

# Rebuild everything
docker-compose build --no-cache
docker-compose up -d

# View logs (real-time)
docker-compose logs -f backend frontend

# Shell access
docker exec -it sujbot_backend bash
docker exec -it sujbot_postgres psql -U postgres -d sujbot

# Backup PostgreSQL
docker exec sujbot_postgres pg_dump -U postgres sujbot > backup.sql

# Restore PostgreSQL
cat backup.sql | docker exec -i sujbot_postgres psql -U postgres sujbot
```

---

## 🔢 Token vs Character Equivalence

**Research Constraint (LegalBench-RAG):**
- Optimal chunk size: **500 characters** for legal/technical documents

**Token-Aware Implementation:**
- Uses **max_tokens=512** (token-based limit)
- **Equivalence:** 512 tokens ≈ 500-640 characters for Czech/English mixed text
- **Calculation:**
  - English: ~4 chars/token → 500 chars ≈ 125 tokens
  - Czech (diacritics): ~5 chars/token → 500 chars ≈ 100 tokens
  - Safety margin: 512 tokens accommodates worst-case Czech text

**Why This Preserves Research Intent:**
1. Same semantic granularity (small chunks for legal precision)
2. Better reliability (guarantees embedding model compatibility)
3. Czech optimization (accounts for UTF-8 multi-byte encoding)
4. Hierarchical structure preserved (HybridChunker respects DoclingDocument hierarchy)

---

## 📂 Key File Locations

### Docker & Infrastructure

- `docker-compose.yml` - Services orchestration (PostgreSQL + Backend + Frontend)
- `docker/backend/Dockerfile` - Backend container (Python 3.10 + uv)
- `docker/frontend/Dockerfile` - Frontend container (Node 22 + Vite/Nginx)
- `docker/postgres/Dockerfile` - PostgreSQL with pgvector + Apache AGE
- `.env` - Environment variables (API keys, ports)
- `start_web.sh` - Convenience script (builds + starts all services)

### Backend (FastAPI + Multi-Agent System)

**Entry Points:**
- `backend/main.py` - FastAPI server, SSE streaming endpoints
- `backend/agent_adapter.py` - Adapter between FastAPI and multi-agent system
- `backend/models.py` - Pydantic request/response models

**Multi-Agent System (Orchestrator-Centric):**
- `src/multi_agent/runner.py` - Main orchestrator (query routing, workflow execution)
- `src/multi_agent/agents/*.py` - 8 agents:
  - **orchestrator** - Dual-phase: routing (PHASE 1) + synthesis/report generation (PHASE 2)
  - **extractor** - Document retrieval
  - **classifier** - Content categorization
  - **requirement_extractor** - ✨ **NEW (SOTA 2024):** Atomic legal requirement extraction from laws/regulations
  - **compliance** - Checklist-based compliance verification (SOTA requirement-first approach)
  - **risk_verifier** - Risk assessment
  - **citation_auditor** - Citation verification
  - **gap_synthesizer** - Knowledge gap analysis with REGULATORY_GAP vs SCOPE_GAP classification
- `src/multi_agent/core/agent_base.py` - Base class with autonomous tool calling loop
- `src/multi_agent/core/event_bus.py` - Real-time event system (agent progress, tool calls)
- `src/multi_agent/tools/adapter.py` - Tool schema conversion (Pydantic → LLM format)
- `src/multi_agent/routing/workflow_builder.py` - Builds workflow with orchestrator synthesis node
- `prompts/agents/orchestrator.txt` - Dual-phase prompt (routing + synthesis instructions)
- `prompts/agents/*.txt` - Other agent system prompts

**Human-in-the-Loop (HITL):**
- `src/multi_agent/hitl/` - HITL components (4 files)
  - `config.py` - Quality thresholds
  - `quality_detector.py` - Multi-metric quality detection
  - `clarification_generator.py` - LLM-based question generation
  - `context_enricher.py` - Query enrichment
- `backend/main.py:268` - `/chat/clarify` endpoint
- **Docs:** [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md)

**RAG Pipeline (PHASE 1-7):**
- `run_pipeline.py` - Document indexing CLI
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/config.py` - Shared configs
- **PHASE 1:** `src/unstructured_extractor.py` (layout model: **yolox**)
- **PHASE 2:** `src/summary_generator.py` (hierarchical summaries)
- **PHASE 3:** `src/multi_layer_chunker.py` (token-aware + SAC)
- **PHASE 4:** `src/embedding_generator.py`, `src/faiss_vector_store.py` (embeddings + FAISS)
- **PHASE 5:** `src/hybrid_search.py`, `src/graph/`, `src/reranker.py` (hybrid search + graph + reranking)
- **PHASE 6:** `src/context_assembly.py` (context prep)
- **PHASE 7:** Multi-agent execution (see Multi-Agent System section above)

**Agent Infrastructure (`src/agent/`):**

Architecture: INFRASTRUCTURE LAYER (tools, providers, config) used by multi-agent orchestration layer

- `src/agent/config.py` - AgentConfig (API keys, paths, settings)
- `src/agent/prompt_loader.py` - Loads prompts from `prompts/` directory
- `src/agent/query_expander.py` - Query expansion for retrieval (used by unified_search tool)
- `src/agent/rag_confidence.py` - Confidence scoring (used by tier1 and tier2 tools)

**LLM Providers:**
- `src/agent/providers/factory.py` - Creates LLM instances (used by ALL 8 agents)
- `src/agent/providers/anthropic_provider.py` - Claude API client
- `src/agent/providers/openai_provider.py` - GPT API client
- `src/agent/providers/gemini_provider.py` - Gemini API client
- `src/agent/providers/base.py` - Base provider interface
- `src/agent/providers/tool_translator.py` - Tool schema translation

**RAG Tools (16 tools in 3 tiers):**
- `src/agent/tools/registry.py` - Tool registry (`get_registry()`)
- `src/agent/tools/base.py` - `BaseTool`, `ToolResult` classes
- `src/agent/tools/tier1_basic.py` - 5 fast tools (100-300ms)
- `src/agent/tools/tier2_advanced.py` - 10 quality tools (500-1000ms)
  - `multi_doc_synthesizer` - Multi-document synthesis
  - `contextual_chunk_enricher` - Contextual Retrieval (-58% context drift)
- `src/agent/tools/tier3_analysis.py` - 2 analysis tools (500ms-3s)
  - `get_stats` - Corpus/index statistics
  - `definition_aligner` - ✨ **NEW:** Legal terminology mapping (Apache AGE graph + pgvector semantic)
- `src/agent/tools/token_manager.py` - Token tracking
- `src/agent/tools/utils.py` - Tool utilities

### Frontend (React + TypeScript + Vite)

**Entry Points:**
- `frontend/src/main.tsx` - React app entry point
- `frontend/src/App.tsx` - Main application component
- `frontend/src/pages/ChatPage.tsx` - Chat interface page

**Core Components:**
- `frontend/src/hooks/useChat.ts` - Chat state management + SSE streaming
- `frontend/src/services/api.ts` - Backend API client
- `frontend/src/components/chat/` - Chat UI components
  - `ChatContainer.tsx` - Main chat container
  - `ChatMessage.tsx` - Message display with inline tool calls
  - `ChatInput.tsx` - Message input with attachments
  - `AgentProgress.tsx` - Real-time agent activity visualization (FIXED 2025-11-12)
  - `ClarificationModal.tsx` - HITL clarification dialog
  - `ToolCallDisplay.tsx` - Tool execution results

**Design System:**
- `frontend/src/design-system/` - Reusable UI components
- `frontend/src/types/index.ts` - TypeScript type definitions

### Tests

- `tests/test_phase*.py` - Pipeline tests
- `tests/agent/` - Agent tests (49 tests)
- `tests/graph/` - Knowledge graph tests
- `tests/multi_agent/integration/` - HITL integration tests
- `test_agent_progress_fix.py` - Real-time progress event verification

---

## 🛠️ Common Development Tasks

### Adding New Tool
1. Create class in `src/agent/tools/tier{1,2,3}_{basic,advanced,analysis}.py`
2. Define `ToolInput` schema (Pydantic)
3. Implement `execute_impl()` method
4. Register with `@register_tool` decorator
5. Add tests in `tests/agent/tools/`

Example skeleton:
```python
class MyToolInput(ToolInput):
    query: str = Field(..., description="User query")

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    tier = 1
    input_schema = MyToolInput

    def execute_impl(self, query: str) -> ToolResult:
        results = self.vector_store.search(query, k=10)
        return ToolResult(success=True, data=results)
```

### New Tools (2025-01)

#### **multi_doc_synthesizer** (Tier 2)
Synthesizes information from multiple documents using LLM. Replaces broken `compare_documents` tool.

**Use cases:**
```python
# Compare documents
tool.execute(
    document_ids=["doc1", "doc2", "doc3"],
    synthesis_query="Compare privacy policies",
    synthesis_mode="compare"
)

# Unified summary
tool.execute(
    document_ids=["standard1", "standard2"],
    synthesis_query="Data retention requirements",
    synthesis_mode="summarize"
)
```

**Key features:**
- Synthesis modes: `compare`, `summarize`, `analyze`
- Uses public API only (hierarchical_search with document filters)
- 2-10 documents supported
- Cites all source documents

#### **contextual_chunk_enricher** (Tier 2)
Implements Anthropic Contextual Retrieval technique (-58% context drift).

**Use cases:**
```python
# Auto mode (intelligent selection)
tool.execute(
    chunk_ids=["doc1:sec1:0", "doc1:sec2:1"],
    enrichment_mode="auto"
)

# Maximum context
tool.execute(
    chunk_ids=["doc1:sec1:0"],
    enrichment_mode="both",  # document + section summaries
    include_metadata=True
)
```

**Enrichment modes:**
- `auto`: Selects best mode (section > document)
- `document_summary`: Prepend document context
- `section_summary`: Prepend section context
- `both`: Maximum context (document + section)

**Research basis:** Anthropic (2024) - Contextual Retrieval

### Adding New Frontend Component

1. Create component in `frontend/src/components/`
2. Follow existing patterns (TypeScript + Tailwind CSS)
3. Use design system components from `frontend/src/design-system/`
4. Add types to `frontend/src/types/index.ts`
5. Update parent component to import

Example skeleton:
```typescript
import React from 'react';
import type { MyComponentProps } from '../../types';

export const MyComponent: React.FC<MyComponentProps> = ({ data }) => {
  return (
    <div className="p-4 bg-white dark:bg-gray-800">
      {/* Component content */}
    </div>
  );
};
```

### GPT-5 and O-Series Model Compatibility

**RECOMMENDED MODELS (production-ready):**
- **Production:** `claude-sonnet-4-5` (highest quality, best for complex queries)
- **Development:** `gpt-4o-mini` (best cost/performance, $0.15/$0.60 per 1M tokens)
- **Budget:** `claude-haiku-4-5` (fastest, cheapest Claude model)

**GPT-5 Support (✅ IMPLEMENTED, EXPERIMENTAL):**

GPT-5 and O-series models (`gpt-5`, `gpt-5-mini`, `o1`, `o3`, `o4-mini`) are **now supported** but require special parameter handling:

```python
# GPT-5/o-series require different parameters
if model.startswith(("gpt-5", "o1", "o3", "o4")):
    params = {
        "model": model,
        "max_completion_tokens": 300,  # NOT max_tokens
        "temperature": 1.0,  # ONLY 1.0 supported (default)
        "reasoning_effort": "minimal"  # Controls reasoning depth
    }
else:
    # GPT-4 and earlier
    params = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0.7
    }
```

**`reasoning_effort` parameter:**
- `"minimal"` - Fastest, used for simple tasks (summarization, context generation)
- `"low"` - Light reasoning
- `"medium"` - Default reasoning (if not specified)
- `"high"` - Deep reasoning for complex tasks

**Why GPT-5/o-series may not be recommended:**
- ⚠️ API parameter differences can cause confusion (`max_completion_tokens` vs `max_tokens`)
- ⚠️ Temperature is fixed at 1.0 (no customization)
- ⚠️ May be more expensive than GPT-4o-mini for simple tasks
- ⚠️ `reasoning_effort` behavior can be unpredictable for certain prompts

**Files with GPT-5 support:**
- ✅ `src/summary_generator.py` - Document/section summaries
- ✅ `src/contextual_retrieval.py` - Context generation
- ⚠️ `src/agent/query_expander.py` - Not yet updated (uses gpt-4o-mini)

**Testing recommendation:** If you want to use GPT-5 models, test thoroughly with your specific use case before deploying to production. Fall back to `gpt-4o-mini` if you encounter issues.

---

## ⚖️ SOTA Compliance Workflow

**Status:** ✅ IMPLEMENTED (2025-01-18)

**Detailed Documentation:** See [`docs/SOTA_COMPLIANCE_IMPLEMENTATION.md`](docs/SOTA_COMPLIANCE_IMPLEMENTATION.md) for complete implementation guide, research foundations, and deployment checklist.

### Quick Overview

**Problem:** Traditional RAG compliance systems have ~40-60% false positives due to "evidence-first" bias (cherry-picking evidence to confirm desired outcome).

**Solution:** Requirement-first approach using Plan-and-Solve pattern:
1. **PHASE 1 (Plan):** Extract atomic requirements FROM law independently → Generate checklist
2. **PHASE 2 (Solve):** Verify EACH requirement sequentially → Classify gaps (REGULATORY_GAP vs SCOPE_GAP)

**Key Agents:**
- `requirement_extractor` - Extracts atomic legal obligations from laws/regulations
- `compliance` - Checklist-based verification (requirement-first, no discovery mode)
- `gap_synthesizer` - Prioritizes gaps (REGULATORY_GAP = must fix, SCOPE_GAP = not applicable)

**Workflow:**
```
Query: "Je dokumentace v souladu s Vyhláškou 157/2025?"
→ Orchestrator routes to: [extractor → requirement_extractor → compliance → gap_synthesizer]
```

**Files:**
- Implementation: `src/multi_agent/agents/{requirement_extractor,compliance}.py`
- Tools: `src/agent/tools/tier3_analysis.py` (DefinitionAlignerTool)
- Prompts: `prompts/agents/{requirement_extractor,compliance,gap_synthesizer}.txt`
- Tests: `tests/multi_agent/integration/test_compliance_workflow.py`

**Performance:**
- Latency: 30-60s for complex queries
- Cost: ~$0.05-0.15 per query (claude-haiku-4-5)
- Accuracy: ~90-95% requirement extraction recall, ~5-10% false positive rate (vs ~40-60% in legacy systems)

---

## 🎯 Best Practices

### Agent Development
- **Tool tier selection:** Always start with TIER 1 tools (fast), escalate to TIER 2/3 only when needed
- **Query expansion:** Use `num_expands=0` (default) for speed, `num_expands=1-2` for recall-critical queries
- **Graph boost:** Enable only for entity-focused queries (organizations, standards, regulations)
- **Prompt caching:** Enable via `ENABLE_PROMPT_CACHING=true` for 90% cost savings on repeated queries
- **Context pruning:** Keep conversation history under 50K tokens to prevent quadratic growth

### Pipeline Indexing
- **Speed modes:** Use `SPEED_MODE=fast` for development, `SPEED_MODE=eco` for overnight bulk processing
- **Batch processing:** Index directories instead of individual files for better throughput
- **Knowledge graph:** Set `KG_BACKEND=neo4j` for production, `simple` for testing
- **Entity deduplication:** Use Layer 1 + Layer 3 (production balanced mode) for legal docs
- **Validation:** Always run `pytest tests/` before committing pipeline changes

### Code Quality
- **Type hints:** Required for all public APIs (use `mypy src/` to verify)
- **Error handling:** Use graceful degradation (e.g., reranker unavailable → fall back to RRF)
- **Logging:** Use appropriate levels (debug/info/warning/error) - avoid print statements
- **Testing:** Write tests BEFORE implementing new features (TDD approach)
- **Documentation:** Update PIPELINE.md if research constraints change
- **Model selection:** ALWAYS use `gpt-4o-mini` (NOT gpt-5-nano) for stability and cost savings

### Performance
- **Embedding cache:** Monitor hit rate with `embedder.get_cache_stats()` (target >80%)
- **FAISS indexes:** Keep layer separation (DO NOT merge L1/L2/L3)
- **Reranker loading:** Lazy load to reduce startup time (~2s savings)
- **Token limits:** Use `max_total_tokens` parameter to prevent context overflow

### Debugging

**Docker Logs:**
```bash
# View real-time logs
docker-compose logs -f backend  # Backend + agent execution
docker-compose logs -f frontend # Frontend build/serve
docker-compose logs -f postgres # Database queries

# Filter logs by error level
docker-compose logs backend | grep ERROR
docker-compose logs backend | grep WARNING
```

**Backend Debugging:**
```bash
# Shell access
docker exec -it sujbot_backend bash

# Python REPL with full context
docker exec -it sujbot_backend python
>>> from src.multi_agent.runner import MultiAgentRunner
>>> runner = MultiAgentRunner()

# Check vector store stats
docker exec -it sujbot_backend python -c "from src.faiss_vector_store import FAISSVectorStore; store = FAISSVectorStore('vector_db'); print(store.get_stats())"

# View environment
docker exec -it sujbot_backend env | grep API_KEY
```

**Frontend Debugging:**
```bash
# Hot reload (development mode)
BUILD_TARGET=development docker-compose up frontend

# View browser console for SSE events
# Open DevTools → Console, filter by "FRONTEND: Received event"

# Check API connection
curl http://localhost:8000/health
curl http://localhost:8000/models
```

**Multi-Agent Debugging:**
- **EventBus:** Check `src/multi_agent/core/event_bus.py` for event emission
- **Agent Progress:** Verify events in `backend/agent_adapter.py:217-254`
- **Frontend SSE:** Trace events in `frontend/src/hooks/useChat.ts:210`
- **Test script:** Run `uv run python test_agent_progress_fix.py` to verify event flow

**Database Debugging:**
```bash
# PostgreSQL shell
docker exec -it sujbot_postgres psql -U postgres -d sujbot

# Check tables
\dt

# Query vectors
SELECT id, document_id FROM vector_embeddings LIMIT 10;

# Check graph (Apache AGE)
SELECT * FROM ag_graph.ag_label;
```

---

## 🔧 Configuration

**SSOT (Single Source of Truth): `config.json.example`** (UPDATED 2025-11-10)

All configuration lives in [`config.json.example`](config.json.example) - DO NOT duplicate config in CLAUDE.md!

**Migration from .env to config.json:**
- Strict validation - NO fallbacks, NO defaults
- If any required parameter is missing, application exits with error
- Hierarchical JSON structure for better organization

**Setup:**
```bash
cp config.json.example config.json
# Edit config.json with your API keys and settings
# ALL required fields must be filled in
```

**Key decisions** (see `config.json.example` for all options):
- **Required:** `api_keys.anthropic_api_key` or `api_keys.openai_api_key`
- **Embedding:** `models.embedding_model="bge-m3"` (macOS M1/M2/M3, free) or `"text-embedding-3-large"` (Windows, cloud)
- **Knowledge Graph:** `knowledge_graph.backend="neo4j"` (production) or `"simple"` (dev/testing)
- **Speed:** `summarization.speed_mode="fast"` (default) or `"eco"` (50% cheaper, overnight jobs)

**For detailed config docs, read `config.json.example` inline comments.**

---

## 📚 Code Style

**Formatting:**
```bash
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

## 📖 Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS, reranking
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending
5. **HybridRAG** (2024) - Graph boosting (+8% factual correctness)

---

---

**Last Updated:** 2025-11-19
**Version:** PHASE 1-7 COMPLETE + Orchestrator-Centric Multi-Agent + HITL + Docker Web UI (16 tools, 7 agents, real-time progress visualization)

**Recent Changes (2025-11-19):**
- ✅ Cleaned up `src/agent/` folder - removed unused modules (SSOT compliance)
- ✅ Deleted: `graph_adapter.py`, `graph_loader.py`, `validation.py` (used only in tests)
- ✅ Deleted: 8 obsolete test files that imported removed modules
- ✅ Architecture clarified: `src/agent/` = INFRASTRUCTURE (tools, providers, config), `src/multi_agent/` = ORCHESTRATION (agents, workflows)
- ✅ All production code verified working after cleanup

**Recent Changes (2025-11-13):**
- ✅ Removed `report_generator` agent (redundant with orchestrator)
- ✅ Orchestrator now handles BOTH routing (PHASE 1) and synthesis/report generation (PHASE 2)
- ✅ Single point of user communication - orchestrator is the only agent that talks to users
- ✅ Cleaner architecture: 7 agents instead of 8, orchestrator called twice per query
- ✅ Workflow pattern: agents → orchestrator_synthesis → END

**Notes:**
- `vector_db/` is tracked in git (contains FAISS + BM25 indexes) - DO NOT add to `.gitignore`
- PostgreSQL volume `postgres_data` contains all vectors/graphs - backup before deletion!
- Frontend real-time progress fixed 2025-11-12 (LangGraph state extraction bug)
