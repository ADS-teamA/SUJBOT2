# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**SUJBOT2** is a production-ready RAG (Retrieval-Augmented Generation) system optimized for legal and technical documents. It implements state-of-the-art techniques from 4 research papers (2024-2025) and features a 7-phase pipeline with an interactive AI agent.

**Status:** PHASE 1-7 COMPLETE ✅ (Full SOTA 2025 RAG System + 17-Tool Agent + RAG Confidence Scoring)

**Visual Documentation:**
- 📥 **Indexing Pipeline (Phase 1-5):** [`indexing_pipeline.html`](indexing_pipeline.html) - Complete indexing process from PDF to searchable vector store
- 💬 **User Search Pipeline (Phase 7):** [`user_search_pipeline.html`](user_search_pipeline.html) - User query flow with 17 agent tools breakdown

**Core Technologies:**
- Document processing: IBM Docling (hierarchical structure extraction)
- Embeddings: OpenAI text-embedding-3-large or local BGE-M3
- Vector store: FAISS (3-layer indexing)
- Retrieval: Hybrid (BM25 + Dense + RRF fusion) with cross-encoder reranking
- Knowledge Graph: Entity/relationship extraction with NetworkX/Neo4j
- Agent: Claude SDK with 17 specialized tools (Anthropic Sonnet/Haiku)
- **RAG Confidence Scoring:** Real-time retrieval quality assessment with 7 metrics

---

## Development Setup

### Installation & Environment

**Package Manager:** Uses `uv` (required)
```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY

# Platform-specific notes:
# Windows: Install PyTorch FIRST before uv sync
# macOS M1/M2/M3: Use EMBEDDING_MODEL=bge-m3 (local, free, GPU-accelerated)
# Windows: Use EMBEDDING_MODEL=text-embedding-3-large (cloud)
```

**Required API Keys:**
- `ANTHROPIC_API_KEY` - Required (for agent and optional summaries)
- `OPENAI_API_KEY` - Optional (for embeddings, summaries, knowledge graph)

### Common Commands

**Run Pipeline (Index Documents):**
```bash
# Single document
uv run python run_pipeline.py data/document.pdf

# Batch processing
uv run python run_pipeline.py data/regulace/

# Fast mode (default): 2-3 min, full price
uv run python run_pipeline.py data/document.pdf

# Eco mode: 15-30 min, 50% cheaper (set SPEED_MODE=eco in .env)
```

**Run Agent (Interactive CLI):**
```bash
# Launch agent
uv run python -m src.agent.cli

# With specific vector store
uv run python -m src.agent.cli --vector-store output/my_doc/phase4_vector_store

# Debug mode
uv run python -m src.agent.cli --debug
```

**Run Tests:**
```bash
# All tests
uv run pytest tests/ -v

# Specific phase/component
uv run pytest tests/test_phase4_indexing.py -v
uv run pytest tests/agent/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Single test
uv run pytest tests/agent/test_validation.py::test_api_key_validation -v
```

**Code Quality:**
```bash
# Format code
uv run black src/ tests/ --line-length 100
uv run isort src/ tests/ --profile black

# Type checking
uv run mypy src/
```

**Benchmark Evaluation:**
```bash
# Index benchmark documents (PrivacyQA dataset)
uv run python scripts/index_benchmark_docs.py

# Run full benchmark evaluation (194 QA pairs)
uv run python scripts/run_benchmark.py

# Quick test with limited queries
uv run python scripts/run_benchmark.py --max-queries 5

# Disable reranking for faster evaluation
uv run python scripts/run_benchmark.py --disable-reranking

# Custom output directory
uv run python scripts/run_benchmark.py --output-dir benchmark_results/custom
```

### Neo4j Knowledge Graph (Optional Production Setup)

**When to Use:**
- Production deployments (recommended)
- Multi-user environments (WebApp)
- Requires `browse_entities` tool (entity discovery by type/confidence/search)
- Requires efficient multi-hop graph traversal
- Enables indexed queries (faster than JSON for large graphs)

**Setup Steps:**
1. **Create Neo4j Aura instance** (free tier available):
   - Visit: https://console.neo4j.io/
   - Click "New Instance" → Select "Free" tier
   - **Save your password** (cannot be recovered!)
   - Copy connection URI (format: `neo4j+s://xxxxx.databases.neo4j.io`)

2. **Configure `.env`:**
   ```bash
   KG_BACKEND=neo4j
   NEO4J_URI=neo4j+s://YOUR_INSTANCE_ID.databases.neo4j.io
   NEO4J_USERNAME=neo4j  # Default username
   NEO4J_PASSWORD=YOUR_PASSWORD  # From step 1
   NEO4J_DATABASE=neo4j  # Default database name
   ```

3. **Migrate existing data:**
   ```bash
   # After running indexing pipeline
   uv run python scripts/migrate_kg_to_neo4j.py --kg-file vector_db/unified_kg.json
   ```

4. **Verify connection:**
   ```bash
   uv run python -m src.agent.cli
   # Should show: "✓ Connected to Neo4j: X entities, Y relationships"
   ```

**Fallback Behavior:**
- **Automatic fallback** (KG_BACKEND not set or =simple): Falls back to JSON with warning if Neo4j fails
- **Fail-fast** (KG_BACKEND=neo4j explicit): No fallback - fails with actionable error messages
  - Auth errors: Check NEO4J_USERNAME, NEO4J_PASSWORD
  - Connection errors: Check NEO4J_URI, server status
  - Timeout errors: Check server load or query complexity

**Degraded Mode Warning:**
When fallback occurs, you'll see:
```
⚠️  WARNING: Running in degraded mode with JSON backend
Some tools will not work optimally:
- browse_entities: Unavailable (requires Neo4j indexed queries)
- multi_hop_search: Slower (no graph database optimization)
- graph_search: Limited to JSON export data
```

**JSON Backend (Development/Testing):**
- Set `KG_BACKEND=simple` in `.env` (or leave unset)
- No external dependencies (uses local JSON files)
- Limited tool functionality:
  - `browse_entities` unavailable
  - `multi_hop_search` slower (linear search)
  - `graph_search` works with reduced performance
- Best for: Development, testing, single-user scenarios

### Entity Deduplication (Incremental Neo4j)

**Purpose:**
Prevent duplicate entities when indexing multiple documents into Neo4j. Uses a sophisticated 3-layer detection strategy to merge entities with different representations (exact matches, semantic variants, acronyms).

**Architecture:**
- **Layer 1 (Exact Match):** Fast O(1) hash lookup on `(type, normalized_value)` - <1ms latency
- **Layer 2 (Semantic Similarity):** Embedding-based cosine similarity - 50-200ms latency
- **Layer 3 (Acronym Expansion):** Domain-specific acronym dictionary + fuzzy matching - 100-500ms latency

Each layer is optional and independently configurable. Layers are applied in order; first match wins.

**Configuration:**

All deduplication settings are controlled via `.env` or `EntityDeduplicationConfig`:

```bash
# Master switch
KG_DEDUPLICATE_ENTITIES=true  # Enable/disable entire system (default: true)

# Layer 2: Semantic similarity (requires embeddings)
KG_DEDUP_USE_EMBEDDINGS=false  # Enable embedding-based similarity (default: false)
KG_DEDUP_SIMILARITY_THRESHOLD=0.90  # Cosine similarity threshold (default: 0.90)

# Layer 3: Acronym expansion
KG_DEDUP_USE_ACRONYM_EXPANSION=false  # Enable acronym matching (default: false)
KG_DEDUP_ACRONYM_FUZZY_THRESHOLD=0.85  # Fuzzy match threshold (default: 0.85)
KG_DEDUP_CUSTOM_ACRONYMS="ACRO1:expansion1,ACRO2:expansion2"  # Custom acronyms

# Neo4j optimizations
KG_DEDUP_APOC_ENABLED=true  # Try APOC, fallback to Cypher (default: true)
```

**Usage Recommendations:**

1. **Development/Testing (Fast mode):**
   ```bash
   KG_DEDUPLICATE_ENTITIES=true  # Layer 1 only
   KG_DEDUP_USE_EMBEDDINGS=false
   KG_DEDUP_USE_ACRONYM_EXPANSION=false
   ```
   - Latency: <1ms per entity
   - Precision: 100% (exact match only)
   - Recall: ~60% (misses variants like "ISO 14001" vs "ISO 14001:2015")

2. **Production (Balanced mode - RECOMMENDED):**
   ```bash
   KG_DEDUPLICATE_ENTITIES=true  # Layer 1 + Layer 3
   KG_DEDUP_USE_EMBEDDINGS=false
   KG_DEDUP_USE_ACRONYM_EXPANSION=true
   ```
   - Latency: ~100-500ms per unique entity
   - Precision: ~98% (high confidence acronym matches)
   - Recall: ~85% (catches acronyms + exact matches)
   - Best for: Legal/regulatory documents with many acronyms

3. **Maximum Quality (Slow mode):**
   ```bash
   KG_DEDUPLICATE_ENTITIES=true  # All 3 layers
   KG_DEDUP_USE_EMBEDDINGS=true
   KG_DEDUP_SIMILARITY_THRESHOLD=0.90
   KG_DEDUP_USE_ACRONYM_EXPANSION=true
   ```
   - Latency: ~200-700ms per unique entity
   - Precision: ~95% (embedding similarity has false positives)
   - Recall: ~95% (catches semantic variants)
   - Best for: High-quality knowledge graphs where recall is critical

**Property Merging Strategy:**

When a duplicate is detected, properties are merged as follows:

- **`confidence`**: `MAX(primary, duplicate)` - keep highest confidence score
- **`source_chunk_ids`**: `UNION(primary, duplicate)` - combine all chunk references
- **`document_id`**: Track which documents mention each entity
- **`first_mention_chunk_id`**: Preserve original first occurrence
- **`metadata`**: Deep merge with `merged_from` tracking

**Example:**

```python
# Before deduplication (2 entities):
Entity 1: {id: "e1", value: "GRI 306", confidence: 0.92, chunks: ["chunk1", "chunk2"]}
Entity 2: {id: "e2", value: "Global Reporting Initiative 306", confidence: 0.95, chunks: ["chunk3"]}

# After Layer 3 acronym expansion (1 merged entity):
Entity 1: {
    id: "e1",
    value: "GRI 306",
    confidence: 0.95,  # MAX(0.92, 0.95)
    chunks: ["chunk1", "chunk2", "chunk3"],  # UNION
    metadata: {merged_from: ["e2"]}
}
```

**Built-in Acronym Dictionary:**

The system includes 29 common acronyms for legal/sustainability domains:

- **Standards:** GRI, GSSB, SASB, TCFD, CDP, SBTi, ISO, IEC
- **Regulations:** GDPR, CCPA, HIPAA, SOX, FCPA
- **Environmental:** EPA, EIA, LCA
- **Organizations:** WHO, ILO, OECD, UN, IFC, EU

Custom acronyms can be added via `KG_DEDUP_CUSTOM_ACRONYMS` environment variable.

**Files:**
- `src/graph/config.py` - `EntityDeduplicationConfig` with all settings
- `src/graph/neo4j_deduplicator.py` - Neo4j incremental deduplication (APOC/Cypher)
- `src/graph/deduplicator.py` - 3-layer deduplicator for SimpleGraphBuilder
- `src/graph/similarity_detector.py` - Layer 2 embedding similarity
- `src/graph/acronym_expander.py` - Layer 3 acronym expansion
- `tests/graph/test_entity_deduplication.py` - Comprehensive test suite (28 tests)

**Performance Characteristics:**

- **APOC mode:** ~10-20ms per 1000 entities (uses `apoc.coll.union` for array merging)
- **Pure Cypher mode:** ~20-50ms per 1000 entities (manual array manipulation)
- **Batch size:** 1000 entities per Neo4j transaction (optimal for throughput)
- **Uniqueness constraint overhead:** ~5ms (enforced at database level)

**Troubleshooting:**

1. **High false positive rate with embeddings:**
   - Increase `KG_DEDUP_SIMILARITY_THRESHOLD` (e.g., 0.95)
   - Or disable Layer 2: `KG_DEDUP_USE_EMBEDDINGS=false`

2. **APOC not available error:**
   - System automatically falls back to pure Cypher
   - No action needed (check logs for fallback message)

3. **Constraint creation fails:**
   - Neo4j 4.2+ required for composite constraints
   - System falls back to composite index
   - Set `create_uniqueness_constraints=false` in config if needed

---

## Architecture Overview

### 7-Phase Pipeline

The system processes documents through 7 distinct phases:

**PHASE 1: Hierarchy Extraction**
- Tool: IBM Docling
- Purpose: Extract document structure using font-size classification
- Output: Hierarchical sections (depth=4), metadata
- File: `src/docling_extractor_v2.py`

**PHASE 2: Document Summary Generation**
- Model: gpt-4o-mini (or gpt-5-nano)
- Purpose: Generate DOCUMENT summary ONLY (150 chars) - NOT section summaries!
- Section summaries deferred to PHASE 3B (eliminates truncation problem)
- **⚠️ CRITICAL CONSTRAINT:** Document summary MUST be generated hierarchically from section summaries
  - **NEVER pass full document text to LLM** (causes token waste, context overflow, poor quality)
  - **ALWAYS wait for section summaries** (PHASE 3B) → then aggregate them into document summary
  - **Flow:** Sections → Section Summaries (PHASE 3B) → Document Summary (hierarchical)
  - **Exception:** Empty fallback "(Document summary unavailable)" if section summaries fail
- Critical: Use GENERIC summaries (NOT expert) - counterintuitive but proven better
- File: `src/summary_generator.py`, `src/docling_extractor_v2.py`

**PHASE 3: Multi-Layer Chunking + Contextual Retrieval** (3 sub-phases)
- Method: RCTS (500 chars, no overlap)
- Layers: Document (L1), Section (L2), Chunk (L3 - PRIMARY)
- **NEW ARCHITECTURE (2025-11-03):** Hierarchical summary generation from chunk contexts
- **3A: Chunk Context Generation** - Contextual Retrieval (-67% retrieval failures)
  - Generates LLM-based context for each chunk
  - Uses document summary + section title + neighboring chunks
  - Files: `src/contextual_retrieval.py`
- **3B: Section Summary Generation** - FROM chunk contexts (NO TRUNCATION!)
  - OLD: `section_text[:2000]` → summary (40% coverage for 5000 char sections) ❌
  - NEW: ALL chunk contexts → summary (100% coverage) ✅
  - Eliminates truncation problem, better quality
  - File: `src/multi_layer_chunker.py:_generate_section_summaries_from_contexts()`
- **3C: Summary Validation** - Quality assurance
  - Validates all summaries >= 50 chars
  - Reports missing/invalid summaries with warnings
  - File: `src/multi_layer_chunker.py:_validate_summaries()`
- Output: Multi-layer chunks with validated summaries for L1/L2/L3

**PHASE 4: Embedding + FAISS Indexing**
- Embeddings: text-embedding-3-large (3072D) or bge-m3 (1024D)
- Storage: 3 separate FAISS indexes (IndexFlatIP, cosine similarity)
- Context: Embeds `context + raw_content`, stores only `raw_content`
- File: `src/embedding_generator.py`, `src/faiss_vector_store.py`

**PHASE 5: Advanced Retrieval** (3 sub-phases)
- **5A: Knowledge Graph** - Entity/relationship extraction (18 types)
  - Files: `src/graph/`, `src/graph_retrieval.py`
- **5B: Hybrid Search** - BM25 + Dense + RRF fusion (+23% precision)
  - File: `src/hybrid_search.py`
- **5C: Reranking** - Cross-encoder 2-stage retrieval (+25% accuracy)
  - File: `src/reranker.py`

**PHASE 6: Context Assembly**
- Purpose: Prepare chunks for LLM
- Tasks: Strip SAC summaries, add citations, manage token limits
- File: `src/context_assembly.py`

**PHASE 7: RAG Agent**
- Framework: Claude SDK (official Anthropic SDK)
- Tools: 17 specialized tools (6 basic + 8 advanced + 3 analysis) - see [`user_search_pipeline.html`](user_search_pipeline.html) for interactive breakdown
- Features: Streaming, prompt caching (90% savings), cost tracking, **query expansion**, **RAG confidence scoring**
- Query Expansion: Multi-query generation (+15-25% recall) with `num_expands` parameter
- Confidence Scoring: 7-metric system (RAGAS-based) for retrieval quality assessment
- Files: `src/agent/`, `src/agent/query_expander.py`

### Configuration Architecture

**Centralized Config System:**
- Main: `src/config.py` - Shared configs (Extraction, Summarization, Chunking, Embedding)
- Pipeline: `src/indexing_pipeline.py` - `IndexingConfig` (orchestrates all phases)
- Agent: `src/agent/config.py` - `AgentConfig` (agent-specific settings)
- Knowledge Graph: `src/graph/config.py` - `KnowledgeGraphConfig`

**Loading Pattern:**
```python
# Load from environment (.env)
from src.indexing_pipeline import IndexingConfig
config = IndexingConfig.from_env()

# Override specific settings
config = IndexingConfig.from_env(
    enable_knowledge_graph=True,
    speed_mode="eco"
)
```

### Cost Tracking System

**Global Tracker Pattern:**
```python
from src.cost_tracker import get_global_tracker, reset_global_tracker

# Reset at start of operation
reset_global_tracker()

# Track usage (automatic in pipeline components)
tracker = get_global_tracker()
tracker.track_llm("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=500)

# Get summary
print(tracker.get_summary())
```

**Supported Providers:**
- Anthropic: Claude Haiku, Sonnet, Opus
- OpenAI: GPT-4o, GPT-5, o-series, embeddings
- Voyage AI: Embeddings
- Local models: Free (bge-m3, etc.)

---

## Critical Implementation Rules

### Research-Based Constraints (DO NOT CHANGE)

These decisions are backed by research papers and extensive testing:

1. **Hierarchical Document Summary** (MANDATORY - Architecture Constraint)
   - Document summary MUST be generated FROM section summaries (hierarchical aggregation)
   - **NEVER pass full document text to LLM** for document summary generation
   - **Flow:** Extract sections (PHASE 1) → Generate section summaries (PHASE 3B) → Aggregate into document summary
   - **Rationale:**
     - Prevents token overflow (100-page docs exceed context limits)
     - Better quality (LLM focuses on pre-digested summaries, not raw text)
     - Cost efficiency (section summaries reused for multiple purposes)
     - Scalability (works for documents of ANY size)
   - **Exception:** Fallback "(Document summary unavailable)" if section summaries fail
   - **Files:** `src/docling_extractor_v2.py`, `src/summary_generator.py`

2. **RCTS > Fixed-size chunking** (LegalBench-RAG)
   - Chunk size: **500 chars** (optimal for legal/technical docs)
   - Overlap: 0 (RCTS handles naturally via hierarchy)

3. **Generic > Expert summaries** (Reuter et al., counterintuitive!)
   - Summary length: **150 chars**
   - Style: **Generic** (NOT expert terminology)

4. **SAC reduces DRM by 58%** (Reuter et al.)
   - Prepend document summary to each chunk during embedding
   - Strip summaries during retrieval (context assembly)

5. **Multi-layer embeddings** (Lima 2024)
   - 3 separate FAISS indexes (not merged)
   - 2.3x essential chunks compared to single-layer

6. **No Cohere reranking** (LegalBench-RAG)
   - Cohere reranker performs WORSE than baseline on legal documents
   - Use cross-encoder models instead (ms-marco, bge-reranker)

7. **Hybrid > Pure Dense** (Industry best practice 2025)
   - BM25 + Dense + RRF fusion outperforms dense-only by +23% precision
   - RRF k=60 is optimal parameter

### Agent Tool Guidelines

**📊 Visual Tool Reference:** See [`user_search_pipeline.html`](user_search_pipeline.html) for interactive tool documentation with examples and use cases.

**Tool Tiers (Speed/Quality Tradeoff):**
- **TIER 1** (6 tools): Fast (100-300ms baseline), basic retrieval - Use first
  - Key tool: **`search`** (unified hybrid search with optional query expansion & graph boosting) ✅ **Query Expansion** ✅ **Graph Boost**
    - **Query Expansion** (`num_expands`):
      - `num_expands=0`: Fast mode (~200ms) - original query only, 1 query total (default)
      - `num_expands=1`: Balanced mode (~500ms) - original + 1 paraphrase, 2 queries total (+15-25% recall est.)
      - `num_expands=2`: Better recall (~800ms) - original + 2 paraphrases, 3 queries total (+15-25% recall est.)
      - `num_expands=3-5`: Best recall (~1.2-2s) - original + 3-5 paraphrases, 4-6 queries total (max quality)
      - Uses LLM-based paraphrasing (GPT-5 nano or Claude Haiku) to find docs with different terminology
    - **Graph Boost** (`enable_graph_boost`):
      - When enabled: +200-500ms overhead, +8% factual correctness on entity queries (HybridRAG 2024)
      - Boosts chunks mentioning query-relevant entities (+30% weight) and central entities (+15% weight)
      - Best for: entity-focused queries (organizations, standards, regulations)
      - Graceful degradation: Falls back to hybrid search if graph unavailable
    - Implementation: `src/agent/query_expander.py` + `src/agent/tools/tier1_basic.py` + `src/graph_retrieval.py`
  - Other tools: `get_document_list`, `get_document_info`, `get_tool_help`, `list_available_tools`, `exact_match_search`
- **TIER 2** (8 tools): Quality (500-1000ms), advanced retrieval - Use for complex queries
  - Tools: `graph_search` (4 modes: entity_mentions, entity_details, relationships, multi_hop), `browse_entities` (NEW), `compare_documents`, `explain_search_results`, `assess_retrieval_confidence`, `filtered_search` (3 search methods), `similarity_search`, `expand_context`
  - **Consolidated:** `graph_search` replaces `multi_hop_search` + `entity_tool`; `filtered_search` replaces `exact_match_search`
  - **NEW (2025-10-30):** `browse_entities` - Discover entities by type/confidence/search term without knowing specific names. Complements `graph_search` (browse to discover, then graph_search to explore)
- **TIER 3** (3 tools): Deep (1-3s), analysis and insights - Use sparingly
  - Tools: `timeline_view`, `summarize_section`, `get_stats`

**Tool Design Principles:**
- All tools inherit from `BaseTool` in `src/agent/tools/base.py`
- Register with `@register_tool` decorator
- Validation via Pydantic `ToolInput` schemas
- Return `ToolResult` with success/data/citations/metadata
- Tools must handle graceful degradation (e.g., reranker unavailable)

**Adding New Tools:**
1. Create tool class in appropriate tier file (`tier1_basic.py`, `tier2_advanced.py`, `tier3_analysis.py`)
2. Define input schema with Pydantic
3. Implement `execute_impl()` method
4. Register with `@register_tool`
5. Add tests in `tests/agent/tools/`

### Prompt Caching Strategy (Anthropic Only)

**Automatic Caching Points:**
- System prompt (agent instructions)
- Tool definitions (17 tools)
- Initial messages (document list)
- Long tool results (>1024 tokens)

**Cache Management:**
- Enabled via `ENABLE_PROMPT_CACHING=true` in `.env`
- Saves 90% on cached tokens (typically $0.0008 vs $0.008 per cached input)
- Context pruning at 50K tokens to prevent quadratic growth

**Implementation:**
See `src/agent/agent_core.py:_create_messages()` for cache control block formatting.

---

## Key File Locations

### Pipeline Core
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/config.py` - Shared configuration classes
- `src/cost_tracker.py` - API cost tracking
- `run_pipeline.py` - CLI entry point

### Phase Implementations
- `src/docling_extractor_v2.py` - PHASE 1 (hierarchy)
- `src/summary_generator.py` - PHASE 2 (summaries)
- `src/multi_layer_chunker.py` - PHASE 3 (chunking + SAC)
- `src/embedding_generator.py` - PHASE 4 (embeddings)
- `src/faiss_vector_store.py` - PHASE 4 (FAISS)
- `src/hybrid_search.py` - PHASE 5B (BM25 + RRF)
- `src/graph/` - PHASE 5A (knowledge graph)
- `src/reranker.py` - PHASE 5C (cross-encoder)
- `src/graph_retrieval.py` - PHASE 5D (graph-vector fusion)
- `src/context_assembly.py` - PHASE 6 (context prep)

### Agent (PHASE 7)
- `src/agent/cli.py` - Interactive CLI and REPL
- `src/agent/agent_core.py` - Core agent with streaming
- `src/agent/config.py` - Agent configuration
- `src/agent/validation.py` - Comprehensive validation
- `src/agent/tools/` - 17 specialized tools (see [`user_search_pipeline.html`](user_search_pipeline.html) for details)
  - `base.py` - Base classes
  - `registry.py` - Tool registry
  - `tier1_basic.py` - 6 fast tools (search, get_document_list, get_document_info, get_tool_help, list_available_tools, exact_match_search)
  - `tier2_advanced.py` - 8 quality tools (graph_search, browse_entities, compare_documents, explain_search_results, filtered_search, similarity_search, expand_context, assess_retrieval_confidence)
  - `tier3_analysis.py` - 3 analysis tools (timeline_view, summarize_section, get_stats)
  - `token_manager.py` - Token estimation
  - `utils.py` - Shared utilities

### Tests
- `tests/agent/` - Agent tests (49 tests)
- `tests/graph/` - Knowledge graph tests
- `tests/test_phase*.py` - Pipeline phase tests
- `tests/test_complete_pipeline.py` - End-to-end integration

---

## Common Development Tasks

### Adding a New Document Format

1. Update `DoclingExtractorV2` in `src/docling_extractor_v2.py`
2. Add format to `supported_formats` list in `IndexingPipeline.index_document()`
3. Add integration test in `tests/test_complete_pipeline.py`

### Modifying Chunk Size

**WARNING:** Changing chunk size invalidates all existing vector stores.

```python
# In .env or IndexingConfig
CHUNK_SIZE=500  # Research-optimal, don't change without testing

# If you must change:
# 1. Re-index ALL documents
# 2. Update tests to expect new chunk counts
# 3. Benchmark retrieval quality (Precision@k, Recall@k)
```

### Adding a New Embedding Model

1. Add pricing to `src/cost_tracker.py:PRICING`
2. Update `EmbeddingGenerator._get_provider()` in `src/embedding_generator.py`
3. Add platform-specific guidance to `.env.example`
4. Test on your platform (especially Windows DLL issues)

### Extending the Agent with New Tools

See "Agent Tool Guidelines" above. Example:

```python
# In src/agent/tools/tier1_basic.py

class MyToolInput(ToolInput):
    """Input schema with validation."""
    query: str = Field(..., description="User query")
    limit: int = Field(10, ge=1, le=100)

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    tier = 1
    input_schema = MyToolInput

    def execute_impl(self, query: str, limit: int = 10) -> ToolResult:
        # Implementation
        results = self.vector_store.search(query, k=limit)

        return ToolResult(
            success=True,
            data=results,
            citations=[...],
            metadata={"query": query, "count": len(results)}
        )
```

### Debugging Retrieval Issues

1. **Enable debug mode:**
   ```bash
   uv run python -m src.agent.cli --debug
   ```

2. **Check vector store stats:**
   ```python
   from src.hybrid_search import HybridVectorStore
   store = HybridVectorStore.load("vector_db")
   print(store.get_stats())
   ```

3. **Inspect search results:**
   ```python
   # In agent, use explain_search_results tool
   # Shows BM25, Dense, RRF score breakdown
   ```

4. **Verify embeddings:**
   ```python
   from src.embedding_generator import EmbeddingGenerator
   embedder = EmbeddingGenerator()

   # Check embedding cache stats
   stats = embedder.get_cache_stats()
   print(f"Hit rate: {stats['hit_rate']:.1%}")
   ```

### Testing Strategy

**Unit Tests:**
- Test individual components (extractors, chunkers, embedders)
- Mock external API calls
- Focus on edge cases and error handling

**Integration Tests:**
- Test phase interactions (extraction → chunking → embedding)
- Use small test documents (<10 pages)
- Verify intermediate outputs

**Agent Tests:**
- Test tool execution and validation
- Test streaming and error recovery
- Test prompt caching behavior

**Run tests before PR:**
```bash
# Full test suite
uv run pytest tests/ -v

# Fast feedback loop (unit tests only)
uv run pytest tests/agent/ tests/graph/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Debug & Optimization System (/debug-optimize)

MY_SUJBOT includes a sophisticated multi-agent debugging system accessible via the `/debug-optimize` slash command. This system uses 5 specialized agents running in parallel to find bugs, optimize costs, and automatically apply fixes.

**When to Use:**
- Agent errors or tool failures
- High API costs or cache misses
- Pipeline failures or phase transition errors
- Validation errors or configuration issues
- Performance problems or slow retrieval

**Usage:**
```bash
/debug-optimize
[Paste your error description and conversation context here]

Example:
Error: Cache not working - showing 0 cached tokens
Conversation:
> What is RAG?
💰 This message: $0.0025
  Input (cached): 0 tokens
...
```

**System Architecture:**

The command orchestrates 5 specialized agents:

1. **cost-optimizer** (Haiku) - Cost optimization expert
   - Finds redundant context copying
   - Identifies cache misses and inefficient API usage
   - Optimizes token consumption
   - Tools: Read, Grep, Glob, Bash

2. **rag-debugger** (Haiku + WebSearch) - RAG pipeline expert
   - Debugs retrieval, chunking, embedding issues
   - Fixes FAISS index problems
   - Validates research constraints
   - Tools: Read, Grep, Glob, Bash, WebSearch

3. **validation-expert** (Haiku) - Configuration & validation expert
   - Fixes API key errors
   - Resolves schema mismatches
   - Debugs type checking issues
   - Tools: Read, Grep, Glob, Bash

4. **pipeline-expert** (Haiku) - Pipeline orchestration expert
   - Fixes phase transition errors
   - Debugs speed mode issues
   - Validates data flow between phases
   - Tools: Read, Grep, Glob, Bash

5. **agent-expert** (Haiku + WebSearch) - Agent framework expert
   - Debugs tool execution failures
   - Fixes streaming errors
   - Resolves Claude SDK integration bugs
   - Tools: Read, Grep, Glob, Bash, WebSearch

**Agent Routing:**
- Cost issues → cost-optimizer
- Retrieval/indexing bugs → rag-debugger
- Config/validation → validation-expert
- Pipeline failures → pipeline-expert
- Agent/tool errors → agent-expert

**Auto-Fix Process:**
1. Parses error and conversation context
2. Routes to relevant agents (parallel execution)
3. Agents analyze and generate fix plans
4. Detects and resolves conflicts
5. Auto-applies fixes (up to 20 per run)
6. Runs validation tests
7. Git commits if all tests pass
8. Generates comprehensive report

**Safety Features:**
- Respects research constraints (won't auto-change chunk size, summary style, etc.)
- Validates fixes before applying
- Automatic rollback on test failures
- Saves fix plans to `output/debug-fixes/` if validation fails
- Max 20 fixes auto-applied (prevents runaway changes)

**Templates & Documentation:**
- `.claude/templates/debug-context.md` - Debug context extraction format
- `.claude/templates/fix-plan.md` - Fix plan structure
- `.claude/agents/*.md` - 5 specialized agent definitions
- `.claude/commands/debug-optimize.md` - Main orchestration logic

**Example Workflow:**
```
User: /debug-optimize
Error: Tool validation failing - k > 100

System:
1. Parses error → type: agent, component: tools
2. Routes to agent-expert + validation-expert (parallel)
3. Agents analyze:
   - agent-expert: "Schema too strict for use case"
   - validation-expert: "Same issue - k max should be 200"
4. Merges identical fixes (high confidence)
5. Applies: k: int = Field(10, ge=1, le=200)
6. Runs: pytest tests/agent/tools/test_tier1_basic.py
7. Commits: "fix: Increase k parameter max to 200"
8. Reports: ✅ 1 fix applied, tests passed
```

**Cost Optimization Features:**
- Identifies duplicate embeddings
- Finds missing prompt cache control
- Detects redundant tool results
- Optimizes context assembly patterns
- Estimates savings per fix

---

## Troubleshooting

### Common Issues

**1. Vector store not found**
```bash
# Run indexing first
uv run python run_pipeline.py data/your_documents/
```

**2. PyTorch DLL errors (Windows)**
```bash
# Install PyTorch FIRST before uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. API key errors**
```bash
# Check .env file
cat .env | grep API_KEY

# Set for current session
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

**4. Out of memory (embedding generation)**
```python
# Reduce batch size in .env or config
EMBEDDING_BATCH_SIZE=32  # Default is 100
```

**5. Knowledge graph construction fails**
```bash
# Check API key for KG LLM provider
echo $OPENAI_API_KEY  # or $ANTHROPIC_API_KEY

# Disable KG if not needed
ENABLE_KNOWLEDGE_GRAPH=false
```

**6. Agent tools failing**
- Check validation output on startup (shows which tools are available)
- Missing reranker → Tier 2 tools degraded
- Missing knowledge graph → Tier 3 graph tools unavailable

**7. GPT-5 API Compatibility Issues**

GPT-5 models (gpt-5-nano, gpt-5-*, o1-*, o3-*) have **breaking changes** compared to GPT-4:

**Error: "Unsupported parameter: 'max_tokens'"**
```python
# WRONG (GPT-4 style):
response = client.chat.completions.create(
    model="gpt-5-nano",
    max_tokens=300  # ❌ Not supported
)

# CORRECT (GPT-5 style):
response = client.chat.completions.create(
    model="gpt-5-nano",
    max_completion_tokens=300  # ✅ Use this instead
)
```

**Error: "Unsupported value: 'temperature' does not support 0.7"**
```python
# WRONG:
response = client.chat.completions.create(
    model="gpt-5-nano",
    temperature=0.7  # ❌ Not supported
)

# CORRECT:
response = client.chat.completions.create(
    model="gpt-5-nano"
    # ✅ Use default temperature (1.0) - don't set parameter
)
```

**Implementation Pattern:**
```python
# Auto-detect model and use correct parameters
params = {"model": model, "messages": messages}

if model.startswith(("gpt-5", "o1-", "o3-")):
    # GPT-5/o-series models
    params["max_completion_tokens"] = 300
    # Don't set temperature (only default 1.0 supported)
else:
    # GPT-4 and earlier
    params["max_tokens"] = 300
    params["temperature"] = 0.7

response = client.chat.completions.create(**params)
```

**Where to check:**
- Query expansion: `src/agent/query_expander.py` ✅ Already fixed
- Summary generation: `src/summary_generator.py` (if using GPT-5)
- Knowledge graph: `src/graph/` (if using GPT-5)
- Any custom LLM calls using OpenAI API

### Performance Optimization

**Indexing Speed:**
- Fast mode (default): ThreadPoolExecutor, 2-3 min per doc
- Eco mode: OpenAI Batch API, 15-30 min per doc, 50% cheaper
- Use eco mode for overnight bulk indexing

**Agent Response Speed:**
- Enable prompt caching: 90% cost reduction on repeated queries
- Use TIER 1 tools first (fast, <100ms)
- Lazy load reranker to speed up startup

**Memory Usage:**
- Local embeddings (bge-m3): ~2GB RAM
- Cloud embeddings: Minimal memory (<500MB)
- FAISS indexes: ~10MB per 1000 documents

---

## Resources

### Documentation
- **README.md** - User guide and quick start
- **PIPELINE.md** - Complete pipeline specification with research
- **INSTALL.md** - Platform-specific installation
- **docs/agent/README.md** - Agent CLI documentation
- **docs/cost-tracking.md** - Cost optimization guide

### Research Papers
1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS chunking, reranking
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending

### API Documentation
- Anthropic Claude: https://docs.anthropic.com/
- OpenAI: https://platform.openai.com/docs
- IBM Docling: https://ds4sd.github.io/docling/

---

## Code Style & Conventions

**Formatting:**
- Black (line length: 100)
- isort (profile: black)
- Type hints encouraged (mypy compatible)

**Naming:**
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

**Documentation:**
- Docstrings: Google style
- Type hints: Required for public APIs
- Comments: Explain WHY, not WHAT

**Logging:**
```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed diagnostic info")
logger.info("High-level progress")
logger.warning("Degraded mode or non-critical issues")
logger.error("Errors that don't crash the program")
```

---

## Version & Status

**Last Updated:** 2025-11-03
**Status:** PHASE 1-7 COMPLETE ✅ + Hierarchical Summary Generation ✅ + Query Expansion ✅ + RAG Confidence Scoring ✅ + Interactive Visual Documentation ✅
**Agent Tools:** 17 (6 basic + 8 advanced + 3 analysis)
**Pipeline:** Full SOTA 2025 RAG (Hybrid + Reranking + Graph + Query Expansion + Confidence Scoring + Hierarchical Summaries)
**Visual Documentation:**
- [`indexing_pipeline.html`](indexing_pipeline.html) - Detailed indexing process (Phase 1-5)
- [`user_search_pipeline.html`](user_search_pipeline.html) - User query flow with complete tool breakdown (Phase 7)

**Recent Updates:**
- **Hierarchical Summary Generation from Chunk Contexts (2025-11-03):** Revolutionary 3-phase PHASE 3 architecture eliminates truncation problem
  - **Problem Fixed:** OLD approach truncated section text to 2000 chars → only 40% coverage for 5000+ char sections → poor retrieval quality
  - **Solution:** NEW hierarchical approach generates section summaries FROM chunk contexts (100% coverage, no truncation!)
  - **Architecture Changes:**
    - **PHASE 2:** Now generates ONLY document summary (section summaries deferred to PHASE 3B)
    - **PHASE 3A:** Chunk context generation via Contextual Retrieval (uses doc summary + section title + neighbors)
    - **PHASE 3B:** Section summary generation FROM all chunk contexts (full coverage, no truncation!)
    - **PHASE 3C:** Summary validation (all summaries >= 50 chars, reports issues with warnings)
  - **Quality Improvements:**
    - 100% section coverage (vs 40% with old truncation approach)
    - Better retrieval for long sections (>2000 chars)
    - Estimated +25% retrieval success rate for queries targeting end of long sections
  - **Implementation:**
    - `src/multi_layer_chunker.py:_generate_section_summaries_from_contexts()` - New hierarchical generator
    - `src/multi_layer_chunker.py:_validate_summaries()` - Quality validation (>= 50 chars)
    - `src/docling_extractor_v2.py:_generate_document_summary_from_text()` - Document summary only
    - Modified pipeline flow: chunks FIRST → section summaries SECOND → validation THIRD
  - **Breaking Change:** Section summaries no longer generated in PHASE 2 (deferred to PHASE 3B for better quality)
  - **Backward Compatibility:** Old `_generate_document_summary()` method kept for compatibility (marked DEPRECATED)
  - **Tests:** New comprehensive tests in `tests/test_phase3_summary_unit.py` (validation + generation logic)
- Entity Deduplication System (2025-11-01): Sophisticated 3-layer incremental deduplication for Neo4j knowledge graphs
  - **3-Layer Strategy:** Layer 1 (exact match, <1ms) → Layer 2 (semantic similarity, 50-200ms) → Layer 3 (acronym expansion, 100-500ms)
  - **APOC Optimization:** Uses `apoc.coll.union` when available, automatic fallback to pure Cypher
  - **Property Merging:** MAX confidence, UNION chunks, deep metadata merge with `merged_from` tracking
  - **Built-in Acronyms:** 29 legal/sustainability acronyms (GRI, ISO, GDPR, OSHA, HSE, etc.) + custom acronym support
  - **Configuration:** Fully configurable via `.env` (all layers optional)
  - **Recommended Mode:** Layer 1 + Layer 3 (production balanced mode) - 98% precision, 85% recall
  - **Components:** 5 new files (`neo4j_deduplicator.py`, `similarity_detector.py`, `acronym_expander.py`, enhanced `deduplicator.py`, updated `config.py`)
  - **Integration:** Automatic activation in `Neo4jGraphBuilder` when enabled
  - **Test Coverage:** 28 comprehensive tests covering all layers and edge cases
- Graph Boost & Prompt Optimization (2025-10-30): Enhanced search tool with optional graph boosting + SOTA prompt engineering
  - **Graph Boost Feature:** Optional `enable_graph_boost` parameter for `search` tool (+8% factual correctness on entity queries)
  - **Dual Boost Strategy:** Entity mention boosting (+30%) + centrality boosting (+15%) based on HybridRAG 2024 research
  - **Performance:** Adds +200-500ms overhead when enabled, graceful fallback to hybrid search if graph unavailable
  - **Prompt Engineering:** Rewrote agent system prompt with Constitutional AI constraints (5 rules) and Chain-of-Thought framework
  - **Entity/Relationship Extractors:** Enhanced with explicit reasoning steps and confidence assessment guidelines
  - **Cleanup:** Removed 7 deprecated prompt files (hyde, query_decomposition, contextual_retrieval, summary_*.txt)
  - **Tier Classification:** SearchTool remains TIER 1 (200-300ms baseline) with optional features extending to 400-1500ms
- CLI + WebApp Backend Unification (2025-10-30): Both now use identical GraphAdapter with Neo4j
  - **Problem Fixed:** WebApp used in-memory KnowledgeGraph (JSON files), CLI used Neo4j
  - **Solution:** Modified `backend/agent_adapter.py` to check `KG_BACKEND=neo4j` (same as CLI)
  - **Result:** `browse_entities` and all graph tools now work identically in both interfaces
  - **Configuration:** Set `KG_BACKEND=neo4j` in `.env` (recommended for production)
  - **Fallback:** Automatic fallback to JSON if Neo4j connection fails
- Browse Entities Tool (2025-10-30): Added `browse_entities` tool for entity discovery (17 tools now)
  - New Tier 2 tool: `browse_entities` - Discover entities by type, confidence, or search term
  - Enables exploratory queries like "list all regulations" or "show high-confidence standards about waste"
  - Uses GraphAdapter.find_entities() for efficient indexed Neo4j queries
  - Complements `graph_search` (browse to discover entities → graph_search to explore relationships)
  - Full test coverage: 13 comprehensive tests covering all filter combinations and edge cases
- RAG Confidence Scoring (2025-10-29): Added 7-metric confidence scoring system (15→16 tools)
  - New tool: `assess_retrieval_confidence` - Explicit confidence assessment
  - Automatic confidence display in search results
  - Legal compliance thresholds: HIGH (≥0.85), MEDIUM (0.70-0.84), LOW (0.50-0.69)
  - Comprehensive error handling and validation fixes applied
- Tool Consolidation (2025-10-29): Reduced from 16 to 14 tools by consolidating duplicate functionality
  - `graph_search` (4 modes) replaces `multi_hop_search` + `entity_tool`
  - `filtered_search` (3 search methods) replaces `exact_match_search`
  - Complete BFS multi-hop implementation (no longer stubbed)
- Visual Documentation (2025-10-26): Interactive HTML visualizations for indexing and search pipelines
- Query Expansion (2025-10-26): Multi-query generation with `num_expands` parameter (research-based +15-25% recall improvement)

**Note:** Never add vector_db/ to .gitignore - it contains tracked merged vector stores.