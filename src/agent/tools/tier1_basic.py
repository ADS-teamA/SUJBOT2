"""
TIER 1: Basic Retrieval Tools

Fast tools (100-300ms baseline) for common retrieval tasks.
Some tools have optional features that may extend performance beyond baseline.
These should handle 80% of user queries.
"""

import logging
from typing import List, Literal, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


# === Tool 0: Get Tool Help (Meta Tool) ===


class GetToolHelpInput(ToolInput):
    """Input for get_tool_help tool."""

    tool_name: str = Field(
        ..., description="Name of tool to get help for (e.g., 'search', 'multi_doc_synthesizer')"
    )


@register_tool
class GetToolHelpTool(BaseTool):
    """Get detailed documentation for a specific tool."""

    name = "get_tool_help"
    description = "Get detailed help for any tool"
    detailed_help = """
    Returns comprehensive documentation for a specific tool including:
    - Full description and use cases
    - All parameters with types and defaults
    - Examples of when to use this tool
    - Performance characteristics (tier, speed)

    Use this whenever you need to understand a tool's capabilities or parameters
    before using it for the first time.
    """
    tier = 1
    input_schema = GetToolHelpInput

    def execute_impl(self, tool_name: str) -> ToolResult:
        """Get detailed help for a tool."""
        from .registry import get_registry

        registry = get_registry()

        # Check if tool exists
        if tool_name not in registry._tool_classes:
            available_tools = sorted(registry._tool_classes.keys())
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools[:10])}...",
                metadata={"requested_tool": tool_name, "available_count": len(available_tools)},
            )

        # Get tool class
        tool_class = registry._tool_classes[tool_name]

        # Build detailed help
        help_text = f"# {tool_class.name}\n\n"
        help_text += f"**Tier:** {tool_class.tier} "
        help_text += f"({'Basic/Fast' if tool_class.tier == 1 else 'Advanced' if tool_class.tier == 2 else 'Analysis/Slow'})\n\n"

        # Description
        help_text += f"**Description:** {tool_class.description}\n\n"

        # Detailed help if available
        if tool_class.detailed_help:
            help_text += f"**Details:**\n{tool_class.detailed_help.strip()}\n\n"

        # Parameters from Pydantic schema
        if tool_class.input_schema and tool_class.input_schema != ToolInput:
            help_text += "**Parameters:**\n"
            schema = tool_class.input_schema.model_json_schema()

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "No description")
                is_required = "✓ Required" if param_name in required else "Optional"
                default = param_info.get("default", "N/A")

                help_text += f"- `{param_name}` ({param_type}) - {is_required}\n"
                help_text += f"  {param_desc}\n"
                if default != "N/A":
                    help_text += f"  Default: {default}\n"
                help_text += "\n"

        # Requirements
        requirements = []
        if tool_class.requires_kg:
            requirements.append("Knowledge Graph")
        if tool_class.requires_reranker:
            requirements.append("Reranker")

        if requirements:
            help_text += f"**Requires:** {', '.join(requirements)}\n\n"

        return ToolResult(
            success=True,
            data={
                "tool_name": tool_name,
                "tier": tool_class.tier,
                "help_text": help_text,
                "short_description": tool_class.description,
                "requires_kg": tool_class.requires_kg,
                "requires_reranker": tool_class.requires_reranker,
            },
            metadata={"tool": tool_name},
        )


# === Tool 1: Unified Search (with Query Expansion) ===


class SearchInput(ToolInput):
    """Input for unified search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(5, description="Number of results to return (3-5 recommended)", ge=1, le=10)
    num_expands: int = Field(
        0,
        description="Number of query paraphrases to generate: 0=original query only (fast, ~200ms), 1=original+1 paraphrase (2 queries total, ~500ms), 2=original+2 paraphrases (3 queries total, ~800ms). Total queries = num_expands + 1. Warning: num_expands > 3 may impact performance",
        ge=0,
        le=5,
    )
    enable_graph_boost: bool = Field(
        False,
        description="Enable knowledge graph boosting for entity-centric queries. Boosts chunks mentioning query entities (+30%) and high-centrality concepts (+15%). Use when query mentions specific entities (organizations, standards, regulations). Performance overhead: +200-500ms. Default: False (performance-first).",
    )
    use_hyde: bool = Field(
        False,
        description="Enable HyDE (Hypothetical Document Embeddings) for better zero-shot retrieval. Generates a hypothetical answer to improve semantic matching. Slower (~1-2s) but higher quality for ambiguous queries.",
    )


@register_tool
class SearchTool(BaseTool):
    """Unified search with optional query expansion."""

    name = "search"
    description = "Unified hybrid search with optional query expansion and graph boosting"
    detailed_help = """
    Unified search tool combining hybrid retrieval (BM25 + Dense + RRF) with optional
    query expansion and knowledge graph boosting for improved recall and precision.

    **Query Expansion:**
    - num_expands=0: Use original query only (fast, ~200-300ms, 1 query total)
    - num_expands=1: Original + 1 paraphrase (~500ms, 2 queries total, +10-15% recall)
    - num_expands=2: Original + 2 paraphrases (~800ms, 3 queries total, +15-25% recall)
    - num_expands=3: Original + 3 paraphrases (~1.2s, 4 queries total, +20-30% recall)

    **Graph Boosting:**
    - enable_graph_boost=False: Standard hybrid search (default, faster)
    - enable_graph_boost=True: Graph-enhanced search (+200-500ms, better for entity queries)
        - Boosts chunks mentioning query entities (+30% score boost)
        - Boosts chunks with high-centrality entities (+15% score boost)
        - Research-backed: +8% factual correctness on entity-centric queries (HybridRAG 2024)

    **When to use:**
    - Most queries: Start with defaults (num_expands=0, enable_graph_boost=False)
    - Entity-centric queries: Use enable_graph_boost=True (e.g., "What did GSSB issue?")
    - Ambiguous queries: Use num_expands=1-2 for better recall
    - Complex entity queries: Combine both (num_expands=1, enable_graph_boost=True)
    - Speed critical: Keep both disabled (default)

    **When NOT to use graph boosting:**
    - Generic queries without specific entities (e.g., "What is waste management?")
    - Keyword-based searches (graph boost has no effect)
    - Speed is priority (adds ~300ms overhead)

    **Method:**
    1. Query expansion (if num_expands > 0): Generate N paraphrases using LLM
    2. Hybrid search for each query:
       - If enable_graph_boost=True: Use GraphEnhancedRetriever (entity extraction + boosting)
       - If enable_graph_boost=False: Use standard hybrid search (BM25 + Dense + RRF)
    3. RRF fusion across queries (if multiple queries)
    4. Cross-encoder reranking (final quality boost)

    **Performance:**
    - num_expands=0, graph_boost=False: ~200-300ms (fastest, default)
    - num_expands=0, graph_boost=True: ~400-700ms (entity-enhanced)
    - num_expands=1, graph_boost=False: ~500ms (better recall)
    - num_expands=1, graph_boost=True: ~700-1000ms (best quality for entity queries)
    - num_expands=2, graph_boost=True: ~1.0-1.5s (maximum quality, slower)

    **Best practices:**
    - Start with defaults for most queries (fast path)
    - Enable graph boost for entity-focused questions (organizations, standards, regulations)
    - Use num_expands=1-2 when user query is ambiguous or recall is critical
    - Combine both features for complex entity queries where quality > speed
    - Check metadata.graph_boost_enabled to verify if boost was applied
    """
    tier = 1  # Tier 1: Fast baseline (200-300ms). Optional features (graph_boost, num_expands) extend to 400-1500ms
    input_schema = SearchInput
    requires_reranker = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_expander = None  # Lazy initialization
        self._hyde_generator = None  # Lazy initialization

    def _get_query_expander(self):
        """Lazy initialization of QueryExpander."""
        if self._query_expander is None:
            import os
            from ..query_expander import QueryExpander

            # Get provider and model from config (self.config is ToolConfig)
            provider = self.config.query_expansion_provider
            model = self.config.query_expansion_model

            # Get API keys from environment variables
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            try:
                self._query_expander = QueryExpander(
                    provider=provider,
                    model=model,
                    anthropic_api_key=anthropic_key,
                    openai_api_key=openai_key,
                )
                logger.info(f"QueryExpander initialized: provider={provider}, model={model}")
            except ValueError as e:
                # Configuration error: missing API key OR invalid provider
                # (QueryExpander __init__ validates both)
                logger.warning(
                    f"QueryExpander configuration error: {e}. "
                    f"Query expansion will be disabled. "
                    f"Common causes: missing API key (ANTHROPIC_API_KEY or OPENAI_API_KEY), "
                    f"or unsupported provider (only 'openai' and 'anthropic' supported)."
                )
                self._query_expander = None
            except ImportError as e:
                # Missing package: openai (for provider='openai') or anthropic (for provider='anthropic')
                package_name = "openai" if provider == "openai" else "anthropic"
                logger.warning(
                    f"QueryExpander package missing: {e}. "
                    f"Query expansion will be disabled. Install required package: "
                    f"'uv pip install {package_name}' (for provider='{provider}')"
                )
                self._query_expander = None
            except Exception as e:
                # Unexpected error: any exception OTHER than ValueError (config) or ImportError (missing package)
                # Examples: network errors, attribute errors from malformed config, type errors
                # This catch-all prevents SearchTool from crashing if QueryExpander has bugs
                logger.error(
                    f"Unexpected error initializing QueryExpander ({type(e).__name__}): {e}. "
                    f"Query expansion will be disabled. This may indicate a bug in QueryExpander or SearchTool."
                )
                self._query_expander = None

        return self._query_expander

    def _get_hyde_generator(self):
        """Lazy initialization of HyDEGenerator."""
        if self._hyde_generator is None:
            import os
            from ..hyde_generator import HyDEGenerator

            # Get provider and model from config (reuse query expansion settings)
            provider = self.config.query_expansion_provider
            model = self.config.query_expansion_model

            # Get API keys from environment variables
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            try:
                self._hyde_generator = HyDEGenerator(
                    provider=provider,
                    model=model,
                    anthropic_api_key=anthropic_key,
                    openai_api_key=openai_key,
                    num_hypotheses=3,  # Multi-hypothesis averaging
                )
                logger.info(f"HyDEGenerator initialized: provider={provider}, model={model}")
            except ValueError as e:
                logger.warning(
                    f"HyDEGenerator configuration error: {e}. "
                    f"HyDE will be disabled. "
                    f"Common causes: missing API key or unsupported provider."
                )
                self._hyde_generator = None
            except ImportError as e:
                package_name = "openai" if provider == "openai" else "anthropic"
                logger.warning(
                    f"HyDEGenerator package missing: {e}. "
                    f"HyDE will be disabled. Install: 'uv pip install {package_name}'"
                )
                self._hyde_generator = None
            except Exception as e:
                logger.error(
                    f"Unexpected error initializing HyDEGenerator ({type(e).__name__}): {e}. "
                    f"HyDE will be disabled."
                )
                self._hyde_generator = None

        return self._hyde_generator

    def execute_impl(
        self, query: str, k: int = 5, num_expands: int = 0, enable_graph_boost: bool = False, use_hyde: bool = False
    ) -> ToolResult:
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # === STEP 0: HyDE Generation (Multi-Hypothesis) ===
        hyde_docs = []
        hyde_metadata = {"use_hyde": use_hyde, "num_hypotheses": 0, "generation_method": "none"}

        if use_hyde:
            hyde_gen = self._get_hyde_generator()

            if hyde_gen:
                try:
                    # Generate 3 hypothetical docs for original query (multi-hypothesis averaging)
                    hyde_result = hyde_gen.generate(query, num_docs=3)
                    hyde_docs = hyde_result.hypothetical_docs

                    hyde_metadata.update(
                        {
                            "num_hypotheses": len(hyde_docs),
                            "generation_method": hyde_result.generation_method,
                            "model_used": hyde_result.model_used,
                        }
                    )

                    # Warn if fallback was used
                    if hyde_result.generation_method == "fallback":
                        logger.warning(
                            "HyDE generation failed internally (LLM error). "
                            "Falling back to standard search."
                        )
                    else:
                        logger.info(
                            f"HyDE generated {len(hyde_docs)} hypothetical docs for original query"
                        )
                except Exception as e:
                    logger.warning(f"HyDE generation failed: {e}. Falling back to standard search.")
            else:
                logger.warning(
                    "HyDE requested but generator not available. Falling back to standard search."
                )

        # === STEP 1: Query Expansion ===
        queries = [query]  # Default: original query only
        expansion_metadata = {"num_expands": num_expands, "expansion_method": "none"}

        if num_expands > 0:
            # Attempt query expansion
            expander = self._get_query_expander()

            if expander:
                try:
                    # num_expands = number of paraphrases to generate
                    # Result will be [original] + num_expands paraphrases
                    expansion_result = expander.expand(query, num_expansions=num_expands)
                    queries = expansion_result.expanded_queries
                    expansion_metadata.update(
                        {
                            "expansion_method": expansion_result.expansion_method,
                            "model_used": expansion_result.model_used,
                            "queries_generated": len(queries),
                            "paraphrases_requested": num_expands,
                            "paraphrases_generated": len(queries) - 1,  # Exclude original
                        }
                    )
                    logger.info(
                        f"Query expansion: original='{query}' + {num_expands} paraphrases "
                        f"→ {len(queries)} queries total: {queries}"
                    )
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}. Using original query only.")
                    # Fallback to original query
                    queries = [query]
                    expansion_metadata["expansion_method"] = "failed"
            else:
                logger.warning("QueryExpander not available. Using original query only.")
                expansion_metadata["expansion_method"] = "unavailable"
        else:
            logger.debug(f"No expansion (num_expands=0): using original query only")

        # === STEP 2: Hybrid Search for Each Query ===
        # Retrieve more candidates for reranking
        candidates_k = k * 3 if self.reranker else k

        all_chunks = []  # All chunks from all queries (with their original scores)
        search_metadata = []  # Track metadata for each query search

        # Determine if graph boost should be used
        use_graph_boost = enable_graph_boost and self.graph_retriever is not None

        if enable_graph_boost and self.graph_retriever is None:
            logger.info(
                "Graph boost requested but graph_retriever not available. "
                "Falling back to standard hybrid search. "
                "Tip: Run indexing with ENABLE_KNOWLEDGE_GRAPH=true to enable graph boosting."
            )

        if use_graph_boost:
            # NEW: Graph-enhanced retrieval path
            logger.info("Using graph-enhanced retrieval (graph boost enabled)")

            for idx, q in enumerate(queries, 1):
                logger.info(f"Searching with graph boost (query {idx}/{len(queries)}): '{q}'")

                # Embed query (extract 1D array for graph retriever)
                # Use HyDE docs if available AND this is the original query
                if hyde_docs and q == query:
                    # Multi-hypothesis averaging: embed all HyDE docs and average
                    import numpy as np
                    embeddings = self.embedder.embed_texts(hyde_docs)
                    query_embedding = np.mean(embeddings, axis=0)
                    logger.info(f"Using {len(hyde_docs)} HyDE docs (multi-hypothesis averaging) for query {idx}")
                else:
                    # Standard embedding
                    query_embedding = self.embedder.embed_texts([q])[0]

                # Delegate to GraphEnhancedRetriever with error handling
                try:
                    results = self.graph_retriever.search(
                        query=q,
                        query_embedding=query_embedding,
                        k=candidates_k,
                        enable_graph_boost=True,
                    )

                    chunks = results.get("layer3", [])

                    if not chunks:
                        logger.warning(
                            f"Graph boost returned empty results for query {idx}. "
                            f"Falling back to standard search for this query."
                        )
                        # Fallback to standard hybrid search for this query
                        results = self.vector_store.hierarchical_search(
                            query_embedding=query_embedding,
                            k_layer3=candidates_k,
                        )
                        chunks = results["layer3"]
                        graph_boost_applied = False
                    else:
                        graph_boost_applied = True

                except (KeyError, AttributeError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Graph boost failed for query {idx} ({type(e).__name__}: {e}). "
                        f"Falling back to standard search for this query."
                    )
                    # Fallback to standard hybrid search
                    results = self.vector_store.hierarchical_search(
                        query_embedding=query_embedding,
                        k_layer3=candidates_k,
                    )
                    chunks = results["layer3"]
                    graph_boost_applied = False
                # Tag chunks with their source query for tracking
                for chunk in chunks:
                    chunk["_source_query"] = q

                all_chunks.extend(chunks)

                # Track search metadata
                search_metadata.append(
                    {
                        "query": q,
                        "chunks_retrieved": len(chunks),
                        "graph_boost_applied": graph_boost_applied,
                    }
                )

                logger.info(f"Query {idx} retrieved {len(chunks)} graph-boosted chunks from layer3")

        else:
            # EXISTING: Standard hybrid search path
            logger.debug(
                f"Using standard hybrid search (graph_boost_enabled={enable_graph_boost}, "
                f"graph_retriever_available={self.graph_retriever is not None})"
            )

            for idx, q in enumerate(queries, 1):
                logger.info(f"Searching with query {idx}/{len(queries)}: '{q}'")

                # Embed query (extract 1D array)
                # Use HyDE docs if available AND this is the original query
                if hyde_docs and q == query:
                    # Multi-hypothesis averaging: embed all HyDE docs and average
                    import numpy as np
                    embeddings = self.embedder.embed_texts(hyde_docs)
                    query_embedding = np.mean(embeddings, axis=0)
                    logger.info(f"Using {len(hyde_docs)} HyDE docs (multi-hypothesis averaging) for query {idx}")
                else:
                    # Standard embedding
                    query_embedding = self.embedder.embed_texts([q])[0]

                # Hybrid search
                results = self.vector_store.hierarchical_search(
                    query_embedding=query_embedding,
                    k_layer3=candidates_k,
                )

                chunks = results["layer3"]
                # Tag chunks with their source query for tracking
                for chunk in chunks:
                    chunk["_source_query"] = q

                all_chunks.extend(chunks)

                # Track search metadata
                search_metadata.append(
                    {
                        "query": q,
                        "chunks_retrieved": len(chunks),
                        "graph_boost_applied": False,
                    }
                )

                logger.info(f"Query {idx} retrieved {len(chunks)} chunks from layer3")

        # Log total candidates before fusion/reranking
        logger.info(
            f"Total candidates from all queries: {len(all_chunks)} chunks "
            f"(from {len(queries)} queries)"
        )

        # === STEP 3: RRF Fusion (if multiple queries) ===
        if len(queries) > 1:
            # Use RRF to combine results from multiple queries
            chunks_before_fusion = len(all_chunks)
            chunks = self._rrf_fusion(all_chunks, k=candidates_k)
            fusion_method = "rrf"
            logger.info(
                f"RRF fusion: {chunks_before_fusion} candidates → {len(chunks)} "
                f"unique chunks (deduped and reranked by RRF)"
            )
        else:
            # Single query: use chunks as-is
            chunks = all_chunks[:candidates_k]
            fusion_method = "none"

        # Get document filter info (if available)
        document_filter = "none"
        if chunks and "document_id" in chunks[0]:
            # Check if results are filtered to specific documents
            unique_docs = set(c.get("document_id") for c in chunks)
            if len(unique_docs) == 1:
                document_filter = list(unique_docs)[0]
            elif len(unique_docs) <= 3:
                document_filter = ", ".join(sorted(unique_docs))

        # === STEP 4: Reranking ===
        chunks_before_rerank = len(chunks)
        if self.reranker and len(chunks) > k:
            logger.info(f"Reranking {len(chunks)} candidates to top {k}...")
            chunks = self.reranker.rerank(query=query, candidates=chunks, top_k=k)
            reranking_applied = True
            logger.info(f"Reranking complete: {chunks_before_rerank} → {len(chunks)} chunks")
        else:
            chunks = chunks[:k]
            reranking_applied = False
            logger.debug(f"No reranking applied, using top {len(chunks)} chunks")

        # === STEP 5: Format Results ===
        formatted = [format_chunk_result(c) for c in chunks]
        final_count = len(formatted)

        # Generate citations with breadcrumb path (uses generate_citation for consistency)
        citations = [generate_citation(c, i + 1, format="inline") for i, c in enumerate(formatted)]

        # === STEP 6: RAG Confidence Scoring ===
        try:
            from src.agent.rag_confidence import RAGConfidenceScorer

            confidence_scorer = RAGConfidenceScorer()
            confidence = confidence_scorer.score_retrieval(chunks, query=query)

            logger.info(
                f"RAG Confidence: {confidence.interpretation} ({confidence.overall_confidence:.3f})"
            )

            # Add warning to citations if low confidence
            if confidence.should_flag:
                citations.insert(0, f"⚠️ {confidence.interpretation}")

        except (AttributeError, KeyError) as e:
            # Data structure issue - chunks missing required fields
            logger.error(
                f"RAG confidence scoring failed - data structure error: {e}. "
                f"Chunks may be missing required fields (bm25_score, dense_score, etc.). "
                f"Continuing without confidence data.",
                exc_info=False,
            )
            confidence = None
        except ImportError as e:
            # Should not happen in production - indicates setup error
            logger.error(
                f"RAG confidence module missing: {e}. This indicates a setup issue. "
                f"Continuing without confidence data.",
                exc_info=False,
            )
            confidence = None
        except ValueError as e:
            # Likely from numpy operations with invalid data
            logger.error(
                f"RAG confidence calculation error - invalid data: {e}. "
                f"Retrieved chunks may contain malformed scores. "
                f"Continuing without confidence data.",
                exc_info=False,
            )
            confidence = None
        except Exception as e:
            # Unexpected error - log with full traceback for debugging
            logger.error(
                f"Unexpected error in RAG confidence scoring ({type(e).__name__}): {e}. "
                f"This may indicate a bug. Continuing without confidence data.",
                exc_info=True,
            )
            confidence = None

        # Enhanced metadata with debug info
        result_metadata = {
            "query": query,
            "k": k,
            "num_expands": num_expands,
            "enable_graph_boost": enable_graph_boost,
            "graph_boost_enabled": use_graph_boost,
            "expansion_metadata": expansion_metadata,
            "fusion_method": fusion_method,
            "reranking_applied": reranking_applied,
            "method": (
                "hybrid+expansion+graph+rerank"
                if num_expands > 1 and use_graph_boost and self.reranker
                else "hybrid+graph+rerank"
                if use_graph_boost and self.reranker
                else "hybrid+expansion+rerank"
                if num_expands > 1 and self.reranker
                else "hybrid+rerank"
                if self.reranker
                else "hybrid"
            ),
            "candidates_retrieved": len(all_chunks),
            "chunks_before_rerank": chunks_before_rerank,
            "chunks_after_rerank": final_count,
            "final_count": final_count,
            "document_filter": document_filter,
            "queries_used": queries,
            "search_metadata": search_metadata,
            "hyde_metadata": hyde_metadata,
        }

        # Add RAG confidence to metadata
        if confidence:
            result_metadata["rag_confidence"] = confidence.to_dict()

        # Log final summary
        logger.info(
            f"Search complete: expanded to {len(queries)} queries, "
            f"{len(all_chunks)} total candidates, "
            f"{chunks_before_rerank} before rerank, "
            f"{final_count} final chunks returned"
        )

        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata=result_metadata,
        )

    def _rrf_fusion(self, chunks: List[dict], k: int = 60) -> List[dict]:
        """
        RRF (Reciprocal Rank Fusion) for combining results from multiple queries.

        RRF formula: score(chunk) = sum(1 / (k + rank)) for all queries where chunk appears

        Args:
            chunks: List of chunks from all queries (may contain duplicates)
            k: RRF parameter (default: 60, optimal from research)

        Returns:
            Deduplicated and reranked chunks sorted by RRF score
        """
        # Group chunks by chunk_id
        chunk_ranks = {}  # {chunk_id: [(rank, source_query), ...]}

        current_rank = {}  # {source_query: current_rank}

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            source_query = chunk.get("_source_query", "")

            if source_query not in current_rank:
                current_rank[source_query] = 0

            rank = current_rank[source_query]
            current_rank[source_query] += 1

            if chunk_id not in chunk_ranks:
                chunk_ranks[chunk_id] = {
                    "chunk": chunk,
                    "ranks": [],
                }

            chunk_ranks[chunk_id]["ranks"].append((rank, source_query))

        # Calculate RRF scores
        rrf_scores = []
        for chunk_id, data in chunk_ranks.items():
            # RRF score: sum of 1/(k + rank) for all occurrences
            rrf_score = sum(1.0 / (k + rank) for rank, _ in data["ranks"])

            rrf_scores.append(
                {
                    "chunk": data["chunk"],
                    "rrf_score": rrf_score,
                    "appearances": len(data["ranks"]),
                }
            )

        # Sort by RRF score (descending)
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Return top chunks
        return [item["chunk"] for item in rrf_scores]


# === Tool 3: Get Document List ===


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool."""

    pass  # No parameters needed


@register_tool
class GetDocumentListTool(BaseTool):
    """List all indexed documents."""

    name = "get_document_list"
    description = "List all indexed documents"
    detailed_help = """
    Returns a list of all indexed documents with their summaries.

    **When to use:**
    - User asks "what documents are available?"
    - Need to discover corpus contents
    - Before document-specific queries

    **Data source:** Vector store metadata (Layer 1 - document level)
    **Speed:** <10ms (metadata lookup only)
    """
    tier = 1
    input_schema = GetDocumentListInput

    def execute_impl(self) -> ToolResult:
        # Extract document IDs and summaries from Layer 1 metadata
        documents_map = {}  # {doc_id: summary}

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer1"):
            for meta in self.vector_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")  # Layer 1 content is the document summary
                if doc_id and doc_id not in documents_map:
                    # Only store first occurrence (all Layer 1 entries for same doc have same summary)
                    documents_map[doc_id] = summary
        elif hasattr(self.vector_store, "faiss_store"):
            for meta in self.vector_store.faiss_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")
                if doc_id and doc_id not in documents_map:
                    documents_map[doc_id] = summary

        # Fallback: If Layer 1 is empty (single-layer optimization), extract from Layer 3
        if not documents_map:
            # Try metadata_layer3 (direct FAISSVectorStore)
            if hasattr(self.vector_store, "metadata_layer3"):
                for meta in self.vector_store.metadata_layer3:
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id not in documents_map:
                        # Layer 3 doesn't have document summaries, use document title or placeholder
                        doc_title = meta.get("document_title", doc_id)
                        documents_map[doc_id] = f"Privacy policy for {doc_title}"
            # Try HybridVectorStore wrapper
            elif hasattr(self.vector_store, "faiss_store") and hasattr(
                self.vector_store.faiss_store, "metadata_layer3"
            ):
                for meta in self.vector_store.faiss_store.metadata_layer3:
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id not in documents_map:
                        doc_title = meta.get("document_title", doc_id)
                        documents_map[doc_id] = f"Privacy policy for {doc_title}"

        # Build list of document objects with id and summary
        document_list = [
            {"id": doc_id, "summary": summary} for doc_id, summary in sorted(documents_map.items())
        ]

        return ToolResult(
            success=True,
            data={"documents": document_list, "count": len(document_list)},
            metadata={"total_documents": len(document_list)},
        )


# === Tool 3: List Available Tools ===


class ListAvailableToolsInput(ToolInput):
    """Input for list_available_tools tool."""

    pass  # No parameters needed


@register_tool
class ListAvailableToolsTool(BaseTool):
    """List all available tools."""

    name = "list_available_tools"
    description = "List all available tools"
    detailed_help = """
    Returns a complete list of all available tools with short descriptions.
    For detailed help on a specific tool, use get_tool_help instead.

    **When to use:**
    - Need to see all available tools
    - Understand available capabilities
    - Select the right tool for a task

    **Best practice:** Use get_tool_help for detailed docs on specific tools.

    **Speed:** <10ms (metadata lookup)
    """
    tier = 1
    input_schema = ListAvailableToolsInput

    def execute_impl(self) -> ToolResult:
        """Get list of all available tools with metadata."""
        from .registry import get_registry

        registry = get_registry()
        all_tools = registry.get_all_tools()

        # Build tool list with metadata
        tools_list = []
        for tool in all_tools:
            # Extract input parameters from schema
            schema = tool.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # Build parameters info
            parameters = []
            for param_name, param_info in properties.items():
                param_desc = param_info.get("description", "No description")
                param_type = param_info.get("type", "unknown")
                is_required = param_name in required

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc,
                        "required": is_required,
                    }
                )

            # Extract "when to use" from description if present
            # Some tools have "Use for:" or "Use when:" in their docstring
            when_to_use = tool.description
            if hasattr(tool.__class__, "__doc__") and tool.__class__.__doc__:
                doc = tool.__class__.__doc__.strip()
                # Look for "Use for:" or "Use when:" lines
                for line in doc.split("\n"):
                    line = line.strip()
                    if line.startswith("Use for:") or line.startswith("Use when:"):
                        when_to_use = line
                        break

            # Add tier info for context (even though not grouping by tier)
            tier_label = {1: "Basic (fast)", 2: "Advanced (quality)", 3: "Analysis (deep)"}

            tools_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "when_to_use": when_to_use,
                    "tier": f"Tier {tool.tier} - {tier_label.get(tool.tier, 'Unknown')}",
                }
            )

        # Sort by name for consistent ordering
        tools_list.sort(key=lambda t: t["name"])

        return ToolResult(
            success=True,
            data={
                "tools": tools_list,
                "total_count": len(tools_list),
                "best_practices": {
                    "general": [
                        "Start with Tier 1 (fast) tools before escalating to Tier 2/3",
                        "Use 'search' for most queries (hybrid + optional expansion + rerank = best quality)",
                        "Start with num_expands=0 for speed, increase to 1-2 for better recall when needed",
                        "For complex queries, decompose into sub-tasks and use multiple tools",
                        "Try multiple retrieval strategies before giving up",
                    ],
                    "selection_strategy": {
                        "most_queries": "search (with num_expands=0 for speed, 1-2 for recall)",
                        "entity_focused": "Use 'search' with entity names, or multi_hop_search if KG available",
                        "specific_document": "Use exact_match_search or filtered_search with document_id filter",
                        "multi_hop_reasoning": "multi_hop_search (requires KG)",
                        "comparison": "multi_doc_synthesizer",
                        "temporal_info": "filtered_search with filter_type='temporal' or timeline_view",
                    },
                },
            },
            metadata={
                "total_tools": len(tools_list),
                "tier1_count": len([t for t in all_tools if t.tier == 1]),
                "tier2_count": len([t for t in all_tools if t.tier == 2]),
                "tier3_count": len([t for t in all_tools if t.tier == 3]),
            },
        )


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================
#
# These tools combine multiple legacy tools for better UX and reduced tool count:
#
# get_document_info:
#   - Replaces: get_document_summary, get_document_metadata, get_document_sections, get_section_details
#   - Benefit: Single tool with info_type parameter instead of 4 separate tools
#
# exact_match_search:
#   - Replaces: keyword_search, cross_reference_search, entity_search
#   - Benefit: Unified interface with search_type parameter + ROI filtering


class GetDocumentInfoInput(ToolInput):
    """Input for unified get_document_info tool."""

    document_id: str = Field(..., description="Document ID")
    info_type: str = Field(
        ...,
        description="Type of information: 'summary' (high-level overview), 'metadata' (comprehensive stats), 'sections' (list all sections), 'section_details' (specific section info)",
    )
    section_id: Optional[str] = Field(
        None, description="Section ID (required for info_type='section_details')"
    )


@register_tool
class GetDocumentInfoTool(BaseTool):
    """Get document information."""

    name = "get_document_info"
    description = "Get document info/metadata"
    detailed_help = """
    Unified tool for retrieving document information with multiple info types:
    - 'summary': High-level document overview
    - 'metadata': Comprehensive stats (sections, chunks, source info)
    - 'sections': List all sections with titles and hierarchy
    - 'section_details': Detailed info about a specific section

    **When to use:**
    - Need document overview before detailed search
    - Want to understand document structure
    - Looking for specific section to search within

    **Best practices:**
    - Use 'summary' for quick overview
    - Use 'sections' to understand structure
    - Use 'metadata' for comprehensive stats
    - Combine with filtered_search to search within sections

    **Data source:** Vector store metadata (Layer 1/2/3)
    **Speed:** <50ms
    """
    tier = 1
    input_schema = GetDocumentInfoInput

    def execute_impl(
        self, document_id: str, info_type: str, section_id: Optional[str] = None
    ) -> ToolResult:
        try:
            # Get layer metadata
            layer1_chunks = []
            layer2_chunks = []
            layer3_chunks = []

            if hasattr(self.vector_store, "metadata_layer1"):
                layer1_chunks = self.vector_store.metadata_layer1
                layer2_chunks = self.vector_store.metadata_layer2
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer1_chunks = self.vector_store.faiss_store.metadata_layer1
                layer2_chunks = self.vector_store.faiss_store.metadata_layer2
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            if info_type == "summary":
                # Get document summary (Layer 1)
                doc_summary = None
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        doc_summary = meta.get("content")
                        break

                if not doc_summary:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data={"document_id": document_id, "summary": doc_summary},
                    metadata={"document_id": document_id, "summary_length": len(doc_summary)},
                )

            elif info_type == "metadata":
                # Get comprehensive metadata (all layers)
                metadata = {"document_id": document_id}

                # Layer 1: Summary
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        metadata["summary"] = meta.get("content")
                        break

                # Layer 2: Sections
                sections = [
                    meta.get("section_title")
                    for meta in layer2_chunks
                    if meta.get("document_id") == document_id
                ]
                metadata["section_count"] = len(sections)
                metadata["sections"] = sections

                # Layer 3: Chunks
                chunk_count = sum(
                    1 for meta in layer3_chunks if meta.get("document_id") == document_id
                )
                metadata["chunk_count"] = chunk_count

                # Estimate document length
                total_chars = sum(
                    len(meta.get("content", ""))
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                )
                metadata["estimated_chars"] = total_chars
                metadata["estimated_words"] = total_chars // 5

                if not metadata.get("summary") and metadata["section_count"] == 0:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data=metadata,
                    metadata={"document_id": document_id, "total_sections": len(sections)},
                )

            elif info_type == "sections":
                # Get list of sections (Layer 2)
                sections = []
                for meta in layer2_chunks:
                    if meta.get("document_id") == document_id:
                        section_info = {
                            "section_id": meta.get("section_id"),
                            "section_title": meta.get("section_title"),
                        }
                        sections.append(section_info)

                if not sections:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                # Sort sections by section_id
                sections.sort(key=lambda x: x.get("section_id", ""))

                # Use adaptive formatter for dynamic section limits
                try:
                    from .token_manager import get_adaptive_formatter

                    formatter = get_adaptive_formatter()
                    formatted_sections, format_metadata = formatter.format_sections_with_budget(
                        sections, include_summary=False
                    )

                    return ToolResult(
                        success=True,
                        data={
                            "document_id": document_id,
                            "sections": formatted_sections,
                            "count": format_metadata["returned_sections"],
                            "total_sections": format_metadata["total_sections"],
                            "truncated": format_metadata["truncated"],
                            "max_sections_allowed": format_metadata["max_sections_allowed"],
                        },
                        metadata={
                            "document_id": document_id,
                            "section_count": format_metadata["returned_sections"],
                            "total_sections": format_metadata["total_sections"],
                            "truncated": format_metadata["truncated"],
                        },
                    )

                except ImportError:
                    # Fallback
                    total_count = len(sections)
                    max_sections = 50
                    truncated = total_count > max_sections
                    sections = sections[:max_sections]

                    return ToolResult(
                        success=True,
                        data={
                            "document_id": document_id,
                            "sections": sections,
                            "count": len(sections),
                            "total_sections": total_count,
                            "truncated": truncated,
                        },
                        metadata={
                            "document_id": document_id,
                            "section_count": len(sections),
                            "total_sections": total_count,
                            "truncated": truncated,
                        },
                    )

            elif info_type == "section_details":
                # Get specific section details (Layer 2 + Layer 3)
                if not section_id:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="section_id is required for info_type='section_details'",
                    )

                # Find section in Layer 2
                section_data = None
                for meta in layer2_chunks:
                    if (
                        meta.get("document_id") == document_id
                        and meta.get("section_id") == section_id
                    ):
                        section_data = {
                            "document_id": document_id,
                            "section_id": section_id,
                            "section_title": meta.get("section_title"),
                            "section_path": meta.get("section_path"),
                            "summary": meta.get("content"),
                            "page_number": meta.get("page_number"),
                        }
                        break

                if not section_data:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={
                            "document_id": document_id,
                            "section_id": section_id,
                            "found": False,
                        },
                    )

                # Get chunk count (Layer 3)
                chunk_count = sum(
                    1
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                    and meta.get("section_id") == section_id
                )
                section_data["chunk_count"] = chunk_count

                return ToolResult(
                    success=True,
                    data=section_data,
                    metadata={
                        "document_id": document_id,
                        "section_id": section_id,
                        "chunk_count": chunk_count,
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid info_type: {info_type}. Must be 'summary', 'metadata', 'sections', or 'section_details'",
                )

        except Exception as e:
            logger.error(f"Get document info failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


# exact_match_search tool removed - replaced by filtered_search with backward compatibility
# Use filtered_search(search_method='bm25_only') instead
