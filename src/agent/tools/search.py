"""
Search Tool

Unified hybrid search with optional query expansion, HyDE, and graph boosting.
"""

import logging
import time
from typing import List

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


# === Tool 1: Unified Search (with Query Expansion) ===


class SearchInput(ToolInput):
    """Input for unified search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(3, description="Number of results to return per search (default: 3). Use targeted searches with low k and call iteratively rather than high k single searches. Recommended: k=3 for focused results, increase only if needed.", ge=1, le=10)
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

    **IMPORTANT - Targeted Search Strategy:**
    - Use LOW k values (k=3 default) for focused, high-quality results
    - Call search ITERATIVELY multiple times with different queries rather than high k single searches
    - Example: 3 searches with k=3 (9 results total) > 1 search with k=9
    - Benefits: Better diversity, more targeted results, easier to track provenance
    - Only increase k if you need more results from a SPECIFIC query

    **Query Expansion:**
    - num_expands=0: Use original query only 
    - num_expands=1: Original + 1 paraphrase 
    - num_expands=2: Original + 2 paraphrases 
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

    **Best practices:**
    - Start with defaults for most queries (fast path)
    - Enable graph boost for entity-focused questions (organizations, standards, regulations)
    - Use num_expands=1-2 when user query is ambiguous or recall is critical
    - Combine both features for complex entity queries where quality > speed
    - Check metadata.graph_boost_enabled to verify if boost was applied
    """

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
                    num_hypotheses=self.config.hyde_num_hypotheses,
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

                # NEW STRATEGY: Separate search for each HyDE doc instead of averaging
                if hyde_docs and q == query:
                    # Search with each HyDE document separately
                    logger.info(f"Using {len(hyde_docs)} HyDE docs (separate search per doc) for query {idx}")

                    for hyde_idx, hyde_doc in enumerate(hyde_docs, 1):
                        # Embed this HyDE doc
                        hyde_embedding = self.embedder.embed_texts([hyde_doc])[0]

                        # Search with HyDE embedding
                        try:
                            hyde_results = self.graph_retriever.search(
                                query=q,
                                query_embedding=hyde_embedding,
                                k=candidates_k,
                                enable_graph_boost=True,
                            )
                            hyde_chunks = hyde_results.get("layer3", [])

                            if not hyde_chunks:
                                logger.debug(f"HyDE doc {hyde_idx} returned no results, trying standard search")
                                hyde_results = self.vector_store.hierarchical_search(
                                    query_embedding=hyde_embedding,
                                    k_layer3=candidates_k,
                                )
                                hyde_chunks = hyde_results["layer3"]

                            # Tag chunks with HyDE source
                            for chunk in hyde_chunks:
                                chunk["_source_query"] = f"{q} (HyDE {hyde_idx})"
                                chunk["_hyde_doc_index"] = hyde_idx

                            all_chunks.extend(hyde_chunks)
                            logger.info(f"HyDE doc {hyde_idx}/{len(hyde_docs)} retrieved {len(hyde_chunks)} chunks")

                        except Exception as e:
                            logger.warning(f"HyDE doc {hyde_idx} search failed: {e}")

                    # Also search with original query (non-HyDE) for comparison
                    query_embedding = self.embedder.embed_texts([q])[0]
                    try:
                        results = self.graph_retriever.search(
                            query=q,
                            query_embedding=query_embedding,
                            k=candidates_k,
                            enable_graph_boost=True,
                        )
                        chunks = results.get("layer3", [])
                        if not chunks:
                            results = self.vector_store.hierarchical_search(
                                query_embedding=query_embedding,
                                k_layer3=candidates_k,
                            )
                            chunks = results["layer3"]
                        graph_boost_applied = True
                    except Exception as e:
                        logger.warning(f"Original query search failed: {e}, skipping")
                        chunks = []
                        graph_boost_applied = False
                else:
                    # Standard embedding for paraphrases
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

                # NEW STRATEGY: Separate search for each HyDE doc instead of averaging
                if hyde_docs and q == query:
                    # Search with each HyDE document separately
                    logger.info(f"Using {len(hyde_docs)} HyDE docs (separate search per doc) for query {idx}")

                    for hyde_idx, hyde_doc in enumerate(hyde_docs, 1):
                        # Embed this HyDE doc
                        hyde_embedding = self.embedder.embed_texts([hyde_doc])[0]

                        # Search with HyDE embedding
                        hyde_results = self.vector_store.hierarchical_search(
                            query_embedding=hyde_embedding,
                            k_layer3=candidates_k,
                        )
                        hyde_chunks = hyde_results["layer3"]

                        # Tag chunks with HyDE source
                        for chunk in hyde_chunks:
                            chunk["_source_query"] = f"{q} (HyDE {hyde_idx})"
                            chunk["_hyde_doc_index"] = hyde_idx

                        all_chunks.extend(hyde_chunks)
                        logger.info(f"HyDE doc {hyde_idx}/{len(hyde_docs)} retrieved {len(hyde_chunks)} chunks")

                    # Also search with original query (non-HyDE) for comparison
                    query_embedding = self.embedder.embed_texts([q])[0]
                    results = self.vector_store.hierarchical_search(
                        query_embedding=query_embedding,
                        k_layer3=candidates_k,
                    )
                    chunks = results["layer3"]
                else:
                    # Standard embedding for paraphrases
                    query_embedding = self.embedder.embed_texts([q])[0]

                    # Hybrid search
                    results = self.vector_store.hierarchical_search(
                        query_embedding=query_embedding,
                        k_layer3=candidates_k,
                    )
                    chunks = results["layer3"]

                # Tag chunks with their source query for tracking
                for chunk in chunks:
                    if "_source_query" not in chunk:  # Don't overwrite HyDE tags
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

        # Log total candidates before deduplication
        logger.info(
            f"Total candidates from all queries: {len(all_chunks)} chunks "
            f"(from {len(queries)} queries, including HyDE searches)"
        )

        # === STEP 2.5: Deduplicate chunks before fusion ===
        # When using HyDE with multiple docs, same chunk may appear multiple times
        chunks_before_dedup = len(all_chunks)
        if chunks_before_dedup > 0:
            # Deduplicate by chunk_id, keeping first occurrence
            seen_ids = set()
            deduped_chunks = []
            for chunk in all_chunks:
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    deduped_chunks.append(chunk)
                elif not chunk_id:
                    # No ID, keep it (shouldn't happen but be safe)
                    deduped_chunks.append(chunk)

            all_chunks = deduped_chunks
            if chunks_before_dedup > len(all_chunks):
                logger.info(
                    f"Deduplication: {chunks_before_dedup} chunks → {len(all_chunks)} unique chunks "
                    f"(removed {chunks_before_dedup - len(all_chunks)} duplicates)"
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

        # Build retrieval_methods based on what actually happened
        retrieval_methods = {
            "hybrid_search": fusion_method == "rrf" or len(queries) > 1,
            "reranking": reranking_applied,
            "graph_boost": use_graph_boost,
            "bm25_only": False,  # We always use hybrid (BM25 + dense)
            "dense_only": False,  # We always use hybrid (BM25 + dense)
        }

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
            "retrieval_methods": retrieval_methods,
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


