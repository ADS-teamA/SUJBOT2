"""
Search Tool

Unified hybrid search with optional query expansion, HyDE, and graph boosting.
"""

import logging
import time
from typing import List, Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


# === Tool 1: Unified Search (with Query Expansion) ===


class SearchInput(ToolInput):
    """Input for unified search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(
        10,
        description="Number of results to return per search (default: 10). Provides comprehensive coverage for most queries. Use lower k for very targeted searches, higher k (up to 200) for benchmarks/evaluation.",
        ge=1,
        le=200,  # Increased for benchmark/evaluation scenarios (internally fetches k*5*2 candidates with reranking+filtering)
    )
    num_expands: int = Field(
        0,
        description="Number of query paraphrases to generate: 0=original query only (fast, ~200ms), 1=original+1 paraphrase (2 queries total, ~500ms), 2=original+2 paraphrases (3 queries total, ~800ms). Total queries = num_expands + 1. Warning: num_expands > 3 may impact performance",
        ge=0,
        le=5,
    )
    enable_graph_boost: bool = Field(
        True,
        description="Enable knowledge graph boosting for entity-centric queries. Boosts chunks mentioning query entities (+30%) and high-centrality concepts (+15%). Use when query mentions specific entities (organizations, standards, regulations). Performance overhead: +200-500ms. Default: True (recall-first).",
    )
    use_hyde: bool = Field(
        False,
        description="Enable HyDE (Hypothetical Document Embeddings) for better zero-shot retrieval. Generates a hypothetical answer to improve semantic matching. Slower (~1-2s) but higher quality for ambiguous queries.",
    )
    search_method: str = Field(
        "hybrid",
        description="Search method: 'hybrid' (BM25+Dense+RRF, default, best quality), 'bm25_only' (keyword only, fast), 'dense_only' (semantic only).",
    )
    filter_type: Optional[str] = Field(
        None,
        description="Type of filter to apply: 'document', 'section', 'metadata', 'temporal'. If None, searches entire database",
    )
    filter_value: Optional[str] = Field(
        None,
        description="Filter value (document_id, section_title). Required if filter_type is set (except 'temporal', which uses start_date/end_date instead)",
    )
    document_type: Optional[str] = Field(None, description="For metadata filter: document type")
    section_type: Optional[str] = Field(None, description="For metadata filter: section type")
    start_date: Optional[str] = Field(
        None, description="For temporal filter: start date (ISO: YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="For temporal filter: end date (ISO: YYYY-MM-DD)"
    )


@register_tool
class SearchTool(BaseTool):
    """Unified search with query expansion, filtering, and graph boosting."""

    name = "search"
    description = "Unified hybrid search with optional query expansion, filtering, and graph boosting"
    detailed_help = """
    Unified search tool combining hybrid retrieval (BM25 + Dense + RRF) with optional
    query expansion, filtering, and knowledge graph boosting.

    **Search Methods:**
    - 'hybrid' (default): BM25 + Dense + RRF fusion (best quality)
    - 'bm25_only': Keyword search only (fastest, good for exact matches)
    - 'dense_only': Semantic search only (good for concepts)

    **Filter Types:**
    - 'document': Search within specific document (index-level filtering, fastest)
    - 'section': Search within specific section (post-filter)
    - 'metadata': Filter by document_type/section_type (post-filter)
    - 'temporal': Search within date range (post-filter)

    **Search Strategy:**
    - Default k=10 provides comprehensive coverage for most queries
    - Use lower k (e.g., k=3-5) for very focused searches
    - Call search ITERATIVELY with different queries for complex information needs
    - Example: 2 searches with k=10 (20 results total) covers multiple aspects

    **Query Expansion:**
    - num_expands=0: Use original query only
    - num_expands=1: Original + 1 paraphrase
    - num_expands=2: Original + 2 paraphrases

    **Graph Boosting:**
    - enable_graph_boost=True: Boosts chunks mentioning query entities (+30% score boost)
    - Use for entity-centric queries (organizations, standards, regulations)
    - Performance overhead: +200-500ms

    **HyDE (Hypothetical Document Embeddings):**
    - use_hyde=True: Generates hypothetical answer for better semantic matching
    - Slower (~1-2s) but higher quality for ambiguous queries
    - Uses single hypothesis for efficiency (balances quality vs. cost)

    **Migration from Removed Tools:**
    - similarity_search functionality → Use expand_context(chunk_ids=[...], expansion_mode='similarity')
    - filtered_search → Use search with filter_type and filter_value parameters

    **Best Practices:**
    - IMPORTANT: Use targeted within document or section search when you are sure about target document or section!
    - Default (fast): search(query, k=3)
    - Entity query: search(query, k=3, enable_graph_boost=True)
    - Ambiguous query: search(query, k=3, num_expands=1-2)
    - Within document: search(query, k=3, filter_type='document', filter_value='doc_id')
    - Keyword match: search(query, k=3, search_method='bm25_only')
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
        self,
        query: str,
        k: int = 3,
        num_expands: int = 0,
        enable_graph_boost: bool = False,
        use_hyde: bool = False,
        search_method: str = "hybrid",
        filter_type: Optional[str] = None,
        filter_value: Optional[str] = None,
        document_type: Optional[str] = None,
        section_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ToolResult:
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Validate search_method
        if search_method not in {"hybrid", "bm25_only", "dense_only"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid search_method: {search_method}. Must be 'hybrid', 'bm25_only', or 'dense_only'",
            )

        # Validate filter_type if provided
        if filter_type and filter_type not in {"document", "section", "metadata", "temporal"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid filter_type: {filter_type}. Must be 'document', 'section', 'metadata', or 'temporal'",
            )

        # Validate filter_value is provided when filter_type is set (except for temporal)
        if filter_type and filter_type != "temporal" and not filter_value:
            return ToolResult(
                success=False,
                data=None,
                error=f"filter_value is required when filter_type='{filter_type}' is set",
            )

        # === STEP 0: HyDE Generation (Multi-Hypothesis) ===
        hyde_docs = []
        hyde_metadata = {"use_hyde": use_hyde, "num_hypotheses": 0, "generation_method": "none"}

        if use_hyde:
            hyde_gen = self._get_hyde_generator()

            if hyde_gen:
                try:
                    # Generate 1 hypothetical doc for original query (efficiency-focused)
                    hyde_result = hyde_gen.generate(query, num_docs=1)
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

        if num_expands > 0 and search_method != "bm25_only":  # Skip expansion for pure keyword search
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

        # === STEP 2: Retrieval for Each Query ===
        # Optimization: Retrieve more candidates (10x) for reranking to improve recall
        candidates_k = k * 10 if self.reranker else k
        if filter_type in {"section", "metadata", "temporal"}:
            # Fetch even more if we have post-filters
            candidates_k = candidates_k * 2

        all_chunks = []
        search_metadata = []

        for idx, q in enumerate(queries, 1):
            # Branch by search_method
            try:
                if search_method == "bm25_only":
                    chunks = self._execute_bm25_only(
                        q,
                        filter_type,
                        filter_value,
                        document_type,
                        section_type,
                        start_date,
                        end_date,
                        candidates_k,
                    )
                elif search_method == "dense_only":
                    chunks = self._execute_dense_only(
                        q,
                        filter_type,
                        filter_value,
                        document_type,
                        section_type,
                        start_date,
                        end_date,
                        candidates_k,
                        hyde_docs=hyde_docs if q == query else None,
                    )
                else:  # hybrid
                    chunks = self._execute_hybrid(
                        q,
                        filter_type,
                        filter_value,
                        document_type,
                        section_type,
                        start_date,
                        end_date,
                        candidates_k,
                        hyde_docs=hyde_docs if q == query else None,
                        enable_graph_boost=enable_graph_boost,
                    )

                # Tag chunks with source query
                for chunk in chunks:
                    if "_source_query" not in chunk:
                        chunk["_source_query"] = q

                all_chunks.extend(chunks)
                search_metadata.append({"query": q, "chunks_retrieved": len(chunks)})

            except Exception as e:
                logger.error(f"Search failed for query '{q}': {e}")
        # === STEP 3: Deduplication ===
        seen_ids = set()
        deduped_chunks = []
        for chunk in all_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduped_chunks.append(chunk)
            elif not chunk_id:
                deduped_chunks.append(chunk)
        all_chunks = deduped_chunks

        # === STEP 4: RRF Fusion (if multiple queries) ===
        if len(queries) > 1 and search_method != "bm25_only":
            chunks = self._rrf_fusion(all_chunks, k=candidates_k)
            fusion_method = "rrf"
        else:
            chunks = all_chunks
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

        # Metadata
        result_metadata = {
            "query": query,
            "k": k,
            "search_method": search_method,
            "filter_type": filter_type,
            "filter_value": filter_value,
            "num_expands": num_expands,
            "enable_graph_boost": enable_graph_boost,
            "fusion_method": fusion_method,
            "reranking_applied": reranking_applied,
            "candidates_retrieved": len(all_chunks),
            "final_count": len(formatted),
            "search_metadata": search_metadata,
            "hyde_metadata": hyde_metadata,
            "expansion_metadata": expansion_metadata,
        }

        # Add RAG confidence to metadata
        if confidence:
            result_metadata["rag_confidence"] = confidence.to_dict()


        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata=result_metadata,
        )

    def _get_query_embedding(self, query: str, hyde_docs: Optional[List[str]] = None):
        """Get query embedding (HyDE or regular)."""
        if hyde_docs and len(hyde_docs) > 0:
            return self.embedder.embed_texts([hyde_docs[0]])
        return self.embedder.embed_texts([query])

    def _execute_bm25_retrieval(
        self, query: str, k: int, document_filter: Optional[str] = None
    ) -> List[dict]:
        """Execute BM25 retrieval with fallback logic."""
        if hasattr(self.vector_store, "bm25_store") and self.vector_store.bm25_store:
            return self.vector_store.bm25_store.search_layer3(
                query=query, k=k, document_filter=document_filter
            )
        else:
            # Fallback to hierarchical with dummy embedding
            import numpy as np

            dummy_embedding = np.zeros((1, self.embedder.dimensions))
            results_dict = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=dummy_embedding,
                k_layer3=k,
                document_filter=document_filter,
            )
            return results_dict.get("layer3", [])

    def _execute_bm25_only(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[dict]:
        """BM25-only search with optional filtering."""
        # Document filter: index-level (fast), otherwise None
        document_filter = filter_value if filter_type == "document" else None
        results = self._execute_bm25_retrieval(query, k, document_filter)

        return self._apply_post_filters(
            results, filter_type, filter_value, document_type, section_type, start_date, end_date, k
        )

    def _execute_dense_only(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
        hyde_docs: Optional[List[str]] = None,
    ) -> List[dict]:
        """Dense-only search with optional filtering."""
        query_embedding = self._get_query_embedding(query, hyde_docs)

        # Dense search - use adapter interface (works with both FAISS and PostgreSQL)
        document_filter = filter_value if filter_type == "document" else None
        dense_results = self.vector_store.search_layer3(
            query_embedding=query_embedding, k=k, document_filter=document_filter
        )

        return self._apply_post_filters(
            dense_results,
            filter_type,
            filter_value,
            document_type,
            section_type,
            start_date,
            end_date,
            k,
        )

    def _execute_hybrid(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
        hyde_docs: Optional[List[str]] = None,
        enable_graph_boost: bool = False,
    ) -> List[dict]:
        """Hybrid search: BM25 + Dense + RRF fusion + Graph Boost."""

        # 1. Graph Boost (if enabled)
        if enable_graph_boost and self.graph_retriever:
            query_embedding = self.embedder.embed_texts([query])[0]
            try:
                results = self.graph_retriever.search(
                    query=query,
                    query_embedding=query_embedding,
                    k=k,
                    enable_graph_boost=True,
                )
                chunks = results.get("layer3", [])
                if chunks:
                    return self._apply_post_filters(
                        chunks,
                        filter_type,
                        filter_value,
                        document_type,
                        section_type,
                        start_date,
                        end_date,
                        k,
                    )
            except Exception as e:
                logger.warning(f"Graph boost failed: {e}. Falling back to standard hybrid.")

        # 2. Standard Hybrid
        query_embedding = self._get_query_embedding(query, hyde_docs)

        # Document filter: use hierarchical search (works with both backends)
        document_filter = filter_value if filter_type == "document" else None
        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=k,
            document_filter=document_filter,
        )
        chunks = results.get("layer3", [])

        return self._apply_post_filters(
            chunks, filter_type, filter_value, document_type, section_type, start_date, end_date, k
        )

    def _apply_metadata_filter(
        self, chunks: List[dict], document_type: Optional[str], section_type: Optional[str]
    ) -> List[dict]:
        """Apply metadata filters (document_type, section_type)."""
        filtered = chunks
        if document_type:
            filtered = [c for c in filtered if c.get("doc_type", "").lower() == document_type.lower()]
        if section_type:
            filtered = [
                c for c in filtered if c.get("section_type", "").lower() == section_type.lower()
            ]
        return filtered

    def _apply_temporal_filter(
        self, chunks: List[dict], start_date: Optional[str], end_date: Optional[str]
    ) -> List[dict]:
        """Apply temporal filter (date range)."""
        from datetime import datetime

        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        filtered = []
        for chunk in chunks:
            chunk_date_str = chunk.get("date") or chunk.get("timestamp")
            if not chunk_date_str:
                continue
            try:
                chunk_date = datetime.fromisoformat(chunk_date_str)
                if start_dt and chunk_date < start_dt:
                    continue
                if end_dt and chunk_date > end_dt:
                    continue
                filtered.append(chunk)
            except (ValueError, TypeError):
                # Invalid date format or type - skip this chunk
                continue
        return filtered

    def _apply_post_filters(
        self,
        chunks: List[dict],
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[dict]:
        """Apply post-retrieval filters."""
        if not chunks:
            return []

        filtered = chunks

        if filter_type == "document":
            # If we didn't filter at index level (e.g. dense only), filter here
            filtered = [c for c in filtered if c.get("document_id") == filter_value]
        elif filter_type == "section":
            section_lower = filter_value.lower()
            filtered = [c for c in filtered if section_lower in c.get("section_title", "").lower()]
        elif filter_type == "metadata":
            filtered = self._apply_metadata_filter(filtered, document_type, section_type)
        elif filter_type == "temporal":
            filtered = self._apply_temporal_filter(filtered, start_date, end_date)

        return filtered[:k]

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

        # Sort by RRF score (descending: highest confidence first)
        # Higher RRF score = chunk appears in more queries at better ranks = BETTER result
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Return top chunks
        return [item["chunk"] for item in rrf_scores]


