"""
Filtered Search Tool

Auto-extracted and cleaned from tier2_advanced.py
"""

import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


@register_tool
class FilteredSearchTool(BaseTool):
    """Unified search with filters and search method control."""

    name = "filtered_search"
    description = "Unified search with filters (hybrid/BM25/dense)"
    detailed_help = """
    Unified search combining keyword (BM25) and semantic (dense) retrieval with filtering.
    Consolidates exact_match_search functionality via search_method parameter.

    **Search Methods:**
    - 'hybrid' (default): BM25 + Dense + RRF fusion (, best quality)
    - 'bm25_only': Keyword search only (, fastest, good for exact matches)
    - 'dense_only': Semantic search only (, good for concepts)

    **Filter Types:**
    - 'document': Search within specific document (index-level filtering, fastest)
    - 'section': Search within specific section (post-filter)
    - 'metadata': Filter by document_type/section_type (post-filter)
    - 'temporal': Search within date range (post-filter)
    - None: Search entire database without filtering

    **When to use each method:**
    - Use 'bm25_only' for: Keywords, exact phrases, references (e.g., "article 5")
    - Use 'hybrid' for: General queries, mixed keyword+semantic (best quality)
    - Use 'dense_only' for: Semantic similarity, paraphrased queries, concepts

    **Performance Guide:**
    - Fastest (): search_method='bm25_only' + filter_type='document'
    - Fast (): search_method='bm25_only' without filter
    - Balanced (): search_method='hybrid' (default)

    **Backward compatibility:**
    - Old exact_match_search(search_type='keywords') → filtered_search(search_method='bm25_only')
    - Old exact_match_search(document_id='X') → filtered_search(filter_type='document', filter_value='X')

    **Method:** BM25 + Dense + RRF (configurable via search_method)
    
    """

    input_schema = FilteredSearchInput

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyde_generator = None  # Lazy initialization

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
        search_method: str = "hybrid",
        filter_type: Optional[str] = None,
        filter_value: Optional[str] = None,
        document_type: Optional[str] = None,
        section_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = 6,
        # Legacy parameters
        search_type: Optional[str] = None,
        document_id: Optional[str] = None,
        section_id: Optional[str] = None,
        use_hyde: bool = False,
    ) -> ToolResult:
        from ._utils import validate_k_parameter

        # Handle HyDE (Hypothetical Document Embeddings) - Multi-hypothesis averaging
        hyde_docs = []
        if use_hyde:
            hyde_gen = self._get_hyde_generator()
            if hyde_gen:
                hyde_result = hyde_gen.generate(query, num_docs=1)  # Tier 2: 1 doc (efficiency)
                hyde_docs = hyde_result.hypothetical_docs

                # Warn if fallback was used
                if hyde_result.generation_method == "fallback":
                    logger.warning(
                        "HyDE generation failed internally (LLM error). Falling back to standard search."
                    )
                elif hyde_docs:
                    logger.info(f"HyDE generated {len(hyde_docs)} hypothetical document(s)")
            else:
                logger.warning("HyDE requested but HyDEGenerator unavailable. Falling back to standard search.")

        # Handle legacy parameter mapping
        if search_type and search_method == "hybrid":
            search_method = "bm25_only"
            logger.debug(f"Legacy search_type='{search_type}' mapped to search_method='bm25_only'")

        if document_id and not filter_type:
            filter_type = "document"
            filter_value = document_id
            logger.debug(f"Legacy document_id mapped to filter_type='document'")

        if section_id and not filter_type:
            filter_type = "section"
            filter_value = section_id
            logger.debug(f"Legacy section_id mapped to filter_type='section'")

        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Validate search_method
        if search_method not in {"hybrid", "bm25_only", "dense_only"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid search_method: {search_method}. Must be 'hybrid', 'bm25_only', or 'dense_only'"
            )

        # Validate filter_type if provided
        if filter_type and filter_type not in {"document", "section", "metadata", "temporal"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid filter_type: {filter_type}. Must be 'document', 'section', 'metadata', or 'temporal'"
            )

        # Validate filter_value is provided when filter_type is set (except for temporal)
        if filter_type and filter_type != "temporal" and not filter_value:
            return ToolResult(
                success=False,
                data=None,
                error=f"filter_value is required when filter_type='{filter_type}' is set"
            )

        try:
            # Branch by search_method
            if search_method == "bm25_only":
                chunks = self._execute_bm25_only(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k
                )
            elif search_method == "dense_only":
                chunks = self._execute_dense_only(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k, hyde_docs=hyde_docs
                )
            else:  # hybrid
                chunks = self._execute_hybrid(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k, hyde_docs=hyde_docs
                )

            if not chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "query": query,
                        "filter_type": filter_type,
                        "filter_value": filter_value,
                        "no_results": True,
                        "hyde_used": bool(hyde_docs),
                    },
                )

            formatted = [format_chunk_result(c) for c in chunks]
            citations = [
                f"[{i+1}] {c['document_id']}: {c['section_title']}" for i, c in enumerate(formatted)
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "query": query,
                    "search_method": search_method,
                    "filter_type": filter_type,
                    "filter_value": filter_value,
                    "results_count": len(formatted),
                    "hyde_used": bool(hyde_docs),
                },
            )

        except Exception as e:
            logger.error(f"Filtered search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))



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
    ) -> List[Dict]:
        """BM25-only search () with optional filtering."""
        retrieval_k = k * 3 if filter_type in {"section", "metadata", "temporal"} else k

        # Document filter: index-level (fast)
        if filter_type == "document":
            if hasattr(self.vector_store, "bm25_store"):
                results = self.vector_store.bm25_store.search_layer3(
                    query=query, k=retrieval_k, document_filter=filter_value
                )
            else:
                # Fallback to hierarchical with dummy embedding
                import numpy as np
                dummy_embedding = np.zeros((1, self.embedder.dimensions))
                results_dict = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=dummy_embedding,
                    k_layer3=retrieval_k,
                    document_filter=filter_value,
                )
                results = results_dict.get("layer3", [])
            return results[:k]

        # No filter or post-filter cases
        if hasattr(self.vector_store, "bm25_store"):
            results = self.vector_store.bm25_store.search_layer3(
                query=query, k=retrieval_k, document_filter=None
            )
        else:
            import numpy as np
            dummy_embedding = np.zeros((1, self.embedder.dimensions))
            results_dict = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=dummy_embedding,
                k_layer3=retrieval_k,
                document_filter=None,
            )
            results = results_dict.get("layer3", [])

        # Apply post-filters
        if filter_type == "section":
            section_lower = filter_value.lower()
            results = [
                c for c in results
                if section_lower in c.get("section_title", "").lower()
            ][:k]
        elif filter_type == "metadata":
            if document_type:
                results = [c for c in results if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                results = [c for c in results if c.get("section_type", "").lower() == section_type.lower()]
            results = results[:k]
        elif filter_type == "temporal":
            results = self._apply_temporal_filter(results, start_date, end_date, k)

        return results

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
    ) -> List[Dict]:
        """Dense-only search () with optional filtering."""
        retrieval_k = k * 3 if filter_type else k

        # Use HyDE document for embedding if available (Tier 2: use first doc)
        if hyde_docs and len(hyde_docs) > 0:
            query_embedding = self.embedder.embed_texts([hyde_docs[0]])
        else:
            query_embedding = self.embedder.embed_texts([query])

        # Dense search (no document filter support in FAISS for dense-only)
        dense_results = self.vector_store.faiss_store.search_layer3(
            query_embedding=query_embedding, k=retrieval_k, document_filter=None
        )
        results = dense_results

        # Apply all filters as post-filters
        if filter_type == "document":
            results = [c for c in results if c.get("document_id") == filter_value][:k]
        elif filter_type == "section":
            section_lower = filter_value.lower()
            results = [c for c in results if section_lower in c.get("section_title", "").lower()][:k]
        elif filter_type == "metadata":
            if document_type:
                results = [c for c in results if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                results = [c for c in results if c.get("section_type", "").lower() == section_type.lower()]
            results = results[:k]
        elif filter_type == "temporal":
            results = self._apply_temporal_filter(results, start_date, end_date, k)

        return results

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
    ) -> List[Dict]:
        """Hybrid search (): BM25 + Dense + RRF fusion."""
        # Use HyDE document for embedding (dense), but original query for BM25 (sparse)
        # Tier 2: use first HyDE doc (efficiency)
        if hyde_docs and len(hyde_docs) > 0:
            query_embedding = self.embedder.embed_texts([hyde_docs[0]])
        else:
            query_embedding = self.embedder.embed_texts([query])
        retrieval_k = k * 3 if filter_type in {"section", "metadata", "temporal"} else k

        # Document filter: index-level (fast path for hybrid)
        if filter_type == "document":
            dense_results = self.vector_store.faiss_store.search_layer3(
                query_embedding=query_embedding, k=retrieval_k, document_filter=filter_value
            )
            sparse_results = self.vector_store.bm25_store.search_layer3(
                query=query, k=retrieval_k, document_filter=filter_value
            )
            chunks = self.vector_store._rrf_fusion(dense_results, sparse_results, k=k)
            return chunks

        # No filter or post-filter cases
        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=retrieval_k,
        )
        chunks = results.get("layer3", [])

        # Apply post-filters
        if filter_type == "section":
            section_lower = filter_value.lower()
            chunks = [c for c in chunks if section_lower in c.get("section_title", "").lower()]
        elif filter_type == "metadata":
            if document_type:
                chunks = [c for c in chunks if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                chunks = [c for c in chunks if c.get("section_type", "").lower() == section_type.lower()]
        elif filter_type == "temporal":
            chunks = self._apply_temporal_filter(chunks, start_date, end_date, k)

        # Rerank if available
        if self.reranker and len(chunks) > k:
            chunks = self.reranker.rerank(query, chunks, top_k=k)
        else:
            chunks = chunks[:k]

        return chunks

    def _apply_temporal_filter(
        self,
        chunks: List[Dict],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[Dict]:
        """Apply temporal filter to chunks."""
        from datetime import datetime

        if not start_date and not end_date:
            return chunks[:k]

        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        filtered_chunks = []
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
                filtered_chunks.append(chunk)
            except (ValueError, TypeError):
                continue

        return filtered_chunks[:k]


class SimilaritySearchInput(ToolInput):
    """Input for unified similarity_search tool."""

    chunk_id: str = Field(..., description="Chunk ID to find similar content for")
    search_mode: str = Field(
        ..., description="Search mode: 'related' (semantically related), 'similar' (more like this)"
    )
    cross_document: bool = Field(
        True, description="Search across all documents or within same document"
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)


