"""
Search Tool - HyDE + Expansion Fusion

Unified search using:
- HyDE (Hypothetical Document Embeddings)
- Query Expansion (2 paraphrases)
- Weighted Fusion (w_hyde=0.6, w_exp=0.4)

All API calls via DeepInfra (Qwen models).
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import create_fusion_retriever, format_chunk_result, generate_citation

logger = logging.getLogger(__name__)


class SearchInput(ToolInput):
    """Input for fusion search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(
        10,
        description="Number of results to return (default: 10, optimized for token efficiency)",
        ge=1,
        le=100,
    )
    filter_document: Optional[str] = Field(
        None,
        description="Optional document ID to filter results (searches within specific document)",
    )


@register_tool
class SearchTool(BaseTool):
    """
    HyDE + Expansion Fusion Search.

    Uses:
    - HyDE: Generate hypothetical document answering the query
    - Expansions: 2 query paraphrases for vocabulary coverage
    - Fusion: Weighted combination (0.6 * hyde + 0.4 * expansions)

    All via DeepInfra API (Qwen3-Embedding-8B + Qwen2.5-7B-Instruct).
    """

    name = "search"
    description = "Search using HyDE + Expansion fusion (DeepInfra/Qwen)"
    detailed_help = """
    HyDE + Expansion Fusion Search Tool

    **Algorithm:**
    1. Generate HyDE document (hypothetical answer) via LLM
    2. Generate 2 query expansions (paraphrases) via LLM
    3. Embed all 3 variants (hyde, exp0, exp1)
    4. Search PostgreSQL with each embedding
    5. Min-max normalize scores
    6. Weighted fusion: final = 0.6 * hyde + 0.4 * avg(expansions)
    7. Return top-k results

    **Models:**
    - Embedding: Qwen3-Embedding-8B (4096 dims)
    - LLM: Qwen2.5-7B-Instruct

    **Usage:**
    - search(query="What is safety margin?", k=10)
    - search(query="...", filter_document="doc_id")

    **Research basis:**
    - HyDE: Gao et al. (2022) - +15-30% recall for zero-shot
    - Fusion weights empirically optimized
    """

    input_schema = SearchInput
    requires_reranker = False  # No reranking in fusion pipeline

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_retriever = None  # Lazy initialization

    def _get_fusion_retriever(self):
        """Lazy initialization of FusionRetriever using SSOT factory."""
        if self._fusion_retriever is None:
            # Use shared factory from _utils.py (SSOT for FusionRetriever creation)
            self._fusion_retriever = create_fusion_retriever(
                vector_store=self.vector_store,
                config=self.config,
                layer=3,  # Layer 3 = chunk-level search
            )
        return self._fusion_retriever

    def execute_impl(
        self,
        query: str,
        k: int = 10,
        filter_document: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute HyDE + Expansion fusion search.

        Args:
            query: Natural language query
            k: Number of results
            filter_document: Optional document ID filter

        Returns:
            ToolResult with formatted chunks and citations
        """
        logger.info(f"Fusion search: '{query[:50]}...' (k={k})")

        try:
            # Get fusion retriever
            retriever = self._get_fusion_retriever()

            # Execute fusion search
            chunks = retriever.search(
                query=query,
                k=k,
                document_filter=filter_document,
            )

            # Format results
            formatted = [format_chunk_result(c) for c in chunks]
            final_count = len(formatted)

            # Generate citations
            citations = [
                generate_citation(c, i + 1, format="inline")
                for i, c in enumerate(formatted)
            ]

            # Build metadata
            result_metadata = {
                "query": query,
                "k": k,
                "filter_document": filter_document,
                "search_method": "hyde_expansion_fusion",
                "final_count": final_count,
                "fusion_weights": {
                    "hyde": retriever.config.hyde_weight,
                    "expansion": retriever.config.expansion_weight,
                },
            }

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata=result_metadata,
            )

        except ValueError as e:
            # Configuration error (missing API key, etc.)
            logger.error(f"Search configuration error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Configuration error: {e}. Check DEEPINFRA_API_KEY in .env",
            )

        except ConnectionError as e:
            # Database connection error
            logger.error(f"Database connection error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Database connection error: {e}",
            )

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {type(e).__name__}: {e}",
            )
