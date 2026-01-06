"""
HyDE + Expansion Fusion Retriever (4-Signal Fusion)

Core retrieval algorithm:
1. Generate HyDE document + 2 query expansions (LLM)
2. Embed all variants (original + hyde + 2 expansions = 4 embeddings)
3. Search vector store with each embedding (PARALLEL with asyncio.gather)
4. Min-max normalize each score set
5. Weighted fusion: final = 0.5 * original + 0.25 * hyde + 0.25 * avg(expansions)
6. Return top-k results

Research basis:
- HyDE: Gao et al. (2022) - +15-30% recall for zero-shot retrieval
- Query Expansion: Standard IR technique for vocabulary mismatch
- 4-Signal Fusion: original query for keyword match + semantic variants
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.cache import TTLCache
from .deepinfra_client import DeepInfraClient
from .hyde_expansion import HyDEExpansionGenerator, HyDEExpansionResult


def _run_async_safe(coro, timeout: float = 30.0, operation_name: str = "async operation"):
    """
    Safely run async coroutine from sync context.

    Handles two scenarios:
    1. No running event loop: Uses asyncio.run() directly
    2. Already in async context: Uses nest_asyncio (applied at startup in backend/main.py)

    Args:
        coro: Async coroutine to execute
        timeout: Timeout in seconds (default: 30)
        operation_name: Name of operation for error messages (default: "async operation")

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If execution exceeds timeout (with actionable message)
        RuntimeError: If execution fails
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop - use asyncio.run() with timeout
        # This is the expected case when called from sync context
        loop = None

    try:
        if loop is None:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        else:
            # Already in async context - nest_asyncio should be applied at startup
            # (backend/main.py applies it early to allow nested loops)
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError as e:
        logger.error(
            f"Timeout ({timeout}s) exceeded during {operation_name}. "
            "Consider increasing timeout or simplifying the query."
        )
        raise TimeoutError(
            f"Operation '{operation_name}' timed out after {timeout} seconds. "
            "The database or search service may be under heavy load. Please try again."
        ) from e

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for fusion retrieval."""

    original_weight: float = 0.5  # Weight for original query (direct match)
    hyde_weight: float = 0.25  # Weight for HyDE scores
    expansion_weight: float = 0.25  # Weight for expansion scores (split between 2)
    default_k: int = 16  # Default number of results (increased for better recall)
    candidates_multiplier: int = 4  # Retrieve k * multiplier candidates per query (increased for 4 signals)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.original_weight <= 1:
            raise ValueError(f"original_weight must be in [0, 1], got {self.original_weight}")
        if not 0 <= self.hyde_weight <= 1:
            raise ValueError(f"hyde_weight must be in [0, 1], got {self.hyde_weight}")
        if not 0 <= self.expansion_weight <= 1:
            raise ValueError(f"expansion_weight must be in [0, 1], got {self.expansion_weight}")
        total = self.original_weight + self.hyde_weight + self.expansion_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"original_weight + hyde_weight + expansion_weight must equal 1.0, "
                f"got {self.original_weight} + {self.hyde_weight} + {self.expansion_weight} = {total}"
            )
        if self.default_k <= 0:
            raise ValueError(f"default_k must be positive, got {self.default_k}")
        if self.candidates_multiplier < 1:
            raise ValueError(f"candidates_multiplier must be >= 1, got {self.candidates_multiplier}")


class FusionRetriever:
    """
    HyDE + Expansion Fusion Retriever.

    Combines HyDE (hypothetical document embeddings) with query expansions
    using weighted score fusion for improved retrieval.

    Algorithm:
    1. Generate HyDE doc + 2 expansions via LLM
    2. Embed original query + hyde_doc + expansion_0 + expansion_1 (4 embeddings)
    3. Search PostgreSQL with each embedding (k * candidates_multiplier)
    4. Collect unique chunks with scores from each method
    5. Min-max normalize each score set to [0, 1]
    6. Fuse: final = 0.5 * original + 0.25 * hyde + 0.25 * avg(expansions)
    7. Sort by fused score, return top-k

    Example:
        >>> retriever = FusionRetriever(client, vector_store)
        >>> results = retriever.search("What is the safety margin?", k=10)
    """

    def __init__(
        self,
        client: DeepInfraClient,
        vector_store: Any,  # PostgresVectorStoreAdapter
        config: Optional[FusionConfig] = None,
    ):
        """
        Initialize fusion retriever.

        Args:
            client: DeepInfraClient for embedding and LLM
            vector_store: Vector store adapter (PostgreSQL)
            config: Fusion configuration
        """
        self.client = client
        self.vector_store = vector_store
        self.config = config or FusionConfig()

        # Initialize HyDE + expansion generator
        self.generator = HyDEExpansionGenerator(client)

        # Search result cache: 5-minute TTL, 500 entries max
        # Reduces latency for repeated/similar queries from ~100ms to ~1ms
        self._result_cache: TTLCache[List[Dict]] = TTLCache(
            ttl_seconds=300,
            max_size=500,
            name="fusion_search_cache"
        )

        logger.info(
            f"FusionRetriever initialized "
            f"(w_orig={self.config.original_weight}, "
            f"w_hyde={self.config.hyde_weight}, "
            f"w_exp={self.config.expansion_weight}, "
            f"k={self.config.default_k}, "
            f"cache_ttl=300s)"
        )

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search using HyDE + Expansion fusion.

        Args:
            query: User query
            k: Number of results to return (default: config.default_k)
            document_filter: Optional document ID to filter by

        Returns:
            List of chunk dicts with keys:
            - chunk_id, document_id, content, score
            - section_id, section_title, hierarchical_path
        """
        k = k or self.config.default_k

        # Check cache first - returns early if hit
        cache_key = hashlib.md5(
            f"{query.strip().lower()}:{k}:{document_filter or ''}"
            .encode()
        ).hexdigest()

        cached_result = self._result_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Fusion search CACHE HIT for '{query[:30]}...' (k={k})")
            return cached_result

        # Optimized: Different candidate counts for hybrid vs pure vector searches
        # Hybrid search needs more candidates for effective RRF fusion
        # Pure vector search can use fewer candidates (still provides good diversity)
        candidates_k_hybrid = k * self.config.candidates_multiplier  # Keep full multiplier for RRF
        candidates_k_pure = max(k * 2, 50)  # Reduced: 50% fewer candidates for pure vector

        logger.info(f"Fusion search: '{query[:50]}...' (k={k}, hybrid_k={candidates_k_hybrid}, pure_k={candidates_k_pure})")

        # Step 1: Generate HyDE + expansions
        try:
            hyde_result = self.generator.generate(query)
            logger.debug(f"Generated HyDE: {hyde_result.hyde_document[:100]}...")
        except Exception as e:
            logger.error(f"HyDE generation failed for query '{query[:50]}...': {e}", exc_info=True)
            raise RuntimeError(f"HyDE generation failed: {e}") from e

        # Step 2: Embed all variants (original + hyde + 2 expansions)
        try:
            texts_to_embed = [
                query,  # Original query for direct match
                hyde_result.hyde_document,
                hyde_result.expansions[0],
                hyde_result.expansions[1],
            ]
            embeddings = self.client.embed_texts(texts_to_embed)
        except Exception as e:
            logger.error(f"Embedding failed for fusion search: {e}", exc_info=True)
            raise RuntimeError(f"Embedding failed: {e}") from e

        orig_emb = embeddings[0]
        hyde_emb = embeddings[1]
        exp_0_emb = embeddings[2]
        exp_1_emb = embeddings[3]

        # Step 3: Search with each embedding (PARALLEL)
        # Original query uses HYBRID search (vector + BM25) for keyword matching
        # HyDE and expansions use pure vector search (semantic)
        try:
            # Check if vector store has async search method
            if hasattr(self.vector_store, 'search_layer3_async'):
                # Use parallel async searches (4x faster)
                orig_results, hyde_results, exp_0_results, exp_1_results = _run_async_safe(
                    self._parallel_search(
                        query, orig_emb, hyde_emb, exp_0_emb, exp_1_emb,
                        candidates_k_hybrid, candidates_k_pure, document_filter
                    ),
                    operation_name="Layer 3 parallel search"
                )
            else:
                # Fallback to sequential (for backwards compatibility)
                # Original query: HYBRID search for keyword matching
                orig_results = self.vector_store.search_layer3(
                    query_embedding=orig_emb,
                    k=candidates_k_hybrid,
                    document_filter=document_filter,
                    query_text=query,  # Enable BM25 hybrid
                )
                # HyDE and expansions: pure vector search (fewer candidates needed)
                hyde_results = self.vector_store.search_layer3(
                    query_embedding=hyde_emb,
                    k=candidates_k_pure,
                    document_filter=document_filter,
                )
                exp_0_results = self.vector_store.search_layer3(
                    query_embedding=exp_0_emb,
                    k=candidates_k_pure,
                    document_filter=document_filter,
                )
                exp_1_results = self.vector_store.search_layer3(
                    query_embedding=exp_1_emb,
                    k=candidates_k_pure,
                    document_filter=document_filter,
                )
        except Exception as e:
            logger.error(f"Vector store search failed: {e}", exc_info=True)
            raise RuntimeError(f"Vector store search failed: {e}") from e

        logger.debug(
            f"Retrieved candidates: orig={len(orig_results)}, hyde={len(hyde_results)}, "
            f"exp0={len(exp_0_results)}, exp1={len(exp_1_results)}"
        )

        # Step 4: Collect unique chunks with scores
        chunk_data = {}  # chunk_id -> {orig: score, hyde: score, exp0: score, exp1: score, data: dict}

        for result in orig_results:
            chunk_id = result["chunk_id"]
            chunk_data[chunk_id] = {
                "orig": result["score"],
                "hyde": None,
                "exp0": None,
                "exp1": None,
                "data": result,
            }

        for result in hyde_results:
            chunk_id = result["chunk_id"]
            if chunk_id in chunk_data:
                chunk_data[chunk_id]["hyde"] = result["score"]
            else:
                chunk_data[chunk_id] = {
                    "orig": None,
                    "hyde": result["score"],
                    "exp0": None,
                    "exp1": None,
                    "data": result,
                }

        for result in exp_0_results:
            chunk_id = result["chunk_id"]
            if chunk_id in chunk_data:
                chunk_data[chunk_id]["exp0"] = result["score"]
            else:
                chunk_data[chunk_id] = {
                    "orig": None,
                    "hyde": None,
                    "exp0": result["score"],
                    "exp1": None,
                    "data": result,
                }

        for result in exp_1_results:
            chunk_id = result["chunk_id"]
            if chunk_id in chunk_data:
                chunk_data[chunk_id]["exp1"] = result["score"]
            else:
                chunk_data[chunk_id] = {
                    "orig": None,
                    "hyde": None,
                    "exp0": None,
                    "exp1": result["score"],
                    "data": result,
                }

        if not chunk_data:
            logger.warning("No results found for fusion search")
            return []

        # Step 5-7: Vectorized normalization, fusion, and top-k selection
        top_k = self._batch_normalize_and_fuse(chunk_data, k)

        logger.info(
            f"Fusion search complete: {len(chunk_data)} candidates → {len(top_k)} results"
        )

        # Cache the result for future identical queries
        self._result_cache.set(cache_key, top_k)

        return top_k

    def _get_min_max(self, scores: List[float]) -> tuple:
        """Get min and max from score list (handles empty)."""
        if not scores:
            return 0.0, 1.0
        return min(scores), max(scores)

    def _normalize_score(
        self,
        score: Optional[float],
        min_val: float,
        max_val: float,
    ) -> float:
        """
        Min-max normalize score to [0, 1].

        Returns 0.0 if score is None (missing).
        Returns 0.5 if min == max (all same).
        """
        if score is None:
            return 0.0

        if max_val == min_val:
            return 0.5

        return (score - min_val) / (max_val - min_val)

    def _batch_normalize_and_fuse(
        self,
        chunk_data: Dict[str, Dict],
        k: int,
    ) -> List[Dict]:
        """
        Vectorized score normalization and fusion using NumPy.

        ~3-5x faster than loop-based implementation for 200+ candidates.
        Uses:
        - Matrix operations for min-max normalization
        - np.dot for weighted fusion
        - np.argpartition for O(n) top-k selection

        Args:
            chunk_data: {chunk_id: {orig, hyde, exp0, exp1, data}}
            k: Number of results to return

        Returns:
            List of top-k results with fused scores
        """
        chunk_ids = list(chunk_data.keys())
        n = len(chunk_ids)

        if n == 0:
            return []

        # Build score matrix (n x 4) with NaN for missing scores
        scores = np.full((n, 4), np.nan, dtype=np.float32)
        data_list = []

        for i, cid in enumerate(chunk_ids):
            d = chunk_data[cid]
            if d["orig"] is not None:
                scores[i, 0] = d["orig"]
            if d["hyde"] is not None:
                scores[i, 1] = d["hyde"]
            if d["exp0"] is not None:
                scores[i, 2] = d["exp0"]
            if d["exp1"] is not None:
                scores[i, 3] = d["exp1"]
            data_list.append((d["data"], d["orig"], d["hyde"], d["exp0"], d["exp1"]))

        # Vectorized min-max normalization per column
        mins = np.nanmin(scores, axis=0)
        maxs = np.nanmax(scores, axis=0)
        ranges = maxs - mins

        # Handle constant columns (all same value or single value)
        # Log warning for debugging but don't fail - this is normal for sparse results
        constant_cols = ranges == 0
        if constant_cols.any():
            col_names = ["orig", "hyde", "exp0", "exp1"]
            constant_names = [col_names[i] for i in range(4) if constant_cols[i]]
            logger.debug(f"Constant score columns detected: {constant_names}")

        ranges = np.where(ranges > 0, ranges, 1.0)
        normalized = (scores - mins) / ranges
        # Replace NaN (missing scores) with 0.0
        normalized = np.nan_to_num(normalized, nan=0.0)

        # Weighted fusion using matrix multiplication
        # Weights: [original=0.5, hyde=0.25, exp0=0.125, exp1=0.125]
        weights = np.array([
            self.config.original_weight,      # 0.5
            self.config.hyde_weight,          # 0.25
            self.config.expansion_weight / 2, # 0.125
            self.config.expansion_weight / 2, # 0.125
        ], dtype=np.float32)

        fused = np.dot(normalized, weights)

        # Efficient top-k selection using argpartition (O(n) vs O(n log n) for full sort)
        if k < n:
            # Get indices of top k values (unordered)
            top_indices = np.argpartition(-fused, k)[:k]
            # Sort only top k indices by fused score
            sorted_top_indices = top_indices[np.argsort(-fused[top_indices])]
        else:
            # Return all, sorted
            sorted_top_indices = np.argsort(-fused)

        # Build result list
        results = []
        for idx in sorted_top_indices:
            data, orig, hyde, exp0, exp1 = data_list[idx]
            result = data.copy()
            result["score"] = float(fused[idx])
            result["orig_score"] = orig
            result["hyde_score"] = hyde
            result["exp0_score"] = exp0
            result["exp1_score"] = exp1
            results.append(result)

        return results

    async def _parallel_search(
        self,
        query_text: str,
        orig_emb: np.ndarray,
        hyde_emb: np.ndarray,
        exp_0_emb: np.ndarray,
        exp_1_emb: np.ndarray,
        k_hybrid: int,
        k_pure: int,
        document_filter: Optional[str],
    ) -> tuple:
        """
        Execute parallel searches with asyncio.gather.

        This is ~4x faster than sequential searches.
        Original query uses HYBRID search (vector + BM25) with more candidates.
        HyDE/expansions use pure vector search with fewer candidates.
        Uses return_exceptions=True to handle partial failures gracefully.
        """
        results = await asyncio.gather(
            # Original query: HYBRID search for keyword matching (more candidates for RRF)
            self.vector_store.search_layer3_async(
                query_embedding=orig_emb,
                k=k_hybrid,
                document_filter=document_filter,
                query_text=query_text,  # Enable BM25 hybrid
            ),
            # HyDE and expansions: pure vector search (fewer candidates needed)
            self.vector_store.search_layer3_async(
                query_embedding=hyde_emb,
                k=k_pure,
                document_filter=document_filter,
            ),
            self.vector_store.search_layer3_async(
                query_embedding=exp_0_emb,
                k=k_pure,
                document_filter=document_filter,
            ),
            self.vector_store.search_layer3_async(
                query_embedding=exp_1_emb,
                k=k_pure,
                document_filter=document_filter,
            ),
            return_exceptions=True,  # Don't fail all if one fails
        )

        # Handle partial failures - log errors and return empty list for failed searches
        search_names = ["original", "hyde", "expansion_0", "expansion_1"]
        processed_results = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Parallel search '{search_names[i]}' failed: {result}",
                    exc_info=result if logger.isEnabledFor(logging.DEBUG) else None
                )
                processed_results.append([])  # Empty list for failed search
                failed_count += 1
            else:
                processed_results.append(result)

        if failed_count == 4:
            # All searches failed - raise the first exception
            first_error = next(r for r in results if isinstance(r, Exception))
            raise RuntimeError(
                f"All parallel searches failed. First error: {first_error}"
            ) from first_error

        if failed_count > 0:
            logger.warning(
                f"{failed_count}/4 parallel searches failed, proceeding with partial results"
            )

        return tuple(processed_results)

    def search_layer2(
        self,
        query: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search Layer 2 (sections) using HyDE + Expansion fusion.

        Same algorithm as search() but operates on section summaries (Layer 2)
        instead of fine-grained chunks (Layer 3).

        Use for:
        - Overview/summary queries ("Co obsahuje kapitola X?")
        - Section discovery ("Které sekce pojednávají o Y?")
        - Document structure questions

        Args:
            query: User query
            k: Number of sections to return (default: 5)
            document_filter: Optional document ID to filter by

        Returns:
            List of section dicts with section_id, section_title, section_path, etc.
        """
        k = k or 5  # Sections are larger, so return fewer by default
        candidates_k = k * self.config.candidates_multiplier

        logger.info(f"Fusion search Layer 2: '{query[:50]}...' (k={k})")

        # Step 1: Generate HyDE + expansions (same as Layer 3)
        try:
            hyde_result = self.generator.generate(query)
            logger.debug(f"Generated HyDE: {hyde_result.hyde_document[:100]}...")
        except Exception as e:
            logger.error(f"HyDE generation failed for L2 query '{query[:50]}...': {e}", exc_info=True)
            from src.exceptions import RetrievalError
            raise RetrievalError(
                f"HyDE generation failed for query: {query[:50]}...",
                details={"query": query[:100]},
                cause=e
            )

        # Step 2: Embed all variants
        try:
            texts_to_embed = [
                hyde_result.hyde_document,
                hyde_result.expansions[0],
                hyde_result.expansions[1],
            ]
            embeddings = self.client.embed_texts(texts_to_embed)
        except Exception as e:
            logger.error(f"Embedding failed for L2 fusion search: {e}", exc_info=True)
            from src.exceptions import EmbeddingError
            raise EmbeddingError(
                f"Embedding failed for L2 fusion search",
                details={"query": query[:100]},
                cause=e
            )

        hyde_emb = embeddings[0]
        exp_0_emb = embeddings[1]
        exp_1_emb = embeddings[2]

        # Step 3: Search Layer 2 with each embedding (PARALLEL if available)
        try:
            if hasattr(self.vector_store, '_async_search_layer2'):
                # Use parallel async searches
                hyde_results, exp_0_results, exp_1_results = _run_async_safe(
                    self._parallel_search_layer2(
                        hyde_emb, exp_0_emb, exp_1_emb,
                        candidates_k, document_filter
                    ),
                    operation_name="Layer 2 parallel search"
                )
            else:
                # Sequential fallback
                hyde_results = self.vector_store.search_layer2(
                    query_embedding=hyde_emb,
                    k=candidates_k,
                    document_filter=document_filter,
                )
                exp_0_results = self.vector_store.search_layer2(
                    query_embedding=exp_0_emb,
                    k=candidates_k,
                    document_filter=document_filter,
                )
                exp_1_results = self.vector_store.search_layer2(
                    query_embedding=exp_1_emb,
                    k=candidates_k,
                    document_filter=document_filter,
                )
        except Exception as e:
            logger.error(f"Layer 2 vector store search failed: {e}", exc_info=True)
            from src.exceptions import SearchError
            raise SearchError(
                f"Layer 2 vector store search failed",
                details={"query": query[:100], "document_filter": document_filter},
                cause=e
            )

        logger.debug(
            f"L2 candidates: hyde={len(hyde_results)}, "
            f"exp0={len(exp_0_results)}, exp1={len(exp_1_results)}"
        )

        # Step 4-7: Same fusion logic as Layer 3 (using section_id as key)
        section_data = {}

        for result in hyde_results:
            section_id = result.get("section_id") or result.get("chunk_id")
            section_data[section_id] = {
                "hyde": result["score"],
                "exp0": None,
                "exp1": None,
                "data": result,
            }

        for result in exp_0_results:
            section_id = result.get("section_id") or result.get("chunk_id")
            if section_id in section_data:
                section_data[section_id]["exp0"] = result["score"]
            else:
                section_data[section_id] = {
                    "hyde": None,
                    "exp0": result["score"],
                    "exp1": None,
                    "data": result,
                }

        for result in exp_1_results:
            section_id = result.get("section_id") or result.get("chunk_id")
            if section_id in section_data:
                section_data[section_id]["exp1"] = result["score"]
            else:
                section_data[section_id] = {
                    "hyde": None,
                    "exp0": None,
                    "exp1": result["score"],
                    "data": result,
                }

        if not section_data:
            logger.warning("No Layer 2 results found for fusion search")
            return []

        # Normalize and fuse scores
        hyde_scores = [d["hyde"] for d in section_data.values() if d["hyde"] is not None]
        exp0_scores = [d["exp0"] for d in section_data.values() if d["exp0"] is not None]
        exp1_scores = [d["exp1"] for d in section_data.values() if d["exp1"] is not None]

        hyde_min, hyde_max = self._get_min_max(hyde_scores)
        exp0_min, exp0_max = self._get_min_max(exp0_scores)
        exp1_min, exp1_max = self._get_min_max(exp1_scores)

        # For Layer 2, we only have hyde + expansions (no original query embedding)
        # Normalize weights to sum to 1.0 for Layer 2
        l2_total_weight = self.config.hyde_weight + self.config.expansion_weight
        l2_hyde_weight = self.config.hyde_weight / l2_total_weight  # 0.5
        l2_exp_weight = self.config.expansion_weight / l2_total_weight  # 0.5

        fused_results = []
        for section_id, data in section_data.items():
            hyde_norm = self._normalize_score(data["hyde"], hyde_min, hyde_max)
            exp0_norm = self._normalize_score(data["exp0"], exp0_min, exp0_max)
            exp1_norm = self._normalize_score(data["exp1"], exp1_min, exp1_max)

            exp_avg = (exp0_norm + exp1_norm) / 2.0
            # Use normalized weights so scores range from 0 to 1.0 (not 0 to 0.5)
            fused_score = (
                l2_hyde_weight * hyde_norm
                + l2_exp_weight * exp_avg
            )

            result = data["data"].copy()
            result["score"] = fused_score
            fused_results.append(result)

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        top_k = fused_results[:k]

        logger.info(
            f"L2 fusion search complete: {len(section_data)} candidates → {len(top_k)} sections"
        )

        return top_k

    async def _parallel_search_layer2(
        self,
        hyde_emb: np.ndarray,
        exp_0_emb: np.ndarray,
        exp_1_emb: np.ndarray,
        k: int,
        document_filter: Optional[str],
    ) -> tuple:
        """
        Execute parallel Layer 2 searches with asyncio.gather.
        Uses return_exceptions=True to handle partial failures gracefully.
        """
        results = await asyncio.gather(
            self.vector_store._async_search_layer2(
                query_embedding=hyde_emb,
                k=k,
                document_filter=document_filter,
            ),
            self.vector_store._async_search_layer2(
                query_embedding=exp_0_emb,
                k=k,
                document_filter=document_filter,
            ),
            self.vector_store._async_search_layer2(
                query_embedding=exp_1_emb,
                k=k,
                document_filter=document_filter,
            ),
            return_exceptions=True,  # Don't fail all if one fails
        )

        # Handle partial failures
        search_names = ["hyde", "expansion_0", "expansion_1"]
        processed_results = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"L2 parallel search '{search_names[i]}' failed: {result}",
                    exc_info=result if logger.isEnabledFor(logging.DEBUG) else None
                )
                processed_results.append([])
                failed_count += 1
            else:
                processed_results.append(result)

        if failed_count == 3:
            first_error = next(r for r in results if isinstance(r, Exception))
            raise RuntimeError(
                f"All L2 parallel searches failed. First error: {first_error}"
            ) from first_error

        if failed_count > 0:
            logger.warning(
                f"{failed_count}/3 L2 parallel searches failed, proceeding with partial results"
            )

        return tuple(processed_results)
