"""
HyDE + Expansion Fusion Retriever

Core retrieval algorithm:
1. Generate HyDE document + 2 query expansions (LLM)
2. Embed all variants (3 embeddings)
3. Search vector store with each embedding
4. Min-max normalize each score set
5. Weighted fusion: final = 0.6 * hyde + 0.4 * avg(expansions)
6. Return top-k results

Research basis:
- HyDE: Gao et al. (2022) - +15-30% recall for zero-shot retrieval
- Query Expansion: Standard IR technique for vocabulary mismatch
- Weighted Fusion: Empirically optimized (w_hyde=0.6, w_exp=0.4)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .deepinfra_client import DeepInfraClient
from .hyde_expansion import HyDEExpansionGenerator, HyDEExpansionResult

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for fusion retrieval."""

    hyde_weight: float = 0.6  # Weight for HyDE scores
    expansion_weight: float = 0.4  # Weight for expansion scores (split between 2)
    default_k: int = 10  # Default number of results
    candidates_multiplier: int = 3  # Retrieve k * multiplier candidates per query

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.hyde_weight <= 1:
            raise ValueError(f"hyde_weight must be in [0, 1], got {self.hyde_weight}")
        if not 0 <= self.expansion_weight <= 1:
            raise ValueError(f"expansion_weight must be in [0, 1], got {self.expansion_weight}")
        if abs(self.hyde_weight + self.expansion_weight - 1.0) > 1e-6:
            raise ValueError(
                f"hyde_weight + expansion_weight must equal 1.0, "
                f"got {self.hyde_weight} + {self.expansion_weight} = {self.hyde_weight + self.expansion_weight}"
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
    2. Embed hyde_doc, expansion_0, expansion_1
    3. Search PostgreSQL with each embedding (k * 3 candidates)
    4. Collect unique chunks with scores from each method
    5. Min-max normalize each score set to [0, 1]
    6. Fuse: final = 0.6 * hyde_norm + 0.4 * avg(exp_0_norm, exp_1_norm)
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

        logger.info(
            f"FusionRetriever initialized "
            f"(w_hyde={self.config.hyde_weight}, "
            f"w_exp={self.config.expansion_weight}, "
            f"k={self.config.default_k})"
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
        candidates_k = k * self.config.candidates_multiplier

        logger.info(f"Fusion search: '{query[:50]}...' (k={k})")

        # Step 1: Generate HyDE + expansions
        try:
            hyde_result = self.generator.generate(query)
            logger.debug(f"Generated HyDE: {hyde_result.hyde_document[:100]}...")
        except Exception as e:
            logger.error(f"HyDE generation failed for query '{query[:50]}...': {e}", exc_info=True)
            raise RuntimeError(f"HyDE generation failed: {e}") from e

        # Step 2: Embed all variants
        try:
            texts_to_embed = [
                hyde_result.hyde_document,
                hyde_result.expansions[0],
                hyde_result.expansions[1],
            ]
            embeddings = self.client.embed_texts(texts_to_embed)
        except Exception as e:
            logger.error(f"Embedding failed for fusion search: {e}", exc_info=True)
            raise RuntimeError(f"Embedding failed: {e}") from e

        hyde_emb = embeddings[0]
        exp_0_emb = embeddings[1]
        exp_1_emb = embeddings[2]

        # Step 3: Search with each embedding
        try:
            hyde_results = self.vector_store.search_layer3(
                query_embedding=hyde_emb,
                k=candidates_k,
                document_filter=document_filter,
            )

            exp_0_results = self.vector_store.search_layer3(
                query_embedding=exp_0_emb,
                k=candidates_k,
                document_filter=document_filter,
            )

            exp_1_results = self.vector_store.search_layer3(
                query_embedding=exp_1_emb,
                k=candidates_k,
                document_filter=document_filter,
            )
        except Exception as e:
            logger.error(f"Vector store search failed: {e}", exc_info=True)
            raise RuntimeError(f"Vector store search failed: {e}") from e

        logger.debug(
            f"Retrieved candidates: hyde={len(hyde_results)}, "
            f"exp0={len(exp_0_results)}, exp1={len(exp_1_results)}"
        )

        # Step 4: Collect unique chunks with scores
        chunk_data = {}  # chunk_id -> {hyde: score, exp0: score, exp1: score, data: dict}

        for result in hyde_results:
            chunk_id = result["chunk_id"]
            chunk_data[chunk_id] = {
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
                    "hyde": None,
                    "exp0": None,
                    "exp1": result["score"],
                    "data": result,
                }

        if not chunk_data:
            logger.warning("No results found for fusion search")
            return []

        # Step 5: Min-max normalize each score set
        hyde_scores = [d["hyde"] for d in chunk_data.values() if d["hyde"] is not None]
        exp0_scores = [d["exp0"] for d in chunk_data.values() if d["exp0"] is not None]
        exp1_scores = [d["exp1"] for d in chunk_data.values() if d["exp1"] is not None]

        hyde_min, hyde_max = self._get_min_max(hyde_scores)
        exp0_min, exp0_max = self._get_min_max(exp0_scores)
        exp1_min, exp1_max = self._get_min_max(exp1_scores)

        # Step 6: Weighted fusion
        fused_results = []
        for chunk_id, data in chunk_data.items():
            # Normalize scores (0 if missing)
            hyde_norm = self._normalize_score(data["hyde"], hyde_min, hyde_max)
            exp0_norm = self._normalize_score(data["exp0"], exp0_min, exp0_max)
            exp1_norm = self._normalize_score(data["exp1"], exp1_min, exp1_max)

            # Average expansion scores
            exp_avg = (exp0_norm + exp1_norm) / 2.0

            # Weighted fusion
            fused_score = (
                self.config.hyde_weight * hyde_norm
                + self.config.expansion_weight * exp_avg
            )

            # Build result
            result = data["data"].copy()
            result["score"] = fused_score
            result["hyde_score"] = data["hyde"]
            result["exp0_score"] = data["exp0"]
            result["exp1_score"] = data["exp1"]
            fused_results.append(result)

        # Step 7: Sort and return top-k
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        top_k = fused_results[:k]

        logger.info(
            f"Fusion search complete: {len(chunk_data)} candidates → {len(top_k)} results"
        )

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
