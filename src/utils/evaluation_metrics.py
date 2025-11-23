"""
Evaluation Metrics for Information Retrieval

This module provides standard IR evaluation metrics (DCG, NDCG, MRR, Precision@k, Recall@k)
used across all evaluation scripts. Single source of truth for metric calculations.

All functions handle edge cases gracefully and return 0.0 for undefined cases.
"""

import numpy as np
from typing import List, Set, Union


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    Formula: DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)

    Implementation note: Uses 0-indexed positions internally, so log2(position+2)
    correctly maps to the 1-indexed formula (position 0 → log2(2), position 1 → log2(3), etc.)
    This avoids log2(1)=0 which would cause division by zero.

    Args:
        relevances: Binary relevance scores (1=relevant, 0=irrelevant) in ranked order
        k: Cutoff position (number of top results to consider)

    Returns:
        DCG score. Returns 0.0 if relevances is empty or k is 0.

    Example:
        >>> dcg_at_k([1, 1, 0, 1], k=4)
        2.630...  # 1/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5)
    """
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    # positions: [1, 2, 3, ...] (1-indexed for formula compatibility)
    positions = np.arange(1, len(relevances) + 1)
    # Formula: rel_i / log2(i+1) where i is 1-indexed position
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Normalizes DCG by the ideal DCG (DCG of perfect ranking). NDCG ranges from 0 to 1,
    where 1.0 indicates perfect ranking of all relevant items.

    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        relevant_ids: Set or list of known relevant item IDs (ground truth)
        k: Cutoff position (number of top results to consider)

    Returns:
        NDCG score in range [0.0, 1.0]. Returns 0.0 if:
        - No relevant items exist (empty relevant_ids)
        - No relevant items retrieved in top k
        - k is 0

    Example:
        >>> ndcg_at_k(['A', 'B', 'C'], {'A', 'C'}, k=3)
        0.869...  # Good but not perfect (B is irrelevant)
        >>> ndcg_at_k(['A', 'C', 'B'], {'A', 'C'}, k=3)
        1.0  # Perfect ranking
    """
    relevant_ids = set(relevant_ids)

    # Calculate actual DCG
    relevances = [1 if cid in relevant_ids else 0 for cid in retrieved_ids[:k]]
    dcg = dcg_at_k(relevances, k)

    # Calculate ideal DCG (all relevant items ranked first)
    # Use min(len(relevant_ids), k) because we can't rank more relevant items than exist
    ideal_relevances = [1] * min(len(relevant_ids), k)
    ideal = dcg_at_k(ideal_relevances, k)

    # Normalize by ideal DCG
    # Edge case: If no relevant items exist, ideal=0 and we return 0.0
    return dcg / ideal if ideal > 0 else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]]) -> float:
    """
    Mean Reciprocal Rank (MRR) calculation.

    Returns the reciprocal of the rank of the FIRST relevant result.
    Used to measure how quickly the system returns a relevant result.

    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        relevant_ids: Set or list of known relevant item IDs (ground truth)

    Returns:
        1/rank for first relevant result, or 0.0 if none found.
        Returns 0.0 if no relevant items in retrieved results (this 0.0 is
        included when computing average MRR across queries).

    Example:
        >>> reciprocal_rank(['A', 'B', 'C'], {'B'})
        0.5  # First relevant result at rank 2, so 1/2 = 0.5
        >>> reciprocal_rank(['A', 'B', 'C'], {'D'})
        0.0  # No relevant results found
    """
    relevant_ids = set(relevant_ids)

    # Find first relevant item (ranks are 1-indexed for humans)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank

    # No relevant items found
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Precision at k.

    Measures what fraction of retrieved results (top k) are relevant.
    Precision = |retrieved ∩ relevant| / k

    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        relevant_ids: Set or list of known relevant item IDs (ground truth)
        k: Cutoff position (number of top results to consider)

    Returns:
        Precision score in range [0.0, 1.0]. Returns 0.0 if k is 0.

    Example:
        >>> precision_at_k(['A', 'B', 'C', 'D'], {'A', 'C'}, k=4)
        0.5  # 2 out of 4 retrieved are relevant
        >>> precision_at_k(['A', 'B', 'C'], {'A', 'B', 'C'}, k=3)
        1.0  # All retrieved are relevant
    """
    if k == 0:
        return 0.0

    relevant_ids = set(relevant_ids)
    retrieved_k = set(retrieved_ids[:k])

    # Count how many retrieved items are relevant
    num_relevant_retrieved = len(retrieved_k & relevant_ids)

    return num_relevant_retrieved / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Recall at k.

    Measures what fraction of all relevant items were found in top k results.
    Recall = |retrieved ∩ relevant| / |relevant|

    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        relevant_ids: Set or list of known relevant item IDs (ground truth)
        k: Cutoff position (number of top results to consider)

    Returns:
        Recall score in range [0.0, 1.0]. Returns 0.0 if no relevant items exist.

    Example:
        >>> recall_at_k(['A', 'B', 'C'], {'A', 'B', 'D', 'E'}, k=3)
        0.5  # Found 2 out of 4 relevant items
        >>> recall_at_k(['A', 'B', 'C', 'D'], {'A', 'B'}, k=4)
        1.0  # Found all relevant items
    """
    relevant_ids = set(relevant_ids)

    # Edge case: No relevant items exist
    if len(relevant_ids) == 0:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])

    # Count how many relevant items were retrieved
    num_relevant_retrieved = len(retrieved_k & relevant_ids)

    return num_relevant_retrieved / len(relevant_ids)
