"""
Evaluation metrics utilities for retrieval benchmarking.

This module provides standard information retrieval metrics used across
all evaluation scripts, following the DRY principle.
"""

from typing import List, Set, Union
import numpy as np


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at position k.

    Args:
        relevances: List of relevance scores (0 or 1) for retrieved items
        k: Number of top results to consider

    Returns:
        DCG@k score
    """
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at position k.

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        NDCG@k score between 0 and 1
    """
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    relevances = [1 if cid in relevant_set else 0 for cid in retrieved_ids[:k]]
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k([1] * min(len(relevant_ids), k), k)
    return dcg / ideal if ideal > 0 else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs

    Returns:
        Reciprocal rank (1/rank of first relevant item, or 0 if none found)
    """
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate Precision at position k.

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Precision@k score between 0 and 1
    """
    if k == 0:
        return 0.0
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    return len(set(retrieved_ids[:k]) & relevant_set) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate Recall at position k.

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Recall@k score between 0 and 1
    """
    if len(relevant_ids) == 0:
        return 0.0
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    return len(set(retrieved_ids[:k]) & relevant_set) / len(relevant_ids)


def f1_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate F1 score at position k.

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        F1@k score between 0 and 1
    """
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int
) -> dict:
    """
    Calculate all standard retrieval metrics at once.

    Args:
        retrieved_ids: List of retrieved chunk IDs in ranked order
        relevant_ids: List or set of ground truth relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Dictionary with all metric scores
    """
    return {
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"f1@{k}": f1_at_k(retrieved_ids, relevant_ids, k),
    }