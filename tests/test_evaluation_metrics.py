"""
Tests for evaluation metrics (DCG, NDCG, MRR, Precision@k, Recall@k)

These tests validate the correctness of IR evaluation metrics used across
all evaluation scripts. Critical for ensuring benchmark validity.
"""

import pytest
import numpy as np
import math
from src.utils.eval_metrics import (
    dcg_at_k,
    ndcg_at_k,
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


class TestDCGCalculation:
    """Test Discounted Cumulative Gain calculations."""

    def test_dcg_perfect_ranking(self):
        """Perfect ranking should give maximum DCG."""
        # All relevant, perfectly ranked
        relevances = [1, 1, 1]
        dcg = dcg_at_k(relevances, k=3)
        # Expected: 1/log2(2) + 1/log2(3) + 1/log2(4)
        # = 1.0 + 0.631 + 0.5 = 2.131
        assert 2.13 < dcg < 2.14

    def test_dcg_with_zeros(self):
        """Non-relevant items should contribute 0 to DCG."""
        relevances = [1, 0, 1, 0]
        dcg = dcg_at_k(relevances, k=4)
        # Expected: 1/log2(2) + 0 + 1/log2(4) + 0
        expected = 1.0 / np.log2(2) + 1.0 / np.log2(4)
        assert abs(dcg - expected) < 0.01

    def test_dcg_empty_list(self):
        """Empty relevance list should return 0."""
        assert dcg_at_k([], k=10) == 0.0

    def test_dcg_k_larger_than_list(self):
        """k larger than list should use full list."""
        relevances = [1, 1]
        dcg_full = dcg_at_k(relevances, k=10)
        dcg_exact = dcg_at_k(relevances, k=2)
        assert dcg_full == dcg_exact

    def test_dcg_k_zero(self):
        """k=0 should return 0."""
        relevances = [1, 1, 1]
        assert dcg_at_k(relevances, k=0) == 0.0

    def test_dcg_rank_order_matters(self):
        """DCG should be higher when relevant items ranked first."""
        # Relevant at position 1
        dcg_first = dcg_at_k([1, 0, 0], k=3)
        # Relevant at position 3
        dcg_last = dcg_at_k([0, 0, 1], k=3)
        assert dcg_first > dcg_last


class TestNDCGCalculation:
    """Test Normalized Discounted Cumulative Gain."""

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk3"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 1.0

    def test_ndcg_worst_ranking(self):
        """No relevant items retrieved should give NDCG = 0.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk4", "chunk5"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 0.0

    def test_ndcg_partial_ranking(self):
        """Partial match should give NDCG between 0 and 1."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk3"}  # chunk2 is not relevant
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 < ndcg < 1.0
        # Should be less than perfect since chunk2 is irrelevant
        assert ndcg < 1.0

    def test_ndcg_empty_relevant_set(self):
        """Empty relevant set should return 0.0 (no correct answer exists)."""
        retrieved = ["chunk1", "chunk2"]
        relevant = set()
        ndcg = ndcg_at_k(retrieved, relevant, k=2)
        assert ndcg == 0.0

    def test_ndcg_fewer_relevant_than_k(self):
        """If fewer relevant items than k, ideal DCG should use min(k, |relevant|)."""
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        relevant = {"chunk1", "chunk3"}  # Only 2 relevant
        ndcg = ndcg_at_k(retrieved, relevant, k=5)
        # Ideal should be calculated with only 2 items, not 5
        assert 0.0 < ndcg <= 1.0

    def test_ndcg_rank_order_matters(self):
        """Better ranking should give higher NDCG."""
        relevant = {"chunk1", "chunk2"}

        # Perfect ranking
        retrieved_perfect = ["chunk1", "chunk2", "chunk3"]
        ndcg_perfect = ndcg_at_k(retrieved_perfect, relevant, k=3)

        # Imperfect ranking
        retrieved_imperfect = ["chunk3", "chunk1", "chunk2"]
        ndcg_imperfect = ndcg_at_k(retrieved_imperfect, relevant, k=3)

        assert ndcg_perfect > ndcg_imperfect
        assert ndcg_perfect == 1.0

    def test_ndcg_handles_list_relevant_ids(self):
        """Should accept list as well as set for relevant_ids."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant_set = {"chunk1", "chunk3"}
        relevant_list = ["chunk1", "chunk3"]

        ndcg_set = ndcg_at_k(retrieved, relevant_set, k=3)
        ndcg_list = ndcg_at_k(retrieved, relevant_list, k=3)

        assert ndcg_set == ndcg_list


class TestReciprocalRank:
    """Test Mean Reciprocal Rank (MRR) calculation."""

    def test_mrr_first_position(self):
        """First result relevant should give RR = 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1"}
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 1.0

    def test_mrr_second_position(self):
        """Second result relevant should give RR = 0.5."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk2"}
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 0.5

    def test_mrr_third_position(self):
        """Third result relevant should give RR = 1/3."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk3"}
        rr = reciprocal_rank(retrieved, relevant)
        assert abs(rr - 1/3) < 0.01

    def test_mrr_no_relevant(self):
        """No relevant results should give RR = 0.0."""
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3"}
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 0.0

    def test_mrr_multiple_relevant_returns_first(self):
        """Should return reciprocal of FIRST relevant match."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk2", "chunk3"}  # Both at ranks 2 and 3
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 0.5  # 1/2, not (1/2 + 1/3)

    def test_mrr_empty_retrieved(self):
        """Empty retrieved list should return 0.0."""
        retrieved = []
        relevant = {"chunk1"}
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 0.0

    def test_mrr_handles_list_relevant_ids(self):
        """Should accept list as well as set for relevant_ids."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant_set = {"chunk2"}
        relevant_list = ["chunk2"]

        rr_set = reciprocal_rank(retrieved, relevant_set)
        rr_list = reciprocal_rank(retrieved, relevant_list)

        assert rr_set == rr_list


class TestPrecisionAtK:
    """Test Precision@k calculations."""

    def test_precision_perfect(self):
        """All retrieved items relevant should give P@k = 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk3", "chunk4"}
        prec = precision_at_k(retrieved, relevant, k=3)
        assert prec == 1.0

    def test_precision_partial(self):
        """Some irrelevant items should give P@k < 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk3"}  # chunk2 not relevant
        prec = precision_at_k(retrieved, relevant, k=3)
        assert prec == 2/3  # 2 out of 3 retrieved are relevant

    def test_precision_none_relevant(self):
        """No relevant items retrieved should give P@k = 0.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk4", "chunk5"}
        prec = precision_at_k(retrieved, relevant, k=3)
        assert prec == 0.0

    def test_precision_zero_k(self):
        """k=0 should return 0.0 (edge case)."""
        retrieved = ["chunk1"]
        relevant = {"chunk1"}
        prec = precision_at_k(retrieved, relevant, k=0)
        assert prec == 0.0

    def test_precision_k_larger_than_retrieved(self):
        """k larger than retrieved list should use full list."""
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk1"}
        # k=5 but only 2 items retrieved, so P@5 = 1/5 (not 1/2)
        prec = precision_at_k(retrieved, relevant, k=5)
        assert prec == 1/5

    def test_precision_only_uses_top_k(self):
        """Should only consider top k items."""
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
        relevant = {"chunk3", "chunk4"}  # Only ranks 3 and 4 are relevant

        # At k=2, neither is relevant
        prec_k2 = precision_at_k(retrieved, relevant, k=2)
        assert prec_k2 == 0.0

        # At k=4, both are relevant
        prec_k4 = precision_at_k(retrieved, relevant, k=4)
        assert prec_k4 == 0.5


class TestRecallAtK:
    """Test Recall@k calculations."""

    def test_recall_perfect(self):
        """All relevant items retrieved should give R@k = 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
        relevant = {"chunk1", "chunk2"}
        rec = recall_at_k(retrieved, relevant, k=4)
        assert rec == 1.0

    def test_recall_partial(self):
        """Missing relevant items should give R@k < 1.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk4", "chunk5"}
        rec = recall_at_k(retrieved, relevant, k=3)
        assert rec == 0.5  # Found 2 out of 4 relevant

    def test_recall_none_found(self):
        """No relevant items found should give R@k = 0.0."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk4", "chunk5"}
        rec = recall_at_k(retrieved, relevant, k=3)
        assert rec == 0.0

    def test_recall_empty_relevant_set(self):
        """Empty relevant set should return 0.0 (no items to recall)."""
        retrieved = ["chunk1", "chunk2"]
        relevant = set()
        rec = recall_at_k(retrieved, relevant, k=2)
        assert rec == 0.0

    def test_recall_increases_with_k(self):
        """Recall should increase (or stay same) as k increases."""
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        relevant = {"chunk2", "chunk4", "chunk6"}  # chunk6 not in retrieved

        rec_k1 = recall_at_k(retrieved, relevant, k=1)
        rec_k2 = recall_at_k(retrieved, relevant, k=2)
        rec_k5 = recall_at_k(retrieved, relevant, k=5)

        assert rec_k1 <= rec_k2 <= rec_k5

    def test_recall_only_uses_top_k(self):
        """Should only consider top k items."""
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
        relevant = {"chunk3", "chunk4"}

        # At k=2, neither is found
        rec_k2 = recall_at_k(retrieved, relevant, k=2)
        assert rec_k2 == 0.0

        # At k=4, both are found
        rec_k4 = recall_at_k(retrieved, relevant, k=4)
        assert rec_k4 == 1.0


class TestEdgeCases:
    """Test edge cases and combinations."""

    def test_all_metrics_with_empty_retrieved(self):
        """All metrics should handle empty retrieved list gracefully."""
        retrieved = []
        relevant = {"chunk1", "chunk2"}
        k = 10

        assert ndcg_at_k(retrieved, relevant, k) == 0.0
        assert reciprocal_rank(retrieved, relevant) == 0.0
        assert precision_at_k(retrieved, relevant, k) == 0.0
        assert recall_at_k(retrieved, relevant, k) == 0.0

    def test_realistic_scenario(self):
        """Test with realistic retrieval scenario."""
        # Retrieved 10 documents
        retrieved = [
            "doc1", "doc2", "doc3", "doc4", "doc5",
            "doc6", "doc7", "doc8", "doc9", "doc10"
        ]
        # 3 relevant documents, 2 were retrieved at ranks 1 and 5
        relevant = {"doc1", "doc5", "doc99"}

        k = 10

        # Should find first relevant at rank 1
        rr = reciprocal_rank(retrieved, relevant)
        assert rr == 1.0

        # Should find 2 out of 3 relevant
        rec = recall_at_k(retrieved, relevant, k)
        assert rec == 2/3

        # Should have 2 out of 10 retrieved relevant
        prec = precision_at_k(retrieved, relevant, k)
        assert prec == 2/10

        # NDCG should be less than perfect but non-zero
        ndcg = ndcg_at_k(retrieved, relevant, k)
        assert 0.0 < ndcg < 1.0

    def test_metrics_are_deterministic(self):
        """Same input should always give same output."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk3"}
        k = 3

        # Call each metric multiple times
        ndcgs = [ndcg_at_k(retrieved, relevant, k) for _ in range(5)]
        rrs = [reciprocal_rank(retrieved, relevant) for _ in range(5)]
        precs = [precision_at_k(retrieved, relevant, k) for _ in range(5)]
        recs = [recall_at_k(retrieved, relevant, k) for _ in range(5)]

        # All values should be identical
        assert len(set(ndcgs)) == 1
        assert len(set(rrs)) == 1
        assert len(set(precs)) == 1
        assert len(set(recs)) == 1
