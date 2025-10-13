"""
Multi-round retrieval module for iterative query refinement.

Based on research showing recall improvement from 57% (single-round)
to 79% (multi-round) for legal document retrieval.

Strategy:
- Round 1: Initial semantic retrieval with contract chunk as query
- Round 2: Query expansion based on Round 1 results + reranking
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from one retrieval round"""
    round_number: int
    query: str
    matches: List[Any]  # DocumentPair objects
    retrieval_time_ms: float
    match_scores: List[float] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        """Average match score"""
        if not self.match_scores:
            return 0.0
        return sum(self.match_scores) / len(self.match_scores)


@dataclass
class MultiRoundResult:
    """Combined result from all retrieval rounds"""
    contract_chunk: Any
    rounds: List[RetrievalResult]
    final_matches: List[Any]  # Top-K after all rounds
    total_time_ms: float

    @property
    def recall_improvement(self) -> float:
        """Estimate recall improvement from multi-round"""
        if len(self.rounds) < 2:
            return 0.0
        # Simple heuristic: compare unique matches round 1 vs final
        round1_ids = {m.target_chunk.chunk_id for m in self.rounds[0].matches}
        final_ids = {m.target_chunk.chunk_id for m in self.final_matches}
        new_matches = len(final_ids - round1_ids)
        return (new_matches / len(final_ids)) * 100 if final_ids else 0.0


class MultiRoundRetriever:
    """
    Multi-round retrieval with iterative query refinement.

    Implements two-round strategy:
    1. Initial retrieval with contract chunk content
    2. Refined retrieval with expanded query based on Round 1 results
    """

    def __init__(
        self,
        comparative_retriever: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-round retriever.

        Args:
            comparative_retriever: ComparativeRetriever instance from cross_doc_retrieval.py
            config: Configuration dictionary
        """
        self.retriever = comparative_retriever
        self.config = config or {}

        # Round configuration
        self.num_rounds = self.config.get('num_rounds', 2)
        self.round1_top_k = self.config.get('round1_top_k', 10)
        self.round2_top_k = self.config.get('round2_top_k', 5)
        self.final_top_k = self.config.get('final_top_k', 5)

        # Query expansion configuration
        self.expand_with_keywords = self.config.get('expand_with_keywords', True)
        self.expand_with_references = self.config.get('expand_with_references', True)

        self.logger = logging.getLogger(__name__)

    async def retrieve_multi_round(
        self,
        contract_chunk: Any,
        target_document_id: str
    ) -> MultiRoundResult:
        """
        Perform multi-round retrieval for a contract chunk.

        Args:
            contract_chunk: LegalChunk from contract
            target_document_id: Law document ID to search

        Returns:
            MultiRoundResult with all rounds and final top matches
        """
        import time
        start_time = time.time()

        rounds = []

        # Round 1: Initial retrieval
        self.logger.debug(f"Round 1: Initial retrieval for chunk {contract_chunk.chunk_id}")
        round1_result = await self._round1_initial_retrieval(
            contract_chunk,
            target_document_id
        )
        rounds.append(round1_result)

        # Round 2: Refined retrieval with query expansion
        if self.num_rounds >= 2 and round1_result.matches:
            self.logger.debug(f"Round 2: Refined retrieval with query expansion")
            round2_result = await self._round2_refined_retrieval(
                contract_chunk,
                target_document_id,
                round1_result
            )
            rounds.append(round2_result)

        # Merge and deduplicate results from all rounds
        final_matches = self._merge_and_rerank_rounds(rounds)

        total_time = (time.time() - start_time) * 1000

        result = MultiRoundResult(
            contract_chunk=contract_chunk,
            rounds=rounds,
            final_matches=final_matches[:self.final_top_k],
            total_time_ms=total_time
        )

        self.logger.info(
            f"Multi-round retrieval complete for chunk {contract_chunk.chunk_id}: "
            f"{len(final_matches)} matches in {total_time:.0f}ms "
            f"(recall improvement: {result.recall_improvement:.1f}%)"
        )

        return result

    async def _round1_initial_retrieval(
        self,
        contract_chunk: Any,
        target_document_id: str
    ) -> RetrievalResult:
        """
        Round 1: Initial semantic retrieval using contract chunk content.

        Uses ComparativeRetriever with default settings (semantic + explicit + structural).
        """
        import time
        start_time = time.time()

        # Build query from contract chunk content
        query = self._build_initial_query(contract_chunk)

        # Use existing ComparativeRetriever
        cross_doc_results = await self.retriever.find_related_provisions(
            source_chunk=contract_chunk,
            target_document_id=target_document_id,
            top_k=self.round1_top_k
        )

        matches = cross_doc_results.pairs if hasattr(cross_doc_results, 'pairs') else cross_doc_results.get_top_pairs(self.round1_top_k)
        match_scores = [m.overall_score for m in matches]

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            round_number=1,
            query=query,
            matches=matches,
            retrieval_time_ms=retrieval_time,
            match_scores=match_scores
        )

    async def _round2_refined_retrieval(
        self,
        contract_chunk: Any,
        target_document_id: str,
        round1_result: RetrievalResult
    ) -> RetrievalResult:
        """
        Round 2: Refined retrieval with query expansion.

        Expands query based on Round 1 results:
        - Extract key terms from top matches
        - Add legal references found
        - Boost with keywords from matched law chunks
        """
        import time
        start_time = time.time()

        # Build refined query with expansion
        refined_query = self._build_refined_query(
            contract_chunk,
            round1_result.matches
        )

        # Create temporary chunk with expanded query
        expanded_chunk = self._create_expanded_chunk(contract_chunk, refined_query)

        # Use ComparativeRetriever with expanded query
        cross_doc_results = await self.retriever.find_related_provisions(
            source_chunk=expanded_chunk,
            target_document_id=target_document_id,
            top_k=self.round2_top_k
        )

        matches = cross_doc_results.pairs if hasattr(cross_doc_results, 'pairs') else cross_doc_results.get_top_pairs(self.round2_top_k)
        match_scores = [m.overall_score for m in matches]

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            round_number=2,
            query=refined_query,
            matches=matches,
            retrieval_time_ms=retrieval_time,
            match_scores=match_scores
        )

    def _build_initial_query(self, contract_chunk: Any) -> str:
        """Build initial query from contract chunk"""
        # Use full content for Round 1
        query = contract_chunk.content

        # Add legal reference if available (explicit matching boost)
        if hasattr(contract_chunk, 'legal_reference') and contract_chunk.legal_reference:
            query = f"{contract_chunk.legal_reference}: {query}"

        return query[:1000]  # Limit query length

    def _build_refined_query(
        self,
        contract_chunk: Any,
        round1_matches: List[Any]
    ) -> str:
        """
        Build refined query with expansion based on Round 1 results.

        Expansion strategies:
        1. Extract keywords from top-3 matches
        2. Add legal references from matched law chunks
        3. Boost query with domain-specific terms
        """
        # Start with original content (shorter version)
        query = contract_chunk.content[:300]

        # Extract keywords from top matches
        if self.expand_with_keywords and round1_matches:
            keywords = self._extract_keywords_from_matches(round1_matches[:3])
            if keywords:
                query += f"\n\nRelevantní pojmy: {', '.join(keywords[:10])}"

        # Add legal references from matches
        if self.expand_with_references and round1_matches:
            references = self._extract_references_from_matches(round1_matches[:3])
            if references:
                query += f"\n\nZákonné odkazy: {', '.join(references[:5])}"

        return query[:1500]  # Expanded query limit

    def _extract_keywords_from_matches(self, matches: List[Any]) -> List[str]:
        """
        Extract important keywords from top matches.

        Uses simple TF-IDF-like heuristic:
        - Nouns and adjectives
        - Appearing in multiple matches
        - Legal terminology
        """
        keywords = []

        # Legal terminology indicators
        legal_terms = ['povinnost', 'právo', 'záruka', 'odpovědnost', 'lhůta', 'termín']

        for match in matches:
            chunk_content = match.target_chunk.content.lower()

            # Extract legal terms
            for term in legal_terms:
                if term in chunk_content and term not in keywords:
                    keywords.append(term)

            # Add from legal reference
            if hasattr(match.target_chunk, 'legal_reference') and match.target_chunk.legal_reference:
                keywords.append(match.target_chunk.legal_reference)

        return keywords

    def _extract_references_from_matches(self, matches: List[Any]) -> List[str]:
        """Extract legal references from matches"""
        references = set()

        for match in matches:
            # Extract from legal_reference field
            if hasattr(match.target_chunk, 'legal_reference') and match.target_chunk.legal_reference:
                references.add(match.target_chunk.legal_reference)

            # Extract from evidence if available
            if hasattr(match, 'evidence') and match.evidence:
                ref = match.evidence.get('reference_text')
                if ref:
                    references.add(ref)

        return list(references)

    def _create_expanded_chunk(self, original_chunk: Any, expanded_query: str) -> Any:
        """
        Create temporary chunk with expanded query content.

        Preserves metadata from original chunk.
        """
        # Create shallow copy
        import copy
        expanded_chunk = copy.copy(original_chunk)

        # Replace content with expanded query
        expanded_chunk.content = expanded_query

        return expanded_chunk

    def _merge_and_rerank_rounds(self, rounds: List[RetrievalResult]) -> List[Any]:
        """
        Merge results from all rounds and rerank.

        Strategy:
        1. Deduplicate by target_chunk_id
        2. Combine scores (weighted by round number)
        3. Sort by combined score
        """
        # Collect all matches with deduplication
        matches_dict = {}  # chunk_id -> (match, combined_score)

        for round_idx, round_result in enumerate(rounds, 1):
            round_weight = 1.0 / round_idx  # Later rounds have lower weight

            for match in round_result.matches:
                chunk_id = match.target_chunk.chunk_id

                if chunk_id in matches_dict:
                    # Already seen - boost score
                    existing_match, existing_score = matches_dict[chunk_id]
                    combined_score = existing_score + (match.overall_score * round_weight * 0.5)
                    matches_dict[chunk_id] = (existing_match, combined_score)
                else:
                    # New match
                    combined_score = match.overall_score * round_weight
                    matches_dict[chunk_id] = (match, combined_score)

        # Sort by combined score
        sorted_matches = sorted(
            matches_dict.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return just the matches (not scores)
        return [match for match, score in sorted_matches]

    async def batch_retrieve_multi_round(
        self,
        contract_chunks: List[Any],
        target_document_id: str,
        max_concurrent: int = 5
    ) -> List[MultiRoundResult]:
        """
        Batch process multiple chunks with concurrency limit.

        Args:
            contract_chunks: List of contract chunks
            target_document_id: Law document ID
            max_concurrent: Maximum concurrent retrievals

        Returns:
            List of MultiRoundResult objects
        """
        self.logger.info(f"Batch multi-round retrieval for {len(contract_chunks)} chunks")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def retrieve_with_limit(chunk):
            async with semaphore:
                return await self.retrieve_multi_round(chunk, target_document_id)

        # Execute all retrievals in parallel (with concurrency limit)
        tasks = [retrieve_with_limit(chunk) for chunk in contract_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Retrieval failed for chunk {idx}: {result}")
            else:
                valid_results.append(result)

        self.logger.info(f"Batch retrieval complete: {len(valid_results)}/{len(contract_chunks)} successful")

        return valid_results
