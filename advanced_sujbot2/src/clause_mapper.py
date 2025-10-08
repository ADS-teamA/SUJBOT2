"""
Contract Clause Mapper for compliance analysis.

Maps each contract clause to relevant law provisions for comparison.
"""

import logging
from typing import List, Dict, Any

from .compliance_types import ClauseMapping, LegalRequirement


class ContractClauseMapper:
    """Map contract clauses to relevant law provisions."""

    def __init__(
        self,
        cross_doc_retriever: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize ContractClauseMapper.

        Args:
            cross_doc_retriever: ComparativeRetriever instance for cross-document search
            config: Configuration dictionary
        """
        self.retriever = cross_doc_retriever
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def map_clauses(
        self,
        contract_chunks: List[Any],
        law_requirements: List[LegalRequirement]
    ) -> List[ClauseMapping]:
        """
        Map contract clauses to law requirements.

        Args:
            contract_chunks: List of contract chunk objects
            law_requirements: List of LegalRequirement objects

        Returns:
            List of ClauseMapping objects
        """
        mappings = []

        for contract_chunk in contract_chunks:
            # Find relevant law requirements for this clause
            relevant_requirements = await self._find_relevant_requirements(
                contract_chunk,
                law_requirements
            )

            if relevant_requirements:
                mapping = ClauseMapping(
                    contract_chunk_id=contract_chunk.chunk_id,
                    contract_reference=getattr(contract_chunk, 'legal_reference', 'N/A'),
                    contract_text=contract_chunk.content,
                    law_requirements=relevant_requirements,
                    match_score=self._compute_average_match_score(relevant_requirements),
                    match_type="semantic"  # Simplified; would use actual match type
                )
                mappings.append(mapping)

        self.logger.info(f"Mapped {len(mappings)} contract clauses to law requirements")

        return mappings

    async def _find_relevant_requirements(
        self,
        contract_chunk: Any,
        all_requirements: List[LegalRequirement]
    ) -> List[LegalRequirement]:
        """
        Find law requirements relevant to this contract clause.

        Args:
            contract_chunk: Contract chunk object
            all_requirements: All available legal requirements

        Returns:
            List of relevant LegalRequirement objects
        """
        # Build a mapping of chunk_id to requirements for fast lookup
        chunk_to_requirements = {}
        for req in all_requirements:
            if req.law_chunk_id not in chunk_to_requirements:
                chunk_to_requirements[req.law_chunk_id] = []
            chunk_to_requirements[req.law_chunk_id].append(req)

        # Use cross-document retriever to find relevant law chunks
        query = f"Jaké zákonné požadavky se vztahují k: {contract_chunk.content[:200]}"

        try:
            if self.retriever:
                cross_doc_results = await self.retriever.search(
                    query=query,
                    source_chunks=[contract_chunk],
                    target_document_types=["law_code"],
                    top_k=5
                )

                # Map results to requirements
                relevant_requirements = []
                for result in cross_doc_results:
                    # Find requirements from this law chunk
                    chunk_id = result.target_chunk.chunk_id
                    if chunk_id in chunk_to_requirements:
                        relevant_requirements.extend(chunk_to_requirements[chunk_id])

                # Deduplicate
                unique_reqs = list({req.requirement_id: req for req in relevant_requirements}.values())

                return unique_reqs[:5]  # Top 5
            else:
                # Fallback: use semantic similarity on summaries
                return self._fallback_matching(contract_chunk, all_requirements)

        except Exception as e:
            self.logger.warning(f"Error in cross-doc retrieval: {e}. Using fallback matching.")
            return self._fallback_matching(contract_chunk, all_requirements)

    def _fallback_matching(
        self,
        contract_chunk: Any,
        all_requirements: List[LegalRequirement]
    ) -> List[LegalRequirement]:
        """
        Fallback matching using simple text similarity.

        Args:
            contract_chunk: Contract chunk object
            all_requirements: All available legal requirements

        Returns:
            List of relevant requirements
        """
        # Simple keyword-based matching
        contract_text_lower = contract_chunk.content.lower()
        scored_requirements = []

        for req in all_requirements:
            # Simple scoring based on shared words
            req_words = set(req.requirement_summary.lower().split())
            contract_words = set(contract_text_lower.split())
            overlap = len(req_words.intersection(contract_words))

            if overlap > 0:
                score = overlap / len(req_words) if req_words else 0.0
                scored_requirements.append((score, req))

        # Sort by score and return top 5
        scored_requirements.sort(reverse=True, key=lambda x: x[0])
        return [req for score, req in scored_requirements[:5]]

    def _compute_average_match_score(self, requirements: List[LegalRequirement]) -> float:
        """Compute average extraction confidence as match score."""
        if not requirements:
            return 0.0
        return sum(req.extraction_confidence for req in requirements) / len(requirements)
