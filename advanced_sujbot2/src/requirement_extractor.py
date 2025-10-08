"""
Requirement Extractor for compliance analysis.

Extracts legal requirements (obligations, prohibitions, conditions) from law documents
using both pattern-based and LLM-based approaches.
"""

import logging
import re
import json
from typing import List, Any, Dict
from anthropic import Anthropic

from .compliance_types import LegalRequirement, RequirementType


class RequirementExtractor:
    """Extract legal requirements from law provisions."""

    def __init__(self, llm_client: Anthropic, config: Dict[str, Any] = None):
        """
        Initialize RequirementExtractor.

        Args:
            llm_client: Anthropic client for LLM-based extraction
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    async def extract_requirements(
        self,
        law_chunks: List[Any]
    ) -> List[LegalRequirement]:
        """
        Extract requirements from law chunks.

        Args:
            law_chunks: List of LegalChunk objects

        Returns:
            List of LegalRequirement objects
        """
        all_requirements = []

        # Process chunks in batches
        for chunk in law_chunks:
            requirements = await self._extract_from_chunk(chunk)
            all_requirements.extend(requirements)

        self.logger.info(f"Extracted {len(all_requirements)} requirements from {len(law_chunks)} chunks")

        return all_requirements

    async def _extract_from_chunk(
        self,
        chunk: Any
    ) -> List[LegalRequirement]:
        """Extract requirements from a single chunk."""
        # First: pattern-based extraction (fast)
        pattern_requirements = self._extract_by_patterns(chunk)

        # Then: LLM-based extraction for nuanced requirements
        if self.config.get("llm_for_nuanced_requirements", True):
            llm_requirements = await self._extract_with_llm(chunk)
        else:
            llm_requirements = []

        # Merge and deduplicate
        all_requirements = pattern_requirements + llm_requirements
        deduplicated = self._deduplicate_requirements(all_requirements)

        return deduplicated

    def _extract_by_patterns(self, chunk: Any) -> List[LegalRequirement]:
        """Fast pattern-based extraction for common requirement phrases."""
        requirements = []
        text = chunk.content

        # Obligation patterns (Czech and English)
        obligation_patterns = [
            r"(musí|je povinen|povinnost)\s+([^.]+\.)",
            r"([^.]+)\s+je\s+povinen\s+([^.]+\.)",
            r"(shall|must)\s+([^.]+\.)",
        ]

        for i, pattern in enumerate(obligation_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_pat_obl_{i}_{len(requirements)}",
                    requirement_type=RequirementType.OBLIGATION,
                    law_reference=getattr(chunk, 'legal_reference', 'N/A'),
                    law_document_id=getattr(chunk, 'document_id', 'unknown'),
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=match.group(0),
                    requirement_summary=match.group(0)[:100],
                    is_mandatory=True,
                    applies_to=["all_parties"],
                    extraction_confidence=0.8
                )
                requirements.append(req)

        # Prohibition patterns (Czech and English)
        prohibition_patterns = [
            r"(nesmí|zakázáno|zákaz)\s+([^.]+\.)",
            r"(must not|shall not|prohibited)\s+([^.]+\.)",
        ]

        for i, pattern in enumerate(prohibition_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_pat_proh_{i}_{len(requirements)}",
                    requirement_type=RequirementType.PROHIBITION,
                    law_reference=getattr(chunk, 'legal_reference', 'N/A'),
                    law_document_id=getattr(chunk, 'document_id', 'unknown'),
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=match.group(0),
                    requirement_summary=match.group(0)[:100],
                    is_mandatory=True,
                    applies_to=["all_parties"],
                    extraction_confidence=0.8
                )
                requirements.append(req)

        # Condition patterns
        condition_patterns = [
            r"(pokud|jestliže|v případě že)\s+([^,]+),\s+([^.]+\.)",
            r"(if|when|in case)\s+([^,]+),\s+([^.]+\.)",
        ]

        for i, pattern in enumerate(condition_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_pat_cond_{i}_{len(requirements)}",
                    requirement_type=RequirementType.CONDITION,
                    law_reference=getattr(chunk, 'legal_reference', 'N/A'),
                    law_document_id=getattr(chunk, 'document_id', 'unknown'),
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=match.group(0),
                    requirement_summary=match.group(0)[:100],
                    is_mandatory=True,
                    applies_to=["all_parties"],
                    extraction_confidence=0.75
                )
                requirements.append(req)

        return requirements

    async def _extract_with_llm(self, chunk: Any) -> List[LegalRequirement]:
        """
        Use LLM to extract nuanced requirements.

        Args:
            chunk: LegalChunk object

        Returns:
            List of LegalRequirement objects
        """
        prompt = f"""Analyze this legal provision and extract all requirements.

Legal provision ({getattr(chunk, 'legal_reference', 'N/A')}):
{chunk.content}

For each requirement, identify:
1. Type (obligation, prohibition, permission, condition, definition)
2. Who it applies to (contractor, all parties, specific role)
3. What the requirement is (brief summary)
4. Whether it's mandatory or optional
5. Any temporal constraints (deadlines, durations)

Output format (JSON array):
[
  {{
    "type": "obligation",
    "applies_to": ["contractor"],
    "summary": "Must submit monthly reports",
    "is_mandatory": true,
    "temporal_constraint": "within 5 days of month end"
  }}
]

If no clear requirements are found, return an empty array [].

Requirements:"""

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()

            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                requirements_data = json.loads(json_str)
            else:
                requirements_data = json.loads(response_text)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM requirements for {chunk.chunk_id}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error in LLM extraction for {chunk.chunk_id}: {e}")
            return []

        # Convert to LegalRequirement objects
        requirements = []
        for i, req_data in enumerate(requirements_data):
            try:
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_llm_{i}",
                    requirement_type=RequirementType(req_data["type"]),
                    law_reference=getattr(chunk, 'legal_reference', 'N/A'),
                    law_document_id=getattr(chunk, 'document_id', 'unknown'),
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=chunk.content,  # Full chunk text
                    requirement_summary=req_data["summary"],
                    is_mandatory=req_data.get("is_mandatory", True),
                    applies_to=req_data.get("applies_to", ["all_parties"]),
                    temporal_constraint=req_data.get("temporal_constraint"),
                    extraction_confidence=0.95  # LLM extraction is high confidence
                )
                requirements.append(req)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping malformed requirement {i} in {chunk.chunk_id}: {e}")
                continue

        return requirements

    def _deduplicate_requirements(
        self,
        requirements: List[LegalRequirement]
    ) -> List[LegalRequirement]:
        """Remove duplicate requirements."""
        # Simple deduplication by summary text similarity
        unique = []
        seen_summaries = set()

        for req in requirements:
            summary_lower = req.requirement_summary.lower().strip()
            if summary_lower and summary_lower not in seen_summaries:
                unique.append(req)
                seen_summaries.add(summary_lower)

        return unique
