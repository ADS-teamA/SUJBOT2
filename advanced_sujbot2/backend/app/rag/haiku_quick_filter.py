"""
Haiku Quick Filter for fast compliance screening.

Uses Claude Haiku (fast, cheap) to quickly screen contract chunks
against law provisions. High-confidence results are accepted directly,
low-confidence results are escalated to Sonnet for deep analysis.

Chain of Verification Strategy:
- Haiku: Fast screening (80% of cases, $0.25/M tokens)
- Sonnet: Deep analysis only for edge cases (20% of cases, $3/M tokens)

Performance: Can process 400 Supreme Court cases for $1 (Haiku capability).
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status from quick check"""
    COMPLIANT = "compliant"           # Clear compliance
    NON_COMPLIANT = "non_compliant"   # Clear violation
    UNCERTAIN = "uncertain"            # Needs deeper analysis
    ERROR = "error"                    # Check failed


@dataclass
class HaikuCheckResult:
    """Result from Haiku quick check"""
    contract_chunk_id: str
    law_chunk_ids: List[str]

    # Compliance assessment
    status: ComplianceStatus
    is_compliant: bool
    confidence: float  # 0.0-1.0

    # Details
    explanation: str
    evidence: str
    potential_issues: List[str]

    # Metadata
    llm_model: str = "claude-3-5-haiku-20241022"
    processing_time_ms: float = 0.0
    needs_deep_analysis: bool = False  # Escalate to Sonnet?

    # For aggregation
    contract_text_preview: str = ""
    law_text_previews: List[str] = None


class HaikuQuickFilter:
    """
    Fast compliance screening using Claude Haiku.

    Optimized for:
    - High throughput (21K tokens/sec for <32K prompts)
    - Low cost ($0.25/M input, $1.25/M output)
    - Quick decisions (1-2 sec per chunk)
    """

    def __init__(
        self,
        llm_client: AsyncAnthropic,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Haiku quick filter.

        Args:
            llm_client: AsyncAnthropic client
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.config = config or {}

        # Model configuration
        self.model = "claude-3-5-haiku-20241022"
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.1)

        # Confidence thresholds
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.85)
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.60)

        # Retry configuration
        self.max_retries = self.config.get('max_retries', 2)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        self.logger = logging.getLogger(__name__)

    async def quick_check(
        self,
        contract_chunk: Any,
        law_matches: List[Any]  # Top-5 law chunks from retrieval
    ) -> HaikuCheckResult:
        """
        Perform quick compliance check on one contract chunk vs its top law matches.

        Args:
            contract_chunk: LegalChunk from contract
            law_matches: List of top-5 matched law chunks (DocumentPair objects)

        Returns:
            HaikuCheckResult with compliance assessment
        """
        import time
        start_time = time.time()

        # Build compact prompt (fit within token limits)
        prompt = self._build_quick_check_prompt(contract_chunk, law_matches)

        # Call Haiku with retry logic
        try:
            response = await self._call_haiku_with_retry(prompt)
            result = self._parse_response(response, contract_chunk, law_matches)
        except Exception as e:
            self.logger.error(f"Haiku check failed for chunk {contract_chunk.chunk_id}: {e}")
            result = HaikuCheckResult(
                contract_chunk_id=contract_chunk.chunk_id,
                law_chunk_ids=[],
                status=ComplianceStatus.ERROR,
                is_compliant=False,
                confidence=0.0,
                explanation=f"Error during check: {str(e)}",
                evidence="",
                potential_issues=[],
                needs_deep_analysis=True  # Escalate on error
            )

        result.processing_time_ms = (time.time() - start_time) * 1000

        # Determine if needs deep analysis
        result.needs_deep_analysis = self._should_escalate_to_sonnet(result)

        return result

    def _build_quick_check_prompt(
        self,
        contract_chunk: Any,
        law_matches: List[Any]
    ) -> str:
        """
        Build optimized prompt for Haiku.

        Keeps it concise (<5000 tokens) for fast processing.
        """
        # Extract contract text
        contract_ref = getattr(contract_chunk, 'legal_reference', 'N/A')
        contract_text = contract_chunk.content[:1500]  # Limit length

        # Extract top-5 law chunks
        law_texts = []
        for i, match in enumerate(law_matches[:5], 1):
            law_ref = getattr(match.target_chunk, 'legal_reference', f'Law {i}')
            law_text = match.target_chunk.content[:800]  # Shorter for each law
            law_texts.append(f"**{law_ref}**:\n{law_text}")

        laws_combined = "\n\n".join(law_texts)

        # Build prompt
        prompt = f"""Jsi rychlý právní compliance checker. Tvým úkolem je **rychle** zjistit, zda klauzule smlouvy je v souladu se zákonnými požadavky.

<klauzule_smlouvy ref="{contract_ref}">
{contract_text}
</klauzule_smlouvy>

<zakonné_požadavky>
{laws_combined}
</zakonné_požadavky>

<instrukce>
Proveď **rychlou analýzu**:

1. Je klauzule smlouvy v **jasném rozporu** se zákonem?
   - Příklad rozporu: smlouva říká "6 měsíců", zákon vyžaduje "24 měsíců"

2. Je klauzule **nejasná nebo sporná**? (confidence < 0.85)
   - Pokud ano, označ jako "uncertain" pro hlubší analýzu

3. Jinak je **v souladu** (nebo není relevantní)

**DŮLEŽITÉ:** Buď stručný! Max 3 věty.
</instrukce>

<output_format>
Vrať JSON:
{{
  "is_compliant": true/false,
  "confidence": 0.0-1.0,
  "status": "compliant" | "non_compliant" | "uncertain",
  "explanation": "Stručné vysvětlení (max 2 věty)",
  "evidence": "Konkrétní text z klauzule nebo zákona",
  "potential_issues": ["Problem 1", "Problem 2"]  // Pokud existují
}}

Pokud není jasné → status="uncertain", confidence<0.85
</output_format>

JSON:"""

        return prompt

    async def _call_haiku_with_retry(self, prompt: str) -> str:
        """Call Haiku with exponential backoff retry"""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.llm.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                return response.content[0].text.strip()

            except Exception as e:
                if attempt == self.max_retries:
                    raise

                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Haiku call failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    def _parse_response(
        self,
        response_text: str,
        contract_chunk: Any,
        law_matches: List[Any]
    ) -> HaikuCheckResult:
        """Parse Haiku JSON response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response_text)

            # Map to ComplianceStatus
            status_str = parsed.get('status', 'uncertain')
            status_map = {
                'compliant': ComplianceStatus.COMPLIANT,
                'non_compliant': ComplianceStatus.NON_COMPLIANT,
                'uncertain': ComplianceStatus.UNCERTAIN
            }
            status = status_map.get(status_str, ComplianceStatus.UNCERTAIN)

            result = HaikuCheckResult(
                contract_chunk_id=contract_chunk.chunk_id,
                law_chunk_ids=[m.target_chunk.chunk_id for m in law_matches],
                status=status,
                is_compliant=parsed.get('is_compliant', False),
                confidence=float(parsed.get('confidence', 0.5)),
                explanation=parsed.get('explanation', ''),
                evidence=parsed.get('evidence', ''),
                potential_issues=parsed.get('potential_issues', []),
                contract_text_preview=contract_chunk.content[:200],
                law_text_previews=[m.target_chunk.content[:150] for m in law_matches[:3]]
            )

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse Haiku response: {e}")
            # Return uncertain result on parse failure
            return HaikuCheckResult(
                contract_chunk_id=contract_chunk.chunk_id,
                law_chunk_ids=[m.target_chunk.chunk_id for m in law_matches],
                status=ComplianceStatus.UNCERTAIN,
                is_compliant=False,
                confidence=0.0,
                explanation="Parse error - needs manual review",
                evidence=response_text[:200],
                potential_issues=["Failed to parse response"],
                needs_deep_analysis=True
            )

    def _should_escalate_to_sonnet(self, result: HaikuCheckResult) -> bool:
        """
        Determine if result should be escalated to Sonnet for deep analysis.

        Escalation criteria:
        - Low confidence (< 0.60)
        - Uncertain status
        - Error during check
        - Non-compliant with medium confidence (0.60-0.85)
        """
        # Always escalate errors and uncertain
        if result.status in [ComplianceStatus.ERROR, ComplianceStatus.UNCERTAIN]:
            return True

        # Escalate low confidence
        if result.confidence < self.low_confidence_threshold:
            return True

        # Escalate non-compliant with medium confidence
        if not result.is_compliant and result.confidence < self.high_confidence_threshold:
            return True

        # High confidence results (>0.85) don't need escalation
        return False

    async def batch_quick_check(
        self,
        contract_chunks_with_matches: List[tuple[Any, List[Any]]],
        max_concurrent: int = 10
    ) -> List[HaikuCheckResult]:
        """
        Batch process multiple chunks in parallel.

        Args:
            contract_chunks_with_matches: List of (contract_chunk, law_matches) tuples
            max_concurrent: Maximum concurrent Haiku calls

        Returns:
            List of HaikuCheckResult objects
        """
        self.logger.info(f"Batch quick check for {len(contract_chunks_with_matches)} chunks")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_with_limit(chunk, matches):
            async with semaphore:
                return await self.quick_check(chunk, matches)

        # Execute all checks in parallel
        tasks = [
            check_with_limit(chunk, matches)
            for chunk, matches in contract_chunks_with_matches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Check failed for chunk {idx}: {result}")
                # Create error result
                chunk, matches = contract_chunks_with_matches[idx]
                error_result = HaikuCheckResult(
                    contract_chunk_id=chunk.chunk_id,
                    law_chunk_ids=[],
                    status=ComplianceStatus.ERROR,
                    is_compliant=False,
                    confidence=0.0,
                    explanation=f"Exception: {str(result)}",
                    evidence="",
                    potential_issues=[],
                    needs_deep_analysis=True
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        # Log statistics
        compliant_count = sum(1 for r in valid_results if r.is_compliant and r.confidence >= self.high_confidence_threshold)
        non_compliant_count = sum(1 for r in valid_results if not r.is_compliant and r.confidence >= self.high_confidence_threshold)
        uncertain_count = sum(1 for r in valid_results if r.needs_deep_analysis)

        self.logger.info(
            f"Batch quick check complete: "
            f"{compliant_count} compliant, "
            f"{non_compliant_count} non-compliant, "
            f"{uncertain_count} need deep analysis "
            f"({(uncertain_count/len(valid_results))*100:.1f}% escalation rate)"
        )

        return valid_results
