"""
Sonnet Deep Analyzer for thorough compliance analysis.

Used for edge cases escalated from Haiku quick filter:
- Low confidence results (< 0.60)
- Uncertain status
- Complex legal reasoning required

Provides:
- Detailed legal analysis
- Comprehensive evidence extraction
- Specific recommendations
- Multi-factor risk scoring
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic

from .haiku_quick_filter import ComplianceStatus

logger = logging.getLogger(__name__)


@dataclass
class SonnetAnalysisResult:
    """Result from Sonnet deep analysis"""
    contract_chunk_id: str
    law_chunk_ids: List[str]

    # Detailed compliance assessment
    status: ComplianceStatus
    is_compliant: bool
    confidence: float  # 0.0-1.0

    # Detailed analysis
    detailed_explanation: str
    legal_reasoning: str
    evidence_from_contract: str
    evidence_from_law: str

    # Issue details
    issue_type: str  # "conflict", "deviation", "gap", "compliant"
    severity: str    # "critical", "high", "medium", "low"
    specific_violations: List[str]

    # Recommendations
    recommendations: List[str]
    suggested_contract_changes: List[str]

    # Risk factors
    risk_factors: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0

    # Metadata
    llm_model: str = "claude-sonnet-4-5-20250929"
    processing_time_ms: float = 0.0
    tokens_used: int = 0


class SonnetDeepAnalyzer:
    """
    Deep compliance analysis using Claude Sonnet for complex cases.

    Used for ~20% of chunks that Haiku marked as uncertain or low-confidence.
    Provides thorough legal analysis with detailed reasoning.
    """

    def __init__(
        self,
        llm_client: AsyncAnthropic,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Sonnet deep analyzer.

        Args:
            llm_client: AsyncAnthropic client
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.config = config or {}

        # Model configuration
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = self.config.get('max_tokens', 3000)
        self.temperature = self.config.get('temperature', 0.1)

        # Retry configuration
        self.max_retries = self.config.get('max_retries', 2)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        self.logger = logging.getLogger(__name__)

    async def deep_analyze(
        self,
        contract_chunk: Any,
        law_matches: List[Any],
        haiku_result: Optional[Any] = None
    ) -> SonnetAnalysisResult:
        """
        Perform deep compliance analysis.

        Args:
            contract_chunk: LegalChunk from contract
            law_matches: List of matched law chunks
            haiku_result: Optional HaikuCheckResult (provides context)

        Returns:
            SonnetAnalysisResult with detailed analysis
        """
        import time
        start_time = time.time()

        # Build comprehensive prompt
        prompt = self._build_deep_analysis_prompt(
            contract_chunk,
            law_matches,
            haiku_result
        )

        # Call Sonnet with retry logic
        try:
            response, tokens_used = await self._call_sonnet_with_retry(prompt)
            result = self._parse_detailed_response(
                response,
                contract_chunk,
                law_matches
            )
            result.tokens_used = tokens_used
        except Exception as e:
            self.logger.error(f"Sonnet analysis failed for chunk {contract_chunk.chunk_id}: {e}")
            result = self._create_error_result(contract_chunk, law_matches, str(e))

        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _build_deep_analysis_prompt(
        self,
        contract_chunk: Any,
        law_matches: List[Any],
        haiku_result: Optional[Any] = None
    ) -> str:
        """
        Build comprehensive prompt for Sonnet deep analysis.

        Includes full context, Haiku's preliminary assessment,
        and detailed instructions for thorough analysis.
        """
        # Extract contract details
        contract_ref = getattr(contract_chunk, 'legal_reference', 'N/A')
        contract_text = contract_chunk.content

        # Extract law chunks (full text for deep analysis)
        law_texts = []
        for i, match in enumerate(law_matches[:5], 1):
            law_ref = getattr(match.target_chunk, 'legal_reference', f'Law {i}')
            law_text = match.target_chunk.content
            match_score = getattr(match, 'overall_score', 0.0)
            law_texts.append(f"### {law_ref} (relevance: {match_score:.2f})\n{law_text}")

        laws_combined = "\n\n".join(law_texts)

        # Add Haiku context if available
        haiku_context = ""
        if haiku_result:
            haiku_context = f"""
<předběžné_hodnocení>
Haiku rychlá kontrola označila tento chunk jako: {haiku_result.status.value}
Confidence: {haiku_result.confidence:.2f}
Vysvětlení: {haiku_result.explanation}
Potenciální problémy: {', '.join(haiku_result.potential_issues) if haiku_result.potential_issues else 'žádné'}
</předběžné_hodnocení>
"""

        # Build prompt
        prompt = f"""Jsi expert na český právní systém specializující se na compliance analýzu smluv.

Proveď **důkladnou a detailní analýzu** souladu klauzule smlouvy se zákonnými požadavky.

<klauzule_smlouvy ref="{contract_ref}">
{contract_text}
</klauzule_smlouvy>

<zakonné_požadavky>
{laws_combined}
</zakonné_požadavky>

{haiku_context}

<instrukce_pro_analýzu>
Proveď komplexní právní analýzu:

1. **Detailní porovnání**:
   - Analyzuj každý relevantní zákonný požadavek zvlášť
   - Identifikuj přesné body souladu/nesouladu
   - Extrahuj konkrétní důkazy z textu

2. **Právní odůvodnění**:
   - Vysvětli právní interpretaci
   - Uveď relevantní právní principy
   - Zvaž kontextuální faktory

3. **Klasifikace problému**:
   - **Conflict**: Přímý rozpor (smlouva říká X, zákon vyžaduje Y)
   - **Deviation**: Odchylka od zákona (může být přípustná/nepřípustná)
   - **Gap**: Chybějící povinný prvek
   - **Compliant**: V souladu

4. **Hodnocení závažnosti**:
   - **Critical**: Porušení povinného zákona, vysoké riziko
   - **High**: Významný problém, střední riziko
   - **Medium**: Menší problém, nízké riziko
   - **Low**: Drobná nepřesnost, minimální riziko

5. **Praktická doporučení**:
   - Konkrétní změny textu smlouvy
   - Alternativní formulace
   - Právní odkazy k doplnění
</instrukce_pro_analýzu>

<output_format>
Vrať strukturovaný JSON:
{{
  "is_compliant": true/false,
  "confidence": 0.0-1.0,
  "status": "compliant" | "non_compliant" | "uncertain",

  "detailed_explanation": "Podrobné vysvětlení (3-5 vět)",
  "legal_reasoning": "Právní odůvodnění s odkazy na paragrafy",
  "evidence_from_contract": "Konkrétní citace ze smlouvy",
  "evidence_from_law": "Konkrétní citace ze zákona",

  "issue_type": "conflict" | "deviation" | "gap" | "compliant",
  "severity": "critical" | "high" | "medium" | "low",
  "specific_violations": ["Porušení 1", "Porušení 2"],

  "recommendations": [
    "Doporučení 1: konkrétní akce",
    "Doporučení 2: konkrétní akce",
    "Doporučení 3: konkrétní akce"
  ],
  "suggested_contract_changes": [
    "Změna 1: 'Původní text' → 'Nový text'",
    "Změna 2: 'Původní text' → 'Nový text'"
  ],

  "risk_factors": {{
    "mandatory_violation": 0.0-1.0,
    "temporal_constraint": 0.0-1.0,
    "financial_impact": 0.0-1.0,
    "enforcement_likelihood": 0.0-1.0
  }}
}}

DŮLEŽITÉ: Buď precizní a konkrétní. Uveď přesné citace a paragrafy.
</output_format>

JSON:"""

        return prompt

    async def _call_sonnet_with_retry(self, prompt: str) -> tuple[str, int]:
        """Call Sonnet with exponential backoff retry"""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.llm.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text.strip()
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

                return text, tokens_used

            except Exception as e:
                if attempt == self.max_retries:
                    raise

                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Sonnet call failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    def _parse_detailed_response(
        self,
        response_text: str,
        contract_chunk: Any,
        law_matches: List[Any]
    ) -> SonnetAnalysisResult:
        """Parse Sonnet JSON response into detailed result"""
        try:
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response_text)

            # Map status
            status_str = parsed.get('status', 'uncertain')
            status_map = {
                'compliant': ComplianceStatus.COMPLIANT,
                'non_compliant': ComplianceStatus.NON_COMPLIANT,
                'uncertain': ComplianceStatus.UNCERTAIN
            }
            status = status_map.get(status_str, ComplianceStatus.UNCERTAIN)

            # Calculate risk score from factors
            risk_factors = parsed.get('risk_factors', {})
            risk_score = sum(risk_factors.values()) / len(risk_factors) if risk_factors else 0.5

            result = SonnetAnalysisResult(
                contract_chunk_id=contract_chunk.chunk_id,
                law_chunk_ids=[m.target_chunk.chunk_id for m in law_matches],
                status=status,
                is_compliant=parsed.get('is_compliant', False),
                confidence=float(parsed.get('confidence', 0.7)),
                detailed_explanation=parsed.get('detailed_explanation', ''),
                legal_reasoning=parsed.get('legal_reasoning', ''),
                evidence_from_contract=parsed.get('evidence_from_contract', ''),
                evidence_from_law=parsed.get('evidence_from_law', ''),
                issue_type=parsed.get('issue_type', 'compliant'),
                severity=parsed.get('severity', 'medium'),
                specific_violations=parsed.get('specific_violations', []),
                recommendations=parsed.get('recommendations', []),
                suggested_contract_changes=parsed.get('suggested_contract_changes', []),
                risk_factors=risk_factors,
                risk_score=risk_score
            )

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse Sonnet response: {e}")
            return self._create_error_result(
                contract_chunk,
                law_matches,
                f"Parse error: {e}"
            )

    def _create_error_result(
        self,
        contract_chunk: Any,
        law_matches: List[Any],
        error_msg: str
    ) -> SonnetAnalysisResult:
        """Create error result when analysis fails"""
        return SonnetAnalysisResult(
            contract_chunk_id=contract_chunk.chunk_id,
            law_chunk_ids=[m.target_chunk.chunk_id for m in law_matches],
            status=ComplianceStatus.ERROR,
            is_compliant=False,
            confidence=0.0,
            detailed_explanation=f"Analysis failed: {error_msg}",
            legal_reasoning="Error during analysis",
            evidence_from_contract="",
            evidence_from_law="",
            issue_type="error",
            severity="high",
            specific_violations=[error_msg],
            recommendations=["Manual review required"],
            suggested_contract_changes=[],
            risk_factors={},
            risk_score=0.8  # High risk for errors
        )

    async def batch_deep_analyze(
        self,
        chunks_with_matches: List[tuple[Any, List[Any], Optional[Any]]],
        max_concurrent: int = 3
    ) -> List[SonnetAnalysisResult]:
        """
        Batch process multiple chunks with concurrency limit.

        Args:
            chunks_with_matches: List of (contract_chunk, law_matches, haiku_result) tuples
            max_concurrent: Maximum concurrent Sonnet calls (lower due to cost)

        Returns:
            List of SonnetAnalysisResult objects
        """
        self.logger.info(f"Batch deep analysis for {len(chunks_with_matches)} chunks")

        # Create semaphore (lower concurrency for Sonnet)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(chunk, matches, haiku_res):
            async with semaphore:
                return await self.deep_analyze(chunk, matches, haiku_res)

        # Execute with concurrency limit
        tasks = [
            analyze_with_limit(chunk, matches, haiku_res)
            for chunk, matches, haiku_res in chunks_with_matches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Deep analysis failed for chunk {idx}: {result}")
                chunk, matches, _ = chunks_with_matches[idx]
                error_result = self._create_error_result(chunk, matches, str(result))
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        self.logger.info(f"Batch deep analysis complete: {len(valid_results)}/{len(chunks_with_matches)} successful")

        return valid_results
