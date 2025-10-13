"""
Advanced Compliance Pipeline - Main Orchestrator.

Implements Approach 2: Clean Architecture with industry best practices:
- Pre-filtering (skip irrelevant chunks)
- Multi-round retrieval (iterative query refinement)
- Chain of verification (Haiku quick filter → Sonnet deep analysis)
- Confidence-based filtering

Based on research showing 79% recall improvement and significant cost optimization.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic

from .chunk_pre_filter import ChunkPreFilter, FilterMetrics
from .multi_round_retriever import MultiRoundRetriever, MultiRoundResult
from .haiku_quick_filter import HaikuQuickFilter, HaikuCheckResult, ComplianceStatus
from .sonnet_deep_analyzer import SonnetDeepAnalyzer, SonnetAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ComplianceIssue:
    """Unified compliance issue from either Haiku or Sonnet"""
    issue_id: str
    contract_chunk_id: str
    contract_reference: str
    contract_text_preview: str

    # Core assessment
    is_compliant: bool
    confidence: float
    status: str  # "compliant", "non_compliant", "uncertain"

    # Issue details
    issue_type: str  # "conflict", "deviation", "gap", "compliant"
    severity: str    # "critical", "high", "medium", "low"
    explanation: str
    evidence: str

    # Law references
    law_chunk_ids: List[str]
    law_references: List[str]

    # Recommendations
    recommendations: List[str]

    # Metadata
    analyzed_by: str  # "haiku" | "sonnet"
    analysis_time_ms: float = 0.0


@dataclass
class ComplianceReport:
    """Final compliance report"""
    # Metadata
    report_id: str
    contract_id: str
    law_ids: List[str]

    # Issues
    all_issues: List[ComplianceIssue]
    critical_issues: List[ComplianceIssue]
    high_issues: List[ComplianceIssue]
    medium_issues: List[ComplianceIssue]
    low_issues: List[ComplianceIssue]

    # Statistics
    total_chunks_analyzed: int
    pre_filter_reduction: float
    haiku_analyzed: int
    sonnet_analyzed: int
    escalation_rate: float

    # Scores
    overall_compliance_score: float
    risk_level: str

    # Performance
    processing_time_seconds: float
    total_llm_calls: int
    estimated_cost_usd: float

    # Metrics
    filter_metrics: Optional[FilterMetrics] = None


class AdvancedCompliancePipeline:
    """
    Advanced compliance pipeline orchestrator.

    Implements complete chain of verification with best practices.
    """

    def __init__(
        self,
        llm_client: AsyncAnthropic,
        comparative_retriever: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline with all components"""
        self.llm = llm_client
        self.config = config or {}

        # Initialize components
        self.pre_filter = ChunkPreFilter(config.get('pre_filter', {}))
        self.multi_round_retriever = MultiRoundRetriever(
            comparative_retriever,
            config.get('multi_round', {})
        )
        self.haiku_filter = HaikuQuickFilter(
            llm_client,
            config.get('haiku', {})
        )
        self.sonnet_analyzer = SonnetDeepAnalyzer(
            llm_client,
            config.get('sonnet', {})
        )

        self.logger = logging.getLogger(__name__)

    async def analyze_compliance(
        self,
        contract_chunks: List[Any],
        law_ids: List[str],
        contract_id: str
    ) -> ComplianceReport:
        """
        Run complete compliance analysis pipeline against multiple laws.

        Pipeline stages:
        1. Pre-filter contract chunks (~30% reduction)
        2. Multi-round retrieval for each chunk against ALL laws (2 rounds per law)
        3. Haiku quick check (fast, cheap)
        4. Sonnet deep analysis (for escalated cases only, ~20%)
        5. Aggregate and build report

        Args:
            contract_chunks: List of contract chunks to analyze
            law_ids: List of law document IDs to check against
            contract_id: Contract document ID

        Returns:
            ComplianceReport with issues found across all laws
        """
        start_time = time.time()

        self.logger.info(f"Starting advanced compliance pipeline for {len(contract_chunks)} chunks against {len(law_ids)} laws")

        # Stage 1: Pre-filter
        self.logger.info("Stage 1/5: Pre-filtering chunks...")
        filtered_chunks, filter_metrics = self.pre_filter.filter_chunks(
            contract_chunks,
            strategy="balanced"  # 30% reduction
        )
        self.logger.info(f"Pre-filter: {len(filtered_chunks)}/{len(contract_chunks)} chunks kept")

        # Stage 2: Multi-round retrieval against ALL laws
        self.logger.info(f"Stage 2/5: Multi-round retrieval for {len(filtered_chunks)} chunks against {len(law_ids)} laws...")

        # For each contract chunk, retrieve from all laws and merge results
        all_retrieval_results = []
        for law_id in law_ids:
            self.logger.info(f"  Retrieving from law: {law_id}")
            law_retrieval_results = await self.multi_round_retriever.batch_retrieve_multi_round(
                filtered_chunks,
                law_id,
                max_concurrent=5
            )
            all_retrieval_results.append((law_id, law_retrieval_results))
            self.logger.info(f"  Retrieved {len(law_retrieval_results)} results from {law_id}")

        # Merge all law matches per contract chunk
        # For each contract chunk, combine top matches from all laws
        merged_retrieval_results = self._merge_multi_law_retrievals(all_retrieval_results, top_k=5)
        self.logger.info(f"Multi-round retrieval complete: {len(merged_retrieval_results)} merged results")

        # Stage 3: Haiku quick check
        self.logger.info(f"Stage 3/5: Haiku quick screening...")
        chunks_with_matches = [
            (r.contract_chunk, r.final_matches)
            for r in merged_retrieval_results
        ]
        haiku_results = await self.haiku_filter.batch_quick_check(
            chunks_with_matches,
            max_concurrent=10
        )
        self.logger.info(f"Haiku screening complete: {len(haiku_results)} checks")

        # Stage 4: Sonnet deep analysis (escalated cases only)
        escalated = [
            (merged_retrieval_results[i].contract_chunk, merged_retrieval_results[i].final_matches, haiku_results[i])
            for i in range(len(haiku_results))
            if haiku_results[i].needs_deep_analysis
        ]
        self.logger.info(f"Stage 4/5: Sonnet deep analysis for {len(escalated)} escalated cases...")
        sonnet_results = []
        if escalated:
            sonnet_results = await self.sonnet_analyzer.batch_deep_analyze(
                escalated,
                max_concurrent=3
            )
        self.logger.info(f"Sonnet analysis complete: {len(sonnet_results)} deep analyses")

        # Stage 5: Aggregate and build report
        self.logger.info("Stage 5/5: Building compliance report...")
        report = self._build_report(
            contract_id=contract_id,
            law_ids=law_ids,
            original_chunk_count=len(contract_chunks),
            filtered_chunks=filtered_chunks,
            filter_metrics=filter_metrics,
            haiku_results=haiku_results,
            sonnet_results=sonnet_results,
            processing_time=time.time() - start_time
        )

        self.logger.info(f"Compliance analysis complete in {report.processing_time_seconds:.1f}s")
        return report

    def _merge_multi_law_retrievals(
        self,
        all_retrieval_results: List[tuple[str, List[MultiRoundResult]]],
        top_k: int = 5
    ) -> List[MultiRoundResult]:
        """
        Merge retrieval results from multiple laws for each contract chunk.

        For each contract chunk, combines and re-ranks matches from all laws,
        keeping the top-K most relevant law provisions across all laws.

        Args:
            all_retrieval_results: List of (law_id, retrieval_results) tuples
            top_k: Maximum number of law matches to keep per contract chunk

        Returns:
            List of merged MultiRoundResult objects
        """
        if not all_retrieval_results:
            return []

        # Get number of contract chunks (should be same across all laws)
        num_chunks = len(all_retrieval_results[0][1]) if all_retrieval_results else 0

        merged_results = []

        # For each contract chunk index
        for chunk_idx in range(num_chunks):
            # Collect all law matches for this chunk from all laws
            all_matches = []
            contract_chunk = None

            for law_id, law_results in all_retrieval_results:
                if chunk_idx < len(law_results):
                    result = law_results[chunk_idx]
                    if contract_chunk is None:
                        contract_chunk = result.contract_chunk
                    # Add all matches from this law
                    all_matches.extend(result.final_matches)

            # Sort by overall_score and take top-K
            all_matches.sort(key=lambda m: m.overall_score, reverse=True)
            top_matches = all_matches[:top_k]

            # Create merged result
            if contract_chunk is not None:
                merged_result = MultiRoundResult(
                    contract_chunk=contract_chunk,
                    rounds=[],  # Rounds not relevant after merging
                    final_matches=top_matches,
                    total_time_ms=0.0
                )
                merged_results.append(merged_result)

        return merged_results

    async def analyze_compliance_stream(
        self,
        contract_chunks: List[Any],
        law_ids: List[str],
        contract_id: str,
        language: str = "cs"
    ) -> AsyncIterator[str]:
        """
        Streaming version of compliance analysis with real-time progress updates.
        Supports multiple laws.
        """
        import json

        # Stage 1: Pre-filter
        yield self._status_message(language, "pre_filtering", 1, 5, 20)
        filtered_chunks, filter_metrics = self.pre_filter.filter_chunks(
            contract_chunks,
            strategy="balanced"
        )
        if language == "cs":
            yield f"✓ Pre-filter: {len(filtered_chunks)}/{len(contract_chunks)} chunků ponecháno\n"
        else:
            yield f"✓ Pre-filter: {len(filtered_chunks)}/{len(contract_chunks)} chunks kept\n"

        # Stage 2: Multi-round retrieval against ALL laws (batch with progress updates)
        yield self._status_message(language, "multi_round_retrieval", 2, 5, 40)

        all_retrieval_results = []
        for law_idx, law_id in enumerate(law_ids, 1):
            if language == "cs":
                yield f"🔍 Vyhledávám v zákonu {law_idx}/{len(law_ids)}: {law_id}\n"
            else:
                yield f"🔍 Searching law {law_idx}/{len(law_ids)}: {law_id}\n"

            law_retrieval_results = []
            batch_size = 10
            for batch_idx in range(0, len(filtered_chunks), batch_size):
                batch = filtered_chunks[batch_idx:batch_idx+batch_size]
                batch_results = await self.multi_round_retriever.batch_retrieve_multi_round(
                    batch,
                    law_id,
                    max_concurrent=5
                )
                law_retrieval_results.extend(batch_results)
                if language == "cs":
                    yield f"  ✓ {len(law_retrieval_results)}/{len(filtered_chunks)} chunků\n"
                else:
                    yield f"  ✓ {len(law_retrieval_results)}/{len(filtered_chunks)} chunks\n"

            all_retrieval_results.append((law_id, law_retrieval_results))

        # Merge results from all laws
        if language == "cs":
            yield f"🔗 Slučuji výsledky z {len(law_ids)} zákonů...\n"
        else:
            yield f"🔗 Merging results from {len(law_ids)} laws...\n"
        retrieval_results = self._merge_multi_law_retrievals(all_retrieval_results, top_k=5)

        # Stage 3: Haiku quick check (batch with progress)
        yield self._status_message(language, "haiku_screening", 3, 5, 60)
        chunks_with_matches = [
            (r.contract_chunk, r.final_matches)
            for r in retrieval_results
        ]
        haiku_results = []
        for batch_idx in range(0, len(chunks_with_matches), 10):
            batch = chunks_with_matches[batch_idx:batch_idx+10]
            batch_results = await self.haiku_filter.batch_quick_check(batch, max_concurrent=10)
            haiku_results.extend(batch_results)
            yield f"✓ Haiku checked {len(haiku_results)}/{len(chunks_with_matches)} chunks\n"

        # Stage 4: Sonnet deep analysis
        escalated = [
            (retrieval_results[i].contract_chunk, retrieval_results[i].final_matches, haiku_results[i])
            for i in range(len(haiku_results))
            if haiku_results[i].needs_deep_analysis
        ]
        if escalated:
            yield self._status_message(language, "sonnet_deep_analysis", 4, 5, 80)
            yield f"🔍 Deep analyzing {len(escalated)} complex cases...\n"
            sonnet_results = await self.sonnet_analyzer.batch_deep_analyze(
                escalated,
                max_concurrent=3
            )
        else:
            sonnet_results = []

        # Stage 5: Build and stream report
        yield self._status_message(language, "building_report", 5, 5, 95)
        all_issues = self._extract_issues(haiku_results, sonnet_results)

        # Stream summary
        yield "\n## 📊 Compliance Report\n\n"
        yield f"**Analyzed**: {len(filtered_chunks)}/{len(contract_chunks)} chunks\n"
        yield f"**Issues found**: {len(all_issues)}\n\n"

        # Stream issues by severity
        critical = [i for i in all_issues if i.severity == "critical"]
        high = [i for i in all_issues if i.severity == "high"]

        if critical:
            yield f"\n### 🔴 Critical Issues ({len(critical)})\n\n"
            for i, issue in enumerate(critical[:5], 1):
                yield f"{i}. **{issue.contract_reference}**: {issue.explanation[:150]}...\n"
                yield f"   💡 {issue.recommendations[0] if issue.recommendations else 'Manual review required'}\n\n"

        if high:
            yield f"\n### 🟠 High Priority ({len(high)})\n\n"
            for i, issue in enumerate(high[:3], 1):
                yield f"{i}. **{issue.contract_reference}**: {issue.explanation[:100]}...\n\n"

        yield self._status_message(language, "complete", 5, 5, 100)

    def _build_report(
        self,
        contract_id: str,
        law_ids: List[str],
        original_chunk_count: int,
        filtered_chunks: List[Any],
        filter_metrics: FilterMetrics,
        haiku_results: List[HaikuCheckResult],
        sonnet_results: List[SonnetAnalysisResult],
        processing_time: float
    ) -> ComplianceReport:
        """Build final compliance report from all results"""
        # Extract issues
        all_issues = self._extract_issues(haiku_results, sonnet_results)

        # Categorize by severity
        critical = [i for i in all_issues if i.severity == "critical"]
        high = [i for i in all_issues if i.severity == "high"]
        medium = [i for i in all_issues if i.severity == "medium"]
        low = [i for i in all_issues if i.severity == "low"]

        # Calculate compliance score
        total_checked = len(filtered_chunks)
        compliance_score = 1.0 - (
            (len(critical) * 0.4 + len(high) * 0.3 + len(medium) * 0.2 + len(low) * 0.1)
            / max(total_checked, 1)
        )

        # Determine risk level
        if critical:
            risk_level = "critical"
        elif len(high) >= 3:
            risk_level = "high"
        elif len(high) > 0 or len(medium) >= 5:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Calculate costs (approximate)
        haiku_cost = len(haiku_results) * 0.002  # ~$0.002 per call
        sonnet_cost = len(sonnet_results) * 0.02  # ~$0.02 per call

        return ComplianceReport(
            report_id=f"compliance_{int(time.time())}",
            contract_id=contract_id,
            law_ids=law_ids,
            all_issues=all_issues,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            total_chunks_analyzed=len(filtered_chunks),
            pre_filter_reduction=filter_metrics.reduction_percentage,
            haiku_analyzed=len(haiku_results),
            sonnet_analyzed=len(sonnet_results),
            escalation_rate=(len(sonnet_results) / max(len(haiku_results), 1)) * 100,
            overall_compliance_score=compliance_score,
            risk_level=risk_level,
            processing_time_seconds=processing_time,
            total_llm_calls=len(haiku_results) + len(sonnet_results),
            estimated_cost_usd=haiku_cost + sonnet_cost,
            filter_metrics=filter_metrics
        )

    def _extract_issues(
        self,
        haiku_results: List[HaikuCheckResult],
        sonnet_results: List[SonnetAnalysisResult]
    ) -> List[ComplianceIssue]:
        """Extract compliance issues from both Haiku and Sonnet results"""
        issues = []

        # Extract from Haiku (non-compliant high-confidence)
        for h in haiku_results:
            if not h.is_compliant and h.confidence >= 0.85 and not h.needs_deep_analysis:
                issues.append(ComplianceIssue(
                    issue_id=f"haiku_{h.contract_chunk_id}",
                    contract_chunk_id=h.contract_chunk_id,
                    contract_reference=h.contract_chunk_id,  # Would need to extract
                    contract_text_preview=h.contract_text_preview,
                    is_compliant=False,
                    confidence=h.confidence,
                    status=h.status.value,
                    issue_type="non_compliant",
                    severity="medium",  # Haiku defaults to medium
                    explanation=h.explanation,
                    evidence=h.evidence,
                    law_chunk_ids=h.law_chunk_ids,
                    law_references=[],
                    recommendations=h.potential_issues,
                    analyzed_by="haiku",
                    analysis_time_ms=h.processing_time_ms
                ))

        # Extract from Sonnet (all non-compliant)
        for s in sonnet_results:
            if not s.is_compliant or s.severity in ["critical", "high"]:
                issues.append(ComplianceIssue(
                    issue_id=f"sonnet_{s.contract_chunk_id}",
                    contract_chunk_id=s.contract_chunk_id,
                    contract_reference=s.contract_chunk_id,
                    contract_text_preview="",  # Would need to extract
                    is_compliant=s.is_compliant,
                    confidence=s.confidence,
                    status=s.status.value,
                    issue_type=s.issue_type,
                    severity=s.severity,
                    explanation=s.detailed_explanation,
                    evidence=f"{s.evidence_from_contract}\n\n{s.evidence_from_law}",
                    law_chunk_ids=s.law_chunk_ids,
                    law_references=[],
                    recommendations=s.recommendations,
                    analyzed_by="sonnet",
                    analysis_time_ms=s.processing_time_ms
                ))

        return issues

    def _status_message(self, language: str, stage: str, step: int, total: int, progress: int) -> str:
        """Generate pipeline status message"""
        import json

        messages = {
            "cs": {
                "pre_filtering": "🔍 Filtruji chunky...",
                "multi_round_retrieval": "🔄 Multi-round retrieval...",
                "haiku_screening": "⚡ Haiku rychlá kontrola...",
                "sonnet_deep_analysis": "🧠 Sonnet hloubková analýza...",
                "building_report": "📊 Generuji report...",
                "complete": "✅ Hotovo!"
            },
            "en": {
                "pre_filtering": "🔍 Filtering chunks...",
                "multi_round_retrieval": "🔄 Multi-round retrieval...",
                "haiku_screening": "⚡ Haiku quick check...",
                "sonnet_deep_analysis": "🧠 Sonnet deep analysis...",
                "building_report": "📊 Building report...",
                "complete": "✅ Complete!"
            }
        }

        msg = messages.get(language, messages["cs"]).get(stage, stage)

        status = json.dumps({
            "type": "pipeline_status",
            "pipeline": "advanced_compliance",
            "stage": stage,
            "step": step,
            "total_steps": total,
            "progress": progress,
            "message": msg
        })

        return f"__STATUS__{status}__STATUS__\n"
