"""
Compliance Analyzer - Main orchestrator for compliance analysis.

This module provides automated clause-level compliance checking by comparing
contract provisions against legal requirements, detecting conflicts, identifying
gaps, and assessing legal risks.

Key capabilities:
- Clause-Level Checking: Verify each contract clause against applicable laws
- Conflict Detection: Identify direct contradictions between contract and law
- Gap Analysis: Find missing mandatory requirements in contract
- Deviation Assessment: Evaluate acceptable vs. problematic differences
- Risk Scoring: Quantify legal risk with severity levels (CRITICAL/HIGH/MEDIUM/LOW)
- Compliance Reporting: Generate structured reports with recommendations
"""

import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from anthropic import Anthropic

from .models import (
    ComplianceReport,
    ComplianceIssue,
    ComplianceMode,
    SeverityLevel
)

# Import compliance helper modules from src/
# backend/app/rag/compliance_analyzer.py -> ../../../src = /app/src (in Docker)
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.requirement_extractor import RequirementExtractor
from src.clause_mapper import ContractClauseMapper
from src.compliance_checkers import ComplianceChecker
from src.risk_scorer import RiskScorer
from src.recommendation_generator import RecommendationGenerator
from src.compliance_reporter import ComplianceReporter


class ComplianceAnalyzer:
    """Main orchestrator for compliance analysis."""

    def __init__(
        self,
        config: Dict[str, Any],
        cross_doc_retriever: Optional[Any] = None,
        llm_client: Optional[Anthropic] = None
    ):
        """
        Initialize ComplianceAnalyzer.

        Args:
            config: Configuration dictionary
            cross_doc_retriever: ComparativeRetriever instance for cross-document search
            llm_client: Anthropic client for LLM operations
        """
        self.config = config
        self.llm = llm_client or self._create_llm_client()

        # Initialize components
        self.requirement_extractor = RequirementExtractor(
            llm_client=self.llm,
            config=config.get('compliance', {})
        )
        self.clause_mapper = ContractClauseMapper(
            cross_doc_retriever=cross_doc_retriever,
            config=config
        )
        self.compliance_checker = ComplianceChecker(llm_client=self.llm)
        self.risk_scorer = RiskScorer(config=config.get('compliance', {}))
        self.recommendation_generator = RecommendationGenerator(
            llm_client=self.llm,
            config=config.get('compliance', {})
        )
        self.reporter = ComplianceReporter(config=config.get('compliance', {}))

        self.logger = logging.getLogger(__name__)
        self.llm_calls_count = 0

    def _create_llm_client(self) -> Anthropic:
        """Create Anthropic client from environment variables."""
        import os
        api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable must be set")
        return Anthropic(api_key=api_key)

    async def analyze_compliance(
        self,
        contract_chunks: List[Any],
        law_chunks: List[Any],
        contract_id: str,
        law_ids: List[str],
        mode: str = "exhaustive"
    ) -> ComplianceReport:
        """
        Perform complete compliance analysis.

        Args:
            contract_chunks: Chunks from contract document
            law_chunks: Chunks from law documents
            contract_id: Contract document ID
            law_ids: Law document IDs
            mode: "exhaustive" (check all) or "sample" (check subset)

        Returns:
            ComplianceReport object
        """
        start_time = time.time()
        self.llm_calls_count = 0

        self.logger.info(f"Starting compliance analysis (mode: {mode})")
        self.logger.info(f"Contract chunks: {len(contract_chunks)}, Law chunks: {len(law_chunks)}")

        # Step 1: Extract requirements from law
        self.logger.info("Step 1/6: Extracting legal requirements...")
        requirements = await self.requirement_extractor.extract_requirements(law_chunks)
        self.llm_calls_count += len(law_chunks)  # Approximate
        self.logger.info(f"Extracted {len(requirements)} requirements")

        # Step 2: Map contract clauses to law requirements
        self.logger.info("Step 2/6: Mapping contract clauses to law requirements...")
        mappings = await self.clause_mapper.map_clauses(contract_chunks, requirements)
        self.logger.info(f"Created {len(mappings)} clause-to-law mappings")

        # Step 3: Check compliance (detect conflicts, gaps, deviations)
        self.logger.info("Step 3/6: Checking compliance (conflicts, gaps, deviations)...")
        issues = await self.compliance_checker.check_compliance(mappings, requirements)
        self.llm_calls_count += len(mappings) * 2  # Conflict + deviation checks
        self.logger.info(f"Detected {len(issues)} compliance issues")

        # Step 4: Score risks
        self.logger.info("Step 4/6: Scoring risks and assigning severity levels...")
        issues = self.risk_scorer.score_issues(issues)
        self.logger.info("Risk scoring complete")

        # Step 5: Generate recommendations
        if self.config.get('compliance', {}).get('generate_recommendations', True):
            self.logger.info("Step 5/6: Generating remediation recommendations...")
            issues = await self.recommendation_generator.generate_recommendations(issues)
            self.llm_calls_count += len(issues)
            self.logger.info("Recommendations generated")
        else:
            self.logger.info("Step 5/6: Skipping recommendations (disabled in config)")

        # Step 6: Generate report
        self.logger.info("Step 6/6: Generating compliance report...")
        processing_time = time.time() - start_time
        report = self.reporter.generate_report(
            contract_id=contract_id,
            law_ids=law_ids,
            mappings=mappings,
            issues=issues,
            requirements=requirements,
            processing_time=processing_time,
            llm_calls=self.llm_calls_count,
            mode=mode
        )

        # Log summary
        self.logger.info(f"Compliance analysis complete in {processing_time:.1f}s")
        self.logger.info(f"Overall compliance score: {report.overall_compliance_score:.1%}")
        self.logger.info(f"Risk level: {report.risk_level}")
        self.logger.info(f"Found {report.total_issues} issues: "
                        f"{len(report.critical_issues)} critical, "
                        f"{len(report.high_issues)} high, "
                        f"{len(report.medium_issues)} medium, "
                        f"{len(report.low_issues)} low")

        return report

    def export_report(
        self,
        report: ComplianceReport,
        output_path: str,
        format: str = "json"
    ):
        """
        Export compliance report to file.

        Args:
            report: ComplianceReport object
            output_path: Path to output file
            format: Output format ('json' or 'markdown')
        """
        if format == "json":
            self.reporter.export_json(report, output_path)
        elif format == "markdown" or format == "md":
            self.reporter.export_markdown(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'markdown'")

        self.logger.info(f"Report exported to {output_path}")


# Convenience functions for standalone usage

async def analyze_contract_compliance(
    contract_chunks: List[Any],
    law_chunks: List[Any],
    contract_id: str,
    law_ids: List[str],
    config: Optional[Dict[str, Any]] = None,
    cross_doc_retriever: Optional[Any] = None,
    llm_client: Optional[Anthropic] = None,
    mode: str = "exhaustive"
) -> ComplianceReport:
    """
    Convenience function to perform compliance analysis.

    Args:
        contract_chunks: List of contract chunks
        law_chunks: List of law chunks
        contract_id: Contract document ID
        law_ids: List of law document IDs
        config: Configuration dictionary (optional)
        cross_doc_retriever: Cross-document retriever (optional)
        llm_client: Anthropic client (optional, will create from env if not provided)
        mode: Analysis mode ('exhaustive' or 'sample')

    Returns:
        ComplianceReport object

    Example:
        >>> report = await analyze_contract_compliance(
        ...     contract_chunks=contract_chunks,
        ...     law_chunks=law_chunks,
        ...     contract_id="contract_123",
        ...     law_ids=["law_89_2012"]
        ... )
        >>> print(f"Compliance Score: {report.overall_compliance_score:.1%}")
        >>> print(f"Critical Issues: {len(report.critical_issues)}")
    """
    config = config or {}
    analyzer = ComplianceAnalyzer(
        config=config,
        cross_doc_retriever=cross_doc_retriever,
        llm_client=llm_client
    )
    return await analyzer.analyze_compliance(
        contract_chunks=contract_chunks,
        law_chunks=law_chunks,
        contract_id=contract_id,
        law_ids=law_ids,
        mode=mode
    )
