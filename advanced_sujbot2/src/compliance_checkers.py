"""
Compliance checkers for detecting conflicts, gaps, and deviations.

This module contains:
- ConflictDetector: Detects conflicts between contract and law
- GapAnalyzer: Identifies missing mandatory requirements
- DeviationAssessor: Assesses whether deviations are acceptable
- ComplianceChecker: Main orchestrator for all compliance checks
"""

import logging
import json
import asyncio
from typing import List, Optional
from anthropic import Anthropic

from .compliance_types import (
    ClauseMapping,
    ComplianceIssue,
    ComplianceStatus,
    IssueSeverity,
    LegalRequirement,
    RequirementType
)


class ConflictDetector:
    """Detect conflicts between contract and law."""

    def __init__(self, llm_client: Anthropic):
        """
        Initialize ConflictDetector.

        Args:
            llm_client: Anthropic client for LLM-based analysis
        """
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    async def detect_conflict(
        self,
        mapping: ClauseMapping
    ) -> Optional[ComplianceIssue]:
        """
        Check if contract clause conflicts with law requirements.

        Args:
            mapping: ClauseMapping object

        Returns:
            ComplianceIssue if conflict detected, else None
        """
        # Use LLM to compare clause vs. requirements
        prompt = self._build_conflict_detection_prompt(mapping)

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text.strip()

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse conflict detection response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in conflict detection: {e}")
            return None

        # If conflict detected, create issue
        if result.get("has_conflict"):
            issue = ComplianceIssue(
                issue_id=f"conflict_{mapping.contract_chunk_id}",
                status=ComplianceStatus.CONFLICT,
                severity=IssueSeverity.HIGH,  # Will be refined by RiskScorer
                contract_chunk_id=mapping.contract_chunk_id,
                contract_reference=mapping.contract_reference,
                contract_text=mapping.contract_text,
                law_requirements=mapping.law_requirements,
                issue_description=result.get("description", "Conflict detected"),
                evidence=result.get("evidence", ""),
                reasoning=result.get("reasoning", ""),
                risk_score=0.0,  # Will be computed by RiskScorer
                risk_factors={},
                recommendations=[],
                priority=1,
                detected_by="ConflictDetector",
                detection_confidence=result.get("confidence", 0.9)
            )
            return issue

        return None

    def _build_conflict_detection_prompt(self, mapping: ClauseMapping) -> str:
        """Build prompt for conflict detection."""
        requirements_text = "\n".join([
            f"- {req.requirement_summary} ({req.law_reference})"
            for req in mapping.law_requirements
        ])

        prompt = f"""Analyze whether this contract clause conflicts with legal requirements.

Contract clause ({mapping.contract_reference}):
{mapping.contract_text}

Legal requirements:
{requirements_text}

Does the contract clause directly contradict or violate any of these legal requirements?

Consider:
- Direct contradictions (e.g., contract allows X, law prohibits X)
- Missing mandatory elements
- Incompatible terms

Output JSON:
{{
  "has_conflict": true/false,
  "description": "Brief description of conflict",
  "evidence": "Specific text from clause causing conflict",
  "reasoning": "Why this is a conflict",
  "confidence": 0.0-1.0
}}

Analysis:"""

        return prompt


class GapAnalyzer:
    """Identify missing mandatory requirements in contract."""

    def __init__(self, llm_client: Anthropic):
        """
        Initialize GapAnalyzer.

        Args:
            llm_client: Anthropic client for LLM-based analysis
        """
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    async def analyze_gaps(
        self,
        all_requirements: List[LegalRequirement],
        all_mappings: List[ClauseMapping]
    ) -> List[ComplianceIssue]:
        """
        Find mandatory requirements not covered by contract.

        Args:
            all_requirements: All legal requirements
            all_mappings: All clause-to-law mappings

        Returns:
            List of ComplianceIssue for missing requirements
        """
        # Build set of requirements covered by contract
        covered_requirement_ids = set()
        for mapping in all_mappings:
            for req in mapping.law_requirements:
                covered_requirement_ids.add(req.requirement_id)

        # Find uncovered mandatory requirements
        missing_requirements = [
            req for req in all_requirements
            if req.is_mandatory and req.requirement_id not in covered_requirement_ids
        ]

        # Create issues for missing requirements
        gap_issues = []
        for req in missing_requirements:
            issue = ComplianceIssue(
                issue_id=f"gap_{req.requirement_id}",
                status=ComplianceStatus.MISSING,
                severity=IssueSeverity.HIGH,  # Will be refined
                contract_chunk_id="N/A",
                contract_reference="N/A",
                contract_text="[Missing clause]",
                law_requirements=[req],
                issue_description=f"Mandatory requirement missing from contract: {req.requirement_summary}",
                evidence="Requirement not found in any contract clause",
                reasoning=f"Law {req.law_reference} requires this, but contract does not address it",
                risk_score=0.0,
                risk_factors={},
                recommendations=[],
                priority=1,
                detected_by="GapAnalyzer",
                detection_confidence=0.85
            )
            gap_issues.append(issue)

        self.logger.info(f"Found {len(gap_issues)} gaps")
        return gap_issues


class DeviationAssessor:
    """Assess whether deviations from law are acceptable."""

    def __init__(self, llm_client: Anthropic):
        """
        Initialize DeviationAssessor.

        Args:
            llm_client: Anthropic client for LLM-based analysis
        """
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    async def assess_deviation(
        self,
        mapping: ClauseMapping
    ) -> Optional[ComplianceIssue]:
        """
        Check if clause deviates from law in potentially problematic way.

        Args:
            mapping: ClauseMapping object

        Returns:
            ComplianceIssue if deviation is problematic
        """
        # Use LLM to assess if deviation is acceptable
        prompt = self._build_deviation_prompt(mapping)

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse deviation assessment: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in deviation assessment: {e}")
            return None

        # If deviation is problematic, create issue
        if result.get("is_problematic"):
            issue = ComplianceIssue(
                issue_id=f"deviation_{mapping.contract_chunk_id}",
                status=ComplianceStatus.DEVIATION,
                severity=IssueSeverity.MEDIUM,
                contract_chunk_id=mapping.contract_chunk_id,
                contract_reference=mapping.contract_reference,
                contract_text=mapping.contract_text,
                law_requirements=mapping.law_requirements,
                issue_description=result.get("description", "Deviation detected"),
                evidence=result.get("evidence", ""),
                reasoning=result.get("reasoning", ""),
                risk_score=0.0,
                risk_factors={},
                recommendations=[],
                priority=2,
                detected_by="DeviationAssessor",
                detection_confidence=result.get("confidence", 0.8)
            )
            return issue

        return None

    def _build_deviation_prompt(self, mapping: ClauseMapping) -> str:
        """Build prompt for deviation assessment."""
        requirements_text = "\n".join([
            f"- {req.requirement_summary} ({req.law_reference})"
            for req in mapping.law_requirements
        ])

        prompt = f"""Analyze whether this contract clause deviates from legal requirements in a problematic way.

Contract clause ({mapping.contract_reference}):
{mapping.contract_text}

Legal requirements:
{requirements_text}

Consider:
- Does clause address the requirement but in a different way?
- Is the deviation acceptable or problematic?
- Does it weaken protections or add unreasonable terms?

Output JSON:
{{
  "is_problematic": true/false,
  "description": "Description of deviation",
  "evidence": "Specific text showing deviation",
  "reasoning": "Why this deviation is/isn't problematic",
  "confidence": 0.0-1.0
}}

Analysis:"""

        return prompt


class ComplianceChecker:
    """Orchestrate compliance checking."""

    def __init__(self, llm_client: Anthropic):
        """
        Initialize ComplianceChecker.

        Args:
            llm_client: Anthropic client for LLM-based checks
        """
        self.conflict_detector = ConflictDetector(llm_client)
        self.gap_analyzer = GapAnalyzer(llm_client)
        self.deviation_assessor = DeviationAssessor(llm_client)
        self.logger = logging.getLogger(__name__)

    async def check_compliance(
        self,
        mappings: List[ClauseMapping],
        all_requirements: List[LegalRequirement]
    ) -> List[ComplianceIssue]:
        """
        Perform complete compliance check.

        Args:
            mappings: List of clause-to-law mappings
            all_requirements: All legal requirements

        Returns:
            List of all ComplianceIssues found
        """
        all_issues = []

        # 1. Detect conflicts
        self.logger.info(f"Checking {len(mappings)} mappings for conflicts...")
        conflict_tasks = [
            self.conflict_detector.detect_conflict(mapping)
            for mapping in mappings
        ]
        conflict_results = await asyncio.gather(*conflict_tasks, return_exceptions=True)
        conflicts = [
            issue for issue in conflict_results
            if isinstance(issue, ComplianceIssue)
        ]
        all_issues.extend(conflicts)
        self.logger.info(f"Found {len(conflicts)} conflicts")

        # 2. Detect deviations
        self.logger.info(f"Checking {len(mappings)} mappings for deviations...")
        deviation_tasks = [
            self.deviation_assessor.assess_deviation(mapping)
            for mapping in mappings
        ]
        deviation_results = await asyncio.gather(*deviation_tasks, return_exceptions=True)
        deviations = [
            issue for issue in deviation_results
            if isinstance(issue, ComplianceIssue)
        ]
        all_issues.extend(deviations)
        self.logger.info(f"Found {len(deviations)} deviations")

        # 3. Find gaps
        self.logger.info("Analyzing gaps...")
        gaps = await self.gap_analyzer.analyze_gaps(all_requirements, mappings)
        all_issues.extend(gaps)
        self.logger.info(f"Found {len(gaps)} gaps")

        return all_issues
