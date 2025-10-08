# 09. Compliance Analyzer Specification

## 1. Purpose

**Objective**: Perform automated clause-level compliance checking by comparing contract provisions against legal requirements, detecting conflicts, identifying gaps, and assessing legal risks.

**Why Compliance Analyzer?**
- Manual compliance review of 1000+ page contracts is time-consuming and error-prone
- Legal requirements are scattered across multiple statutes and regulations
- Need systematic approach to detect all deviations, not just obvious conflicts
- Risk assessment requires consistent methodology
- Actionable insights need structured reporting

**Key Capabilities**:
1. **Clause-Level Checking** - Verify each contract clause against applicable laws
2. **Conflict Detection** - Identify direct contradictions between contract and law
3. **Gap Analysis** - Find missing mandatory requirements in contract
4. **Deviation Assessment** - Evaluate acceptable vs. problematic differences
5. **Risk Scoring** - Quantify legal risk with severity levels (CRITICAL/HIGH/MEDIUM/LOW)
6. **Compliance Reporting** - Generate structured reports with recommendations

---

## 2. Compliance Analyzer Architecture

### High-Level Flow

```
┌─────────────────────────────────────┐
│  Compliance Check Request           │
│  - Contract document                │
│  - Applicable laws                  │
│  - Check mode (exhaustive | sample)│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Requirement Extractor              │
│  - Extract legal requirements       │
│  - Classify (obligation, prohibition)│
│  → List of LegalRequirement         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Contract Clause Mapper             │
│  - Map each contract clause         │
│  - Find relevant law provisions     │
│  → List of ClauseMapping            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Compliance Checker                 │
│  - Compare clause vs. requirement   │
│  - Detect conflicts                 │
│  - Identify gaps                    │
│  → List of ComplianceIssue          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Risk Scorer                        │
│  - Assess severity                  │
│  - Compute risk scores              │
│  → ComplianceIssue with risk scores │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Recommendation Generator           │
│  - Suggest remediation actions      │
│  - Prioritize by risk               │
│  → ComplianceIssue with actions     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Compliance Reporter                │
│  - Aggregate issues                 │
│  - Generate summary                 │
│  → ComplianceReport                 │
└─────────────────────────────────────┘
```

### Component Interaction

```
ComplianceAnalyzer
├── RequirementExtractor
│   └── Extracts obligations/prohibitions from law
├── ContractClauseMapper
│   ├── Uses CrossDocumentRetriever
│   └── Maps contract clauses to law provisions
├── ComplianceChecker
│   ├── ConflictDetector
│   ├── GapAnalyzer
│   └── DeviationAssessor
├── RiskScorer
│   ├── SeverityClassifier
│   └── ImpactCalculator
├── RecommendationGenerator
│   └── Uses Claude Sonnet for suggestions
└── ComplianceReporter
    └── Formats final report
```

---

## 3. Data Structures

### 3.1 Core Types

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class RequirementType(Enum):
    """Type of legal requirement."""
    OBLIGATION = "obligation"       # Must do X
    PROHIBITION = "prohibition"     # Must not do X
    PERMISSION = "permission"       # May do X
    DEFINITION = "definition"       # X is defined as...
    CONDITION = "condition"         # If X then Y

class ComplianceStatus(Enum):
    """Compliance status of a clause."""
    COMPLIANT = "compliant"         # Fully complies
    CONFLICT = "conflict"           # Direct contradiction
    DEVIATION = "deviation"         # Differs but may be acceptable
    MISSING = "missing"             # Required provision absent
    UNCERTAIN = "uncertain"         # Cannot determine
    NOT_APPLICABLE = "not_applicable"  # Requirement doesn't apply

class IssueSeverity(Enum):
    """Severity level for compliance issues."""
    CRITICAL = "critical"   # Showstopper, contract invalid
    HIGH = "high"           # Serious risk, needs immediate fix
    MEDIUM = "medium"       # Moderate risk, should fix
    LOW = "low"             # Minor issue, consider fixing
    INFO = "info"           # Informational, no action needed

@dataclass
class LegalRequirement:
    """A requirement extracted from law."""
    requirement_id: str
    requirement_type: RequirementType

    # Source
    law_reference: str          # §89 odst. 2
    law_document_id: str
    law_chunk_id: str

    # Content
    requirement_text: str       # Full text of requirement
    requirement_summary: str    # Short description

    # Classification
    is_mandatory: bool          # vs. optional
    applies_to: List[str]       # ["contractor", "all_parties"]
    temporal_constraint: Optional[str] = None  # "within 30 days"

    # Metadata
    extraction_confidence: float = 1.0
    related_requirements: List[str] = field(default_factory=list)  # IDs

@dataclass
class ClauseMapping:
    """Mapping between contract clause and law provision."""
    # Contract side
    contract_chunk_id: str
    contract_reference: str     # Článek 5.2
    contract_text: str

    # Law side
    law_requirements: List[LegalRequirement]

    # Matching metadata
    match_score: float          # Confidence in mapping
    match_type: str             # explicit_ref | semantic | structural

@dataclass
class ComplianceIssue:
    """A compliance issue found during analysis."""
    issue_id: str
    status: ComplianceStatus
    severity: IssueSeverity

    # Contract clause
    contract_chunk_id: str
    contract_reference: str
    contract_text: str

    # Related law requirement(s)
    law_requirements: List[LegalRequirement]

    # Issue details
    issue_description: str      # What's wrong
    evidence: str               # Specific text causing issue
    reasoning: str              # Why it's an issue

    # Risk assessment
    risk_score: float           # 0.0 to 1.0
    risk_factors: Dict[str, float]  # {factor: weight}

    # Recommendations
    recommendations: List[str]  # Actions to remediate
    priority: int               # 1 (highest) to 5 (lowest)

    # Metadata
    detected_by: str            # Component that detected issue
    detection_confidence: float = 1.0
    verification_status: Optional[str] = None  # For human review

@dataclass
class ComplianceReport:
    """Final compliance analysis report."""
    # Metadata
    report_id: str
    generated_at: datetime
    contract_id: str
    law_ids: List[str]
    analysis_mode: str          # exhaustive | sample

    # Summary statistics
    total_clauses_checked: int
    total_requirements_checked: int
    total_issues: int
    issues_by_severity: Dict[IssueSeverity, int]
    issues_by_status: Dict[ComplianceStatus, int]

    # Issues
    critical_issues: List[ComplianceIssue]
    high_issues: List[ComplianceIssue]
    medium_issues: List[ComplianceIssue]
    low_issues: List[ComplianceIssue]

    # Overall assessment
    overall_compliance_score: float  # 0.0 to 1.0
    is_compliant: bool              # Overall pass/fail
    risk_level: str                 # critical | high | medium | low

    # Recommendations summary
    top_recommendations: List[str]
    estimated_remediation_effort: str  # hours, simple/moderate/complex

    # Detailed results
    clause_mappings: List[ClauseMapping]
    all_issues: List[ComplianceIssue]

    # Analysis metadata
    processing_time: float
    llm_calls_made: int
    confidence_score: float         # Confidence in analysis
```

---

## 4. Requirement Extractor

### 4.1 Purpose

Extract legal requirements (obligations, prohibitions, conditions) from law documents.

### 4.2 Implementation

```python
from anthropic import Anthropic
import re

class RequirementExtractor:
    """Extract legal requirements from law provisions."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    async def extract_requirements(
        self,
        law_chunks: List[LegalChunk]
    ) -> List[LegalRequirement]:
        """
        Extract requirements from law chunks.

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
        chunk: LegalChunk
    ) -> List[LegalRequirement]:
        """Extract requirements from a single chunk."""
        # First: pattern-based extraction (fast)
        pattern_requirements = self._extract_by_patterns(chunk)

        # Then: LLM-based extraction for nuanced requirements
        llm_requirements = await self._extract_with_llm(chunk)

        # Merge and deduplicate
        all_requirements = pattern_requirements + llm_requirements
        deduplicated = self._deduplicate_requirements(all_requirements)

        return deduplicated

    def _extract_by_patterns(self, chunk: LegalChunk) -> List[LegalRequirement]:
        """Fast pattern-based extraction for common requirement phrases."""
        requirements = []
        text = chunk.content

        # Obligation patterns
        obligation_patterns = [
            r"(musí|je povinen|povinnost|shall|must)\s+([^.]+\.)",
            r"([^.]+)\s+je\s+povinen\s+([^.]+\.)",
        ]

        for i, pattern in enumerate(obligation_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_pat_{i}",
                    requirement_type=RequirementType.OBLIGATION,
                    law_reference=chunk.legal_reference,
                    law_document_id=chunk.document_id,
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=match.group(0),
                    requirement_summary=match.group(0)[:100],
                    is_mandatory=True,
                    applies_to=["all_parties"],
                    extraction_confidence=0.8
                )
                requirements.append(req)

        # Prohibition patterns
        prohibition_patterns = [
            r"(nesmí|zakázáno|zákaz|must not|shall not)\s+([^.]+\.)",
        ]

        for i, pattern in enumerate(prohibition_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                req = LegalRequirement(
                    requirement_id=f"{chunk.chunk_id}_pat_proh_{i}",
                    requirement_type=RequirementType.PROHIBITION,
                    law_reference=chunk.legal_reference,
                    law_document_id=chunk.document_id,
                    law_chunk_id=chunk.chunk_id,
                    requirement_text=match.group(0),
                    requirement_summary=match.group(0)[:100],
                    is_mandatory=True,
                    applies_to=["all_parties"],
                    extraction_confidence=0.8
                )
                requirements.append(req)

        return requirements

    async def _extract_with_llm(self, chunk: LegalChunk) -> List[LegalRequirement]:
        """
        Use LLM to extract nuanced requirements.
        """
        prompt = f"""Analyze this legal provision and extract all requirements.

Legal provision ({chunk.legal_reference}):
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
  }},
  ...
]

Requirements:"""

        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        import json
        try:
            requirements_data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse LLM requirements for {chunk.chunk_id}")
            return []

        # Convert to LegalRequirement objects
        requirements = []
        for i, req_data in enumerate(requirements_data):
            req = LegalRequirement(
                requirement_id=f"{chunk.chunk_id}_llm_{i}",
                requirement_type=RequirementType(req_data["type"]),
                law_reference=chunk.legal_reference,
                law_document_id=chunk.document_id,
                law_chunk_id=chunk.chunk_id,
                requirement_text=chunk.content,  # Full chunk text
                requirement_summary=req_data["summary"],
                is_mandatory=req_data.get("is_mandatory", True),
                applies_to=req_data.get("applies_to", ["all_parties"]),
                temporal_constraint=req_data.get("temporal_constraint"),
                extraction_confidence=0.95  # LLM extraction is high confidence
            )
            requirements.append(req)

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
            summary_lower = req.requirement_summary.lower()
            if summary_lower not in seen_summaries:
                unique.append(req)
                seen_summaries.add(summary_lower)

        return unique
```

---

## 5. Contract Clause Mapper

### 5.1 Purpose

Map each contract clause to relevant law provisions for comparison.

### 5.2 Implementation

```python
class ContractClauseMapper:
    """Map contract clauses to relevant law provisions."""

    def __init__(
        self,
        cross_doc_retriever: 'ComparativeRetriever',
        config: Dict[str, Any]
    ):
        self.retriever = cross_doc_retriever
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def map_clauses(
        self,
        contract_chunks: List[LegalChunk],
        law_requirements: List[LegalRequirement]
    ) -> List[ClauseMapping]:
        """
        Map contract clauses to law requirements.

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
                    contract_reference=contract_chunk.legal_reference,
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
        contract_chunk: LegalChunk,
        all_requirements: List[LegalRequirement]
    ) -> List[LegalRequirement]:
        """
        Find law requirements relevant to this contract clause.
        """
        # Use cross-document retriever to find relevant law chunks
        query = f"Jaké zákonné požadavky se vztahují k: {contract_chunk.content[:200]}"

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
            matching_reqs = [
                req for req in all_requirements
                if req.law_chunk_id == result.target_chunk.chunk_id
            ]
            relevant_requirements.extend(matching_reqs)

        # Deduplicate
        unique_reqs = list({req.requirement_id: req for req in relevant_requirements}.values())

        return unique_reqs[:5]  # Top 5

    def _compute_average_match_score(self, requirements: List[LegalRequirement]) -> float:
        """Compute average extraction confidence as match score."""
        if not requirements:
            return 0.0
        return sum(req.extraction_confidence for req in requirements) / len(requirements)
```

---

## 6. Compliance Checker

### 6.1 Conflict Detector

```python
class ConflictDetector:
    """Detect conflicts between contract and law."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client

    async def detect_conflict(
        self,
        mapping: ClauseMapping
    ) -> Optional[ComplianceIssue]:
        """
        Check if contract clause conflicts with law requirements.

        Returns:
            ComplianceIssue if conflict detected, else None
        """
        # Use LLM to compare clause vs. requirements
        prompt = self._build_conflict_detection_prompt(mapping)

        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        import json
        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse conflict detection response")
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
                issue_description=result["description"],
                evidence=result["evidence"],
                reasoning=result["reasoning"],
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
```

### 6.2 Gap Analyzer

```python
class GapAnalyzer:
    """Identify missing mandatory requirements in contract."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client

    async def analyze_gaps(
        self,
        all_requirements: List[LegalRequirement],
        all_mappings: List[ClauseMapping]
    ) -> List[ComplianceIssue]:
        """
        Find mandatory requirements not covered by contract.

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
```

### 6.3 Deviation Assessor

```python
class DeviationAssessor:
    """Assess whether deviations from law are acceptable."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client

    async def assess_deviation(
        self,
        mapping: ClauseMapping
    ) -> Optional[ComplianceIssue]:
        """
        Check if clause deviates from law in potentially problematic way.

        Returns:
            ComplianceIssue if deviation is problematic
        """
        # Use LLM to assess if deviation is acceptable
        prompt = self._build_deviation_prompt(mapping)

        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
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
                issue_description=result["description"],
                evidence=result["evidence"],
                reasoning=result["reasoning"],
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
```

### 6.4 Main Compliance Checker

```python
class ComplianceChecker:
    """Orchestrate compliance checking."""

    def __init__(self, llm_client: Anthropic):
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

        Returns:
            List of all ComplianceIssues found
        """
        all_issues = []

        # 1. Detect conflicts
        conflict_tasks = [
            self.conflict_detector.detect_conflict(mapping)
            for mapping in mappings
        ]
        conflict_results = await asyncio.gather(*conflict_tasks)
        conflicts = [issue for issue in conflict_results if issue is not None]
        all_issues.extend(conflicts)
        self.logger.info(f"Found {len(conflicts)} conflicts")

        # 2. Detect deviations
        deviation_tasks = [
            self.deviation_assessor.assess_deviation(mapping)
            for mapping in mappings
        ]
        deviation_results = await asyncio.gather(*deviation_tasks)
        deviations = [issue for issue in deviation_results if issue is not None]
        all_issues.extend(deviations)
        self.logger.info(f"Found {len(deviations)} deviations")

        # 3. Find gaps
        gaps = await self.gap_analyzer.analyze_gaps(all_requirements, mappings)
        all_issues.extend(gaps)
        self.logger.info(f"Found {len(gaps)} gaps")

        return all_issues
```

---

## 7. Risk Scorer

```python
class RiskScorer:
    """Assess risk severity of compliance issues."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def score_issues(self, issues: List[ComplianceIssue]) -> List[ComplianceIssue]:
        """
        Assign risk scores and severity levels to issues.

        Modifies issues in-place and returns them.
        """
        for issue in issues:
            # Compute risk factors
            risk_factors = self._compute_risk_factors(issue)

            # Aggregate into overall risk score
            risk_score = self._aggregate_risk_score(risk_factors)

            # Classify severity
            severity = self._classify_severity(issue.status, risk_score)

            # Update issue
            issue.risk_factors = risk_factors
            issue.risk_score = risk_score
            issue.severity = severity
            issue.priority = self._severity_to_priority(severity)

        return issues

    def _compute_risk_factors(self, issue: ComplianceIssue) -> Dict[str, float]:
        """Compute individual risk factors."""
        factors = {}

        # Factor 1: Mandatory vs. optional requirement
        if any(req.is_mandatory for req in issue.law_requirements):
            factors["mandatory"] = 1.0
        else:
            factors["mandatory"] = 0.3

        # Factor 2: Type of issue
        status_weights = {
            ComplianceStatus.CONFLICT: 1.0,
            ComplianceStatus.MISSING: 0.9,
            ComplianceStatus.DEVIATION: 0.5,
            ComplianceStatus.UNCERTAIN: 0.3
        }
        factors["issue_type"] = status_weights.get(issue.status, 0.5)

        # Factor 3: Requirement type (prohibition violations are serious)
        has_prohibition = any(
            req.requirement_type == RequirementType.PROHIBITION
            for req in issue.law_requirements
        )
        factors["prohibition_violation"] = 1.0 if has_prohibition else 0.5

        # Factor 4: Temporal constraints (deadlines increase urgency)
        has_temporal = any(
            req.temporal_constraint is not None
            for req in issue.law_requirements
        )
        factors["temporal"] = 0.8 if has_temporal else 0.4

        # Factor 5: Detection confidence
        factors["confidence"] = issue.detection_confidence

        return factors

    def _aggregate_risk_score(self, factors: Dict[str, float]) -> float:
        """Aggregate risk factors into single score."""
        # Weighted average
        weights = {
            "mandatory": 0.3,
            "issue_type": 0.3,
            "prohibition_violation": 0.2,
            "temporal": 0.1,
            "confidence": 0.1
        }

        risk_score = sum(factors[k] * weights[k] for k in weights if k in factors)

        return np.clip(risk_score, 0.0, 1.0)

    def _classify_severity(
        self,
        status: ComplianceStatus,
        risk_score: float
    ) -> IssueSeverity:
        """Classify severity based on status and risk score."""
        # Conflicts with high risk are critical
        if status == ComplianceStatus.CONFLICT and risk_score >= 0.8:
            return IssueSeverity.CRITICAL

        # Missing mandatory requirements are high severity
        if status == ComplianceStatus.MISSING and risk_score >= 0.7:
            return IssueSeverity.HIGH

        # General risk-based classification
        if risk_score >= 0.8:
            return IssueSeverity.CRITICAL
        elif risk_score >= 0.6:
            return IssueSeverity.HIGH
        elif risk_score >= 0.4:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW

    def _severity_to_priority(self, severity: IssueSeverity) -> int:
        """Convert severity to priority number."""
        priority_map = {
            IssueSeverity.CRITICAL: 1,
            IssueSeverity.HIGH: 2,
            IssueSeverity.MEDIUM: 3,
            IssueSeverity.LOW: 4,
            IssueSeverity.INFO: 5
        }
        return priority_map[severity]
```

---

## 8. Recommendation Generator

```python
class RecommendationGenerator:
    """Generate remediation recommendations for compliance issues."""

    def __init__(self, llm_client: Anthropic):
        self.llm = llm_client

    async def generate_recommendations(
        self,
        issues: List[ComplianceIssue]
    ) -> List[ComplianceIssue]:
        """
        Generate recommendations for each issue.

        Modifies issues in-place.
        """
        # Process in batches
        for issue in issues:
            recommendations = await self._generate_for_issue(issue)
            issue.recommendations = recommendations

        return issues

    async def _generate_for_issue(self, issue: ComplianceIssue) -> List[str]:
        """Generate recommendations for a single issue."""
        prompt = self._build_recommendation_prompt(issue)

        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse recommendations (expect bullet list)
        text = response.content[0].text
        recommendations = self._parse_recommendations(text)

        return recommendations

    def _build_recommendation_prompt(self, issue: ComplianceIssue) -> str:
        req_text = "\n".join([
            f"- {req.requirement_summary} ({req.law_reference})"
            for req in issue.law_requirements
        ])

        prompt = f"""Generate actionable recommendations to remediate this compliance issue.

Issue type: {issue.status.value}
Severity: {issue.severity.value}

Contract clause ({issue.contract_reference}):
{issue.contract_text}

Legal requirements:
{req_text}

Issue: {issue.issue_description}

Generate 2-3 specific, actionable recommendations to fix this issue. Focus on:
- What text to add/change in the contract
- How to align with legal requirements
- Risk mitigation strategies

Format as bullet list:
- [Recommendation 1]
- [Recommendation 2]
...

Recommendations:"""

        return prompt

    def _parse_recommendations(self, text: str) -> List[str]:
        """Parse bullet list of recommendations."""
        lines = text.strip().split("\n")
        recommendations = []

        for line in lines:
            # Match bullet points
            match = re.match(r"^\s*[-*•]\s*(.+)$", line)
            if match:
                recommendations.append(match.group(1).strip())

        return recommendations
```

---

## 9. Compliance Reporter

```python
class ComplianceReporter:
    """Generate final compliance report."""

    def generate_report(
        self,
        contract_id: str,
        law_ids: List[str],
        mappings: List[ClauseMapping],
        issues: List[ComplianceIssue],
        requirements: List[LegalRequirement],
        processing_time: float,
        llm_calls: int,
        mode: str = "exhaustive"
    ) -> ComplianceReport:
        """Generate compliance report from analysis results."""
        # Group issues by severity
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high = [i for i in issues if i.severity == IssueSeverity.HIGH]
        medium = [i for i in issues if i.severity == IssueSeverity.MEDIUM]
        low = [i for i in issues if i.severity == IssueSeverity.LOW]

        # Compute statistics
        issues_by_severity = {
            IssueSeverity.CRITICAL: len(critical),
            IssueSeverity.HIGH: len(high),
            IssueSeverity.MEDIUM: len(medium),
            IssueSeverity.LOW: len(low)
        }

        issues_by_status = {}
        for status in ComplianceStatus:
            issues_by_status[status] = len([i for i in issues if i.status == status])

        # Overall compliance score (1.0 = perfect, 0.0 = many critical issues)
        compliance_score = self._compute_compliance_score(issues, requirements)

        # Overall pass/fail
        is_compliant = len(critical) == 0 and len(high) <= 2

        # Risk level
        if len(critical) > 0:
            risk_level = "critical"
        elif len(high) > 3:
            risk_level = "high"
        elif len(medium) > 5:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Top recommendations (from critical/high issues)
        top_issues = critical + high
        top_recommendations = []
        for issue in top_issues[:5]:  # Top 5 issues
            if issue.recommendations:
                top_recommendations.append(f"{issue.contract_reference}: {issue.recommendations[0]}")

        # Confidence score
        avg_confidence = np.mean([i.detection_confidence for i in issues]) if issues else 1.0

        report = ComplianceReport(
            report_id=f"report_{contract_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            contract_id=contract_id,
            law_ids=law_ids,
            analysis_mode=mode,
            total_clauses_checked=len(mappings),
            total_requirements_checked=len(requirements),
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues_by_status=issues_by_status,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            overall_compliance_score=compliance_score,
            is_compliant=is_compliant,
            risk_level=risk_level,
            top_recommendations=top_recommendations,
            estimated_remediation_effort=self._estimate_effort(issues),
            clause_mappings=mappings,
            all_issues=issues,
            processing_time=processing_time,
            llm_calls_made=llm_calls,
            confidence_score=avg_confidence
        )

        return report

    def _compute_compliance_score(
        self,
        issues: List[ComplianceIssue],
        requirements: List[LegalRequirement]
    ) -> float:
        """
        Compute overall compliance score.
        1.0 = fully compliant, 0.0 = many critical issues.
        """
        if not requirements:
            return 1.0

        # Penalty for each issue type
        penalties = {
            IssueSeverity.CRITICAL: 0.2,
            IssueSeverity.HIGH: 0.1,
            IssueSeverity.MEDIUM: 0.05,
            IssueSeverity.LOW: 0.02
        }

        total_penalty = sum(penalties.get(i.severity, 0.0) for i in issues)

        # Score = 1.0 - total_penalty (clamped to [0, 1])
        score = max(0.0, 1.0 - total_penalty)

        return score

    def _estimate_effort(self, issues: List[ComplianceIssue]) -> str:
        """Estimate remediation effort."""
        # Count by severity
        critical = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high = len([i for i in issues if i.severity == IssueSeverity.HIGH])
        medium = len([i for i in issues if i.severity == IssueSeverity.MEDIUM])

        # Rough estimate (hours)
        hours = critical * 4 + high * 2 + medium * 1

        if hours > 40:
            return "complex (40+ hours)"
        elif hours > 16:
            return "moderate (16-40 hours)"
        elif hours > 4:
            return "simple (4-16 hours)"
        else:
            return "minimal (<4 hours)"
```

---

## 10. Main Compliance Analyzer

```python
class ComplianceAnalyzer:
    """Main orchestrator for compliance analysis."""

    def __init__(
        self,
        config: Dict[str, Any],
        cross_doc_retriever: 'ComparativeRetriever',
        llm_client: Anthropic
    ):
        self.config = config
        self.llm = llm_client

        # Initialize components
        self.requirement_extractor = RequirementExtractor(llm_client)
        self.clause_mapper = ContractClauseMapper(cross_doc_retriever, config)
        self.compliance_checker = ComplianceChecker(llm_client)
        self.risk_scorer = RiskScorer()
        self.recommendation_generator = RecommendationGenerator(llm_client)
        self.reporter = ComplianceReporter()

        self.logger = logging.getLogger(__name__)
        self.llm_calls_count = 0

    async def analyze_compliance(
        self,
        contract_chunks: List[LegalChunk],
        law_chunks: List[LegalChunk],
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
            ComplianceReport
        """
        start_time = time.time()
        self.llm_calls_count = 0

        self.logger.info(f"Starting compliance analysis (mode: {mode})")
        self.logger.info(f"Contract chunks: {len(contract_chunks)}, Law chunks: {len(law_chunks)}")

        # Step 1: Extract requirements from law
        self.logger.info("Extracting legal requirements...")
        requirements = await self.requirement_extractor.extract_requirements(law_chunks)
        self.llm_calls_count += len(law_chunks)  # Approx

        # Step 2: Map contract clauses to law requirements
        self.logger.info("Mapping contract clauses to law requirements...")
        mappings = await self.clause_mapper.map_clauses(contract_chunks, requirements)

        # Step 3: Check compliance (detect conflicts, gaps, deviations)
        self.logger.info("Checking compliance...")
        issues = await self.compliance_checker.check_compliance(mappings, requirements)
        self.llm_calls_count += len(mappings) * 2  # Conflict + deviation checks

        # Step 4: Score risks
        self.logger.info("Scoring risks...")
        issues = self.risk_scorer.score_issues(issues)

        # Step 5: Generate recommendations
        self.logger.info("Generating recommendations...")
        issues = await self.recommendation_generator.generate_recommendations(issues)
        self.llm_calls_count += len(issues)

        # Step 6: Generate report
        self.logger.info("Generating report...")
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

        self.logger.info(f"Compliance analysis complete in {processing_time:.1f}s")
        self.logger.info(f"Found {report.total_issues} issues: "
                        f"{len(report.critical_issues)} critical, "
                        f"{len(report.high_issues)} high, "
                        f"{len(report.medium_issues)} medium, "
                        f"{len(report.low_issues)} low")

        return report
```

---

## 11. Usage Examples

### 11.1 Basic Compliance Check

```python
from src.compliance_analyzer import ComplianceAnalyzer

# Initialize
analyzer = ComplianceAnalyzer(config, cross_doc_retriever, llm_client)

# Get document chunks (from indexing pipeline)
contract_chunks = await index.get_chunks(document_id="contract_123")
law_chunks = await index.get_chunks(document_id="law_89_2012")

# Run analysis
report = await analyzer.analyze_compliance(
    contract_chunks=contract_chunks,
    law_chunks=law_chunks,
    contract_id="contract_123",
    law_ids=["law_89_2012"],
    mode="exhaustive"
)

# Print summary
print(f"Compliance Score: {report.overall_compliance_score:.2f}")
print(f"Risk Level: {report.risk_level}")
print(f"Total Issues: {report.total_issues}")
print(f"  Critical: {len(report.critical_issues)}")
print(f"  High: {len(report.high_issues)}")
print(f"Is Compliant: {report.is_compliant}")
```

### 11.2 Export Report

```python
# Export to JSON
import json

with open("compliance_report.json", "w", encoding="utf-8") as f:
    json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2, default=str)

# Generate Markdown summary
markdown = f"""# Compliance Report

**Contract**: {report.contract_id}
**Laws**: {', '.join(report.law_ids)}
**Generated**: {report.generated_at}

## Summary

- **Compliance Score**: {report.overall_compliance_score:.2%}
- **Risk Level**: {report.risk_level.upper()}
- **Is Compliant**: {"✅ Yes" if report.is_compliant else "❌ No"}

## Issues Found

- **Critical**: {len(report.critical_issues)}
- **High**: {len(report.high_issues)}
- **Medium**: {len(report.medium_issues)}
- **Low**: {len(report.low_issues)}

## Top Issues

{chr(10).join([f"### {i+1}. {issue.contract_reference}: {issue.issue_description}" for i, issue in enumerate(report.critical_issues[:3])])}

## Recommendations

{chr(10).join([f"- {rec}" for rec in report.top_recommendations[:5]])}

**Estimated Remediation Effort**: {report.estimated_remediation_effort}
"""

with open("compliance_report.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

---

## 12. Configuration

### config.yaml

```yaml
compliance:
  # Analysis mode
  default_mode: "exhaustive"  # exhaustive | sample

  # Requirement extraction
  extract_all_requirement_types: true
  llm_for_nuanced_requirements: true

  # Risk scoring
  risk_scoring:
    mandatory_weight: 0.3
    issue_type_weight: 0.3
    prohibition_weight: 0.2
    temporal_weight: 0.1
    confidence_weight: 0.1

  # Severity thresholds
  severity_thresholds:
    critical: 0.8
    high: 0.6
    medium: 0.4
    low: 0.0

  # Recommendations
  max_recommendations_per_issue: 3
  generate_recommendations: true

  # Reporting
  include_low_severity: true
  include_clause_mappings: true
  export_formats: ["json", "markdown", "html"]
```

---

## 13. Summary

Compliance Analyzer provides automated, thorough compliance checking:

1. **Requirement Extraction** - Identifies all legal obligations, prohibitions, conditions from law
2. **Clause Mapping** - Maps each contract clause to relevant law provisions
3. **Multi-Faceted Checking** - Detects conflicts, gaps, and problematic deviations
4. **Risk-Based Prioritization** - Scores severity and prioritizes issues
5. **Actionable Recommendations** - Suggests specific remediation steps

**Next Steps**:
- See [10. Knowledge Graph](10_knowledge_graph.md) for graph construction
- See [11. API Interfaces](11_api_interfaces.md) for integration

---

**Page Count**: ~22 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
