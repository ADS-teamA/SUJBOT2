"""Multi-agent implementations for SUJBOT2.

This package contains 8 specialized agents:
- OrchestratorAgent: Query complexity analysis, routing, and report synthesis
- ExtractorAgent: Document retrieval and context extraction
- ClassifierAgent: Content categorization and classification
- RequirementExtractorAgent: Atomic legal requirement extraction (SOTA 2024)
- ComplianceAgent: Regulatory compliance verification
- RiskVerifierAgent: Risk assessment and severity scoring
- CitationAuditorAgent: Citation verification and validation
- GapSynthesizerAgent: Knowledge gap analysis
"""

from .orchestrator import OrchestratorAgent
from .extractor import ExtractorAgent
from .classifier import ClassifierAgent
from .requirement_extractor import RequirementExtractorAgent
from .compliance import ComplianceAgent
from .risk_verifier import RiskVerifierAgent
from .citation_auditor import CitationAuditorAgent
from .gap_synthesizer import GapSynthesizerAgent

__all__ = [
    "OrchestratorAgent",
    "ExtractorAgent",
    "ClassifierAgent",
    "RequirementExtractorAgent",
    "ComplianceAgent",
    "RiskVerifierAgent",
    "CitationAuditorAgent",
    "GapSynthesizerAgent",
]
