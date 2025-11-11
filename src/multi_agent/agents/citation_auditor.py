"""
Citation Auditor Agent - Citation verification and validation.

Responsibilities:
1. Citation existence verification
2. Citation accuracy checking (text matches)
3. Citation completeness validation
4. Citation format standardization
5. Broken reference detection
"""

import logging
import re
from typing import Any, Dict, List

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("citation_auditor")
class CitationAuditorAgent(BaseAgent):
    """
    Citation Auditor Agent - Verifies citation accuracy and completeness.

    Audits all citations to ensure they are accurate, complete, properly
    formatted, and point to accessible sources.
    """

    def __init__(self, config):
        """Initialize citation auditor with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("citation_auditor")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"CitationAuditorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit citations for accuracy and completeness.

        Args:
            state: Current workflow state

        Returns:
            Updated state with citation audit results
        """
        citations = state.get("citations", [])

        if not citations:
            logger.info("No citations to audit")
            return state

        logger.info(f"Auditing {len(citations)} citations...")

        try:
            # Extract and parse citations
            parsed_citations = self._parse_citations(citations)

            # Verify each citation
            verification_results = await self._verify_citations(parsed_citations)

            # Calculate audit metrics
            audit_results = self._calculate_audit_metrics(verification_results)

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["citation_auditor"] = audit_results

            logger.info(
                f"Citation audit complete: {audit_results['verified_citations']}/{audit_results['total_citations']} verified, "
                f"quality_score={audit_results['quality_score']}"
            )

            return state

        except Exception as e:
            logger.error(f"Citation audit failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Citation audit error: {str(e)}")
            return state

    def _parse_citations(self, citations: List[str]) -> List[Dict[str, Any]]:
        """Parse citation strings into structured format."""
        parsed = []

        for citation in citations:
            try:
                # Extract document, section, page from citation
                # Format: [Doc: filename, Section: X.Y, Page: N]
                doc_match = re.search(r'Doc:\s*([^,\]]+)', citation)
                section_match = re.search(r'Section:\s*([^,\]]+)', citation)
                page_match = re.search(r'Page:\s*([^,\]]+)', citation)

                parsed_citation = {
                    "original": citation,
                    "document": doc_match.group(1).strip() if doc_match else None,
                    "section": section_match.group(1).strip() if section_match else None,
                    "page": page_match.group(1).strip() if page_match else None
                }

                parsed.append(parsed_citation)

            except Exception as e:
                logger.warning(f"Failed to parse citation: {citation}, error: {e}")
                parsed.append({
                    "original": citation,
                    "document": None,
                    "section": None,
                    "page": None,
                    "parse_error": str(e)
                })

        return parsed

    async def _verify_citations(
        self,
        parsed_citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify each citation."""
        verification_results = []

        for citation in parsed_citations:
            result = {
                "citation": citation["original"],
                "verified": False,
                "issues": []
            }

            # Check if citation has document reference
            if not citation.get("document"):
                result["issues"].append("missing_document")
                verification_results.append(result)
                continue

            # Verify document exists
            doc_verified = await self._verify_document(citation["document"])
            if not doc_verified:
                result["issues"].append("document_not_found")
                verification_results.append(result)
                continue

            # Verify citation format
            if not citation.get("section") and not citation.get("page"):
                result["issues"].append("incomplete_reference")

            # If all checks pass
            if not result["issues"]:
                result["verified"] = True

            verification_results.append(result)

        return verification_results

    async def _verify_document(self, document_ref: str) -> bool:
        """Verify that document exists."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="get_document_info",
                inputs={"document_id": document_ref},
                agent_name=self.config.name
            )

            return result["success"]

        except Exception as e:
            logger.warning(f"Document verification failed for {document_ref}: {e}")
            return False

    def _calculate_audit_metrics(
        self,
        verification_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate audit metrics from verification results."""
        total = len(verification_results)
        verified = sum(1 for r in verification_results if r["verified"])
        broken = [r for r in verification_results if not r["verified"]]

        verification_rate = (verified / total * 100) if total > 0 else 0

        # Quality score considers both verification rate and issue severity
        quality_score = verification_rate

        # Penalize for specific issues
        for result in broken:
            if "document_not_found" in result["issues"]:
                quality_score -= 5  # Major issue
            elif "incomplete_reference" in result["issues"]:
                quality_score -= 2  # Minor issue

        quality_score = max(0, min(100, quality_score))

        return {
            "total_citations": total,
            "verified_citations": verified,
            "broken_citations": [
                {
                    "citation": r["citation"],
                    "issue": ", ".join(r["issues"]),
                    "suggestion": self._suggest_fix(r)
                }
                for r in broken
            ],
            "verification_rate": round(verification_rate, 1),
            "quality_score": round(quality_score, 1),
            "recommendations": self._generate_recommendations(broken)
        }

    def _suggest_fix(self, result: Dict[str, Any]) -> str:
        """Suggest fix for broken citation."""
        issues = result.get("issues", [])

        if "missing_document" in issues:
            return "Add document reference"
        elif "document_not_found" in issues:
            return "Verify document exists or update reference"
        elif "incomplete_reference" in issues:
            return "Add section and/or page number"
        else:
            return "Review citation format"

    def _generate_recommendations(self, broken: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on broken citations."""
        recommendations = []

        if len(broken) > 0:
            recommendations.append(f"Review and fix {len(broken)} broken citation(s)")

        doc_not_found_count = sum(
            1 for r in broken if "document_not_found" in r.get("issues", [])
        )
        if doc_not_found_count > 0:
            recommendations.append(
                f"Verify {doc_not_found_count} document reference(s) exist"
            )

        incomplete_count = sum(
            1 for r in broken if "incomplete_reference" in r.get("issues", [])
        )
        if incomplete_count > 0:
            recommendations.append(
                f"Complete {incomplete_count} citation(s) with section/page numbers"
            )

        return recommendations
