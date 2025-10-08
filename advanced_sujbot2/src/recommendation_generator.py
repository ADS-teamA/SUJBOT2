"""
Recommendation Generator for compliance analysis.

Generates remediation recommendations for compliance issues using LLM.
"""

import logging
import re
from typing import List
from anthropic import Anthropic

from .compliance_types import ComplianceIssue


class RecommendationGenerator:
    """Generate remediation recommendations for compliance issues."""

    def __init__(self, llm_client: Anthropic, config: dict = None):
        """
        Initialize RecommendationGenerator.

        Args:
            llm_client: Anthropic client for LLM-based generation
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.max_recommendations = self.config.get('max_recommendations_per_issue', 3)

    async def generate_recommendations(
        self,
        issues: List[ComplianceIssue]
    ) -> List[ComplianceIssue]:
        """
        Generate recommendations for each issue.

        Modifies issues in-place.

        Args:
            issues: List of ComplianceIssue objects

        Returns:
            Same list with recommendations populated
        """
        # Process in batches
        for issue in issues:
            try:
                recommendations = await self._generate_for_issue(issue)
                issue.recommendations = recommendations
            except Exception as e:
                self.logger.error(f"Error generating recommendations for {issue.issue_id}: {e}")
                issue.recommendations = ["Error generating recommendations. Manual review required."]

        self.logger.info(f"Generated recommendations for {len(issues)} issues")
        return issues

    async def _generate_for_issue(self, issue: ComplianceIssue) -> List[str]:
        """Generate recommendations for a single issue."""
        prompt = self._build_recommendation_prompt(issue)

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse recommendations (expect bullet list)
            text = response.content[0].text
            recommendations = self._parse_recommendations(text)

            # Limit to max recommendations
            return recommendations[:self.max_recommendations]

        except Exception as e:
            self.logger.error(f"LLM call failed for recommendations: {e}")
            return [f"Manual review required due to generation error"]

    def _build_recommendation_prompt(self, issue: ComplianceIssue) -> str:
        """Build prompt for recommendation generation."""
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
            # Match bullet points (-, *, •, or numbered)
            match = re.match(r"^\s*[-*•]\s*(.+)$", line)
            if match:
                recommendations.append(match.group(1).strip())
            else:
                # Try numbered list
                match = re.match(r"^\s*\d+\.\s*(.+)$", line)
                if match:
                    recommendations.append(match.group(1).strip())

        # If no bullet points found, treat each non-empty line as a recommendation
        if not recommendations:
            recommendations = [
                line.strip() for line in lines
                if line.strip() and not line.strip().endswith(':')
            ]

        return recommendations
