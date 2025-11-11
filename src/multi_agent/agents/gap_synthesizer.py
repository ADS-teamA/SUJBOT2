"""
Gap Synthesizer Agent - Knowledge gap analysis and completeness assessment.

Responsibilities:
1. Regulatory gap identification (missing required clauses)
2. Coverage gap analysis (topics not fully addressed)
3. Consistency gap detection (contradictions)
4. Citation gap finding (claims without evidence)
5. Temporal gap identification (outdated information)
"""

import logging
from typing import Any, Dict, List

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("gap_synthesizer")
class GapSynthesizerAgent(BaseAgent):
    """
    Gap Synthesizer Agent - Identifies knowledge gaps and missing information.

    Analyzes completeness across 5 gap types: Regulatory, Coverage,
    Consistency, Citation, and Temporal gaps.
    """

    def __init__(self, config):
        """Initialize gap synthesizer with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("gap_synthesizer")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"GapSynthesizerAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify knowledge gaps and completeness issues.

        Args:
            state: Current workflow state

        Returns:
            Updated state with gap analysis
        """
        query = state.get("query", "")
        agent_outputs = state.get("agent_outputs", {})

        logger.info("Analyzing knowledge gaps...")

        try:
            # Browse knowledge graph entities to find missing connections
            entity_analysis = await self._analyze_entity_coverage()

            # Perform multi-hop graph traversal for relationship gaps
            relationship_gaps = await self._find_relationship_gaps(query)

            # Compare documents to identify coverage gaps
            coverage_gaps = await self._analyze_coverage_gaps(
                agent_outputs.get("extractor", {})
            )

            # Synthesize all gap findings using LLM
            gap_analysis = await self._synthesize_gaps(
                query=query,
                agent_outputs=agent_outputs,
                entity_analysis=entity_analysis,
                relationship_gaps=relationship_gaps,
                coverage_gaps=coverage_gaps
            )

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["gap_synthesizer"] = gap_analysis

            logger.info(
                f"Gap analysis complete: {len(gap_analysis.get('gaps', []))} gaps identified, "
                f"completeness_score={gap_analysis.get('completeness_score', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Gap synthesis failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Gap synthesis error: {str(e)}")
            return state

    async def _analyze_entity_coverage(self) -> Dict[str, Any]:
        """Browse knowledge graph entities to assess coverage."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="browse_entities",
                inputs={
                    "entity_types": ["regulation", "requirement", "clause", "standard"],
                    "limit": 50
                },
                agent_name=self.config.name
            )

            return result.get("data", {}) if result["success"] else {}

        except Exception as e:
            logger.warning(f"Entity coverage analysis failed: {e}")
            return {}

    async def _find_relationship_gaps(self, query: str) -> List[Dict[str, Any]]:
        """Find relationship gaps using multi-hop graph traversal."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="graph_search",
                inputs={
                    "query": query,
                    "max_hops": 3,
                    "find_missing_links": True
                },
                agent_name=self.config.name
            )

            return result.get("data", {}).get("missing_links", []) if result["success"] else []

        except Exception as e:
            logger.warning(f"Relationship gap analysis failed: {e}")
            return []

    async def _analyze_coverage_gaps(
        self,
        extractor_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze coverage gaps by comparing documents."""
        document_summaries = extractor_output.get("document_summaries", [])

        if len(document_summaries) < 2:
            return {}

        try:
            doc_ids = [doc.get("document_id") for doc in document_summaries[:3]]

            result = await self.tool_adapter.execute(
                tool_name="compare_documents",
                inputs={
                    "document_ids": doc_ids,
                    "comparison_type": "coverage_gaps"
                },
                agent_name=self.config.name
            )

            return result.get("data", {}) if result["success"] else {}

        except Exception as e:
            logger.warning(f"Coverage gap analysis failed: {e}")
            return {}

    async def _synthesize_gaps(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        entity_analysis: Dict[str, Any],
        relationship_gaps: List[Dict[str, Any]],
        coverage_gaps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize gap analysis using LLM."""
        # Prepare synthesis prompt
        compliance_output = agent_outputs.get("compliance", {})
        risk_output = agent_outputs.get("risk_verifier", {})
        citation_output = agent_outputs.get("citation_auditor", {})

        user_message = f"""Analyze knowledge gaps in the following context:

Query: {query}

Compliance Findings:
- Violations: {len(compliance_output.get('violations', []))}
- Gaps: {len(compliance_output.get('gaps', []))}

Risk Assessment:
- Overall Risk Score: {risk_output.get('overall_risk_score', 'N/A')}
- Critical Items: {len(risk_output.get('critical_items', []))}

Citation Audit:
- Broken Citations: {len(citation_output.get('broken_citations', []))}
- Quality Score: {citation_output.get('quality_score', 'N/A')}

Knowledge Graph Analysis:
- Entity Coverage: {str(entity_analysis)[:500]}
- Relationship Gaps: {len(relationship_gaps)} missing links found

Coverage Gaps:
{str(coverage_gaps)[:500]}

Provide comprehensive gap analysis in the JSON format specified in the system prompt."""

        try:
            api_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": user_message}]
            }

            if self.config.enable_prompt_caching:
                api_params["system"] = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

            response = self.client.messages.create(**api_params)
            response_text = response.content[0].text

            # Parse gap analysis
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                return analysis
            else:
                raise ValueError("No JSON found in gap analysis response")

        except Exception as e:
            logger.error(f"Gap synthesis LLM call failed: {e}", exc_info=True)
            return self._get_fallback_gap_analysis()

    def _get_fallback_gap_analysis(self) -> Dict[str, Any]:
        """Fallback gap analysis."""
        return {
            "gaps": [],
            "completeness_score": 60,
            "critical_gaps": [],
            "recommended_actions": [
                {
                    "action": "Manual review recommended due to processing error",
                    "priority": 5,
                    "estimated_effort": "medium"
                }
            ]
        }
