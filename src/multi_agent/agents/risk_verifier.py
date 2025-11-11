"""
Risk Verifier Agent - Risk assessment and verification.

Responsibilities:
1. Risk identification (Legal, Financial, Operational, Compliance, Reputational)
2. Severity and likelihood assessment
3. Comparison with industry standards
4. Mitigation recommendations
"""

import logging
from typing import Any, Dict, List

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("risk_verifier")
class RiskVerifierAgent(BaseAgent):
    """
    Risk Verifier Agent - Assesses and verifies risks.

    Identifies risks across 5 categories (Legal, Financial, Operational,
    Compliance, Reputational) and provides severity scores with mitigation.
    """

    def __init__(self, config):
        """Initialize risk verifier with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("risk_verifier")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"RiskVerifierAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess and verify risks.

        Args:
            state: Current workflow state

        Returns:
            Updated state with risk assessment
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping risk verification")
            return state

        logger.info("Assessing risks...")

        try:
            # Search for similar risk patterns
            similar_risks = await self._find_similar_risks(query)

            # Compare with multiple documents if available
            document_comparison = await self._compare_documents(
                extractor_output.get("document_summaries", [])
            )

            # Perform risk assessment using LLM
            risk_assessment = await self._assess_risks(
                query=query,
                extractor_output=extractor_output,
                similar_risks=similar_risks,
                document_comparison=document_comparison
            )

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["risk_verifier"] = risk_assessment

            logger.info(
                f"Risk assessment complete: {len(risk_assessment.get('risks', []))} risks identified, "
                f"overall_score={risk_assessment.get('overall_risk_score', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Risk verification failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Risk verification error: {str(e)}")
            return state

    async def _find_similar_risks(self, query: str) -> List[Dict[str, Any]]:
        """Find similar risk patterns across documents."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="similarity_search",
                inputs={
                    "query": query,
                    "search_type": "risk_patterns",
                    "k": 5
                },
                agent_name=self.config.name
            )

            return result.get("data", []) if result["success"] else []

        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []

    async def _compare_documents(
        self,
        document_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple documents for risk analysis."""
        if len(document_summaries) < 2:
            return {}

        try:
            # Get first two documents for comparison
            doc_ids = [doc.get("document_id") for doc in document_summaries[:2]]

            result = await self.tool_adapter.execute(
                tool_name="compare_documents",
                inputs={
                    "document_ids": doc_ids,
                    "comparison_type": "risk_analysis"
                },
                agent_name=self.config.name
            )

            return result.get("data", {}) if result["success"] else {}

        except Exception as e:
            logger.warning(f"Document comparison failed: {e}")
            return {}

    async def _assess_risks(
        self,
        query: str,
        extractor_output: Dict[str, Any],
        similar_risks: List[Dict[str, Any]],
        document_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks using LLM."""
        # Prepare risk assessment prompt
        chunks = extractor_output.get("chunks", [])[:10]
        chunks_text = "\n\n".join([
            f"Chunk {i+1}: {chunk.get('text', '')[:500]}"
            for i, chunk in enumerate(chunks)
        ])

        similar_risks_text = "\n".join([
            f"- {risk.get('description', '')[:200]}"
            for risk in similar_risks[:3]
        ])

        user_message = f"""Assess risks in the following content:

Query: {query}

Extracted Content:
{chunks_text}

Similar Risk Patterns Found:
{similar_risks_text}

Document Comparison:
{str(document_comparison)[:500]}

Provide risk assessment in the JSON format specified in the system prompt."""

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

            # Parse risk assessment
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group(0))
                return assessment
            else:
                raise ValueError("No JSON found in risk assessment response")

        except Exception as e:
            logger.error(f"Risk assessment LLM call failed: {e}", exc_info=True)
            return self._get_fallback_risk_assessment()

    def _get_fallback_risk_assessment(self) -> Dict[str, Any]:
        """Fallback risk assessment."""
        return {
            "risks": [],
            "overall_risk_score": 50,
            "critical_items": [],
            "recommendations": ["Manual review recommended due to processing error"]
        }
