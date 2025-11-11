"""
Compliance Agent - Regulatory compliance verification.

Responsibilities:
1. GDPR, CCPA, HIPAA, SOX compliance verification
2. Bidirectional checking (Contract → Law, Law → Contract)
3. Violation identification
4. Gap analysis for missing requirements
"""

import logging
from typing import Any, Dict, List

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("compliance")
class ComplianceAgent(BaseAgent):
    """
    Compliance Agent - Verifies regulatory compliance.

    Checks for GDPR, CCPA, HIPAA, SOX compliance and identifies
    violations and gaps using bidirectional verification.
    """

    def __init__(self, config):
        """Initialize compliance agent with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("compliance")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"ComplianceAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify compliance with regulations.

        Args:
            state: Current workflow state

        Returns:
            Updated state with compliance findings
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping compliance check")
            return state

        logger.info("Verifying regulatory compliance...")

        try:
            # Identify relevant framework from query
            framework = self._identify_framework(query)

            # Perform graph search for regulatory entities
            graph_results = await self._search_regulatory_graph(query, framework)

            # Verify compliance using LLM
            compliance_findings = await self._verify_compliance(
                query=query,
                framework=framework,
                extractor_output=extractor_output,
                graph_results=graph_results
            )

            # Assess confidence in findings
            confidence_result = await self.tool_adapter.execute(
                tool_name="assess_confidence",
                inputs={"findings": compliance_findings},
                agent_name=self.config.name
            )

            if confidence_result["success"]:
                compliance_findings["confidence"] = confidence_result["data"].get("confidence", 70)

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["compliance"] = compliance_findings

            logger.info(
                f"Compliance check complete: framework={framework}, "
                f"violations={len(compliance_findings.get('violations', []))}, "
                f"gaps={len(compliance_findings.get('gaps', []))}"
            )

            return state

        except Exception as e:
            logger.error(f"Compliance verification failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Compliance error: {str(e)}")
            return state

    def _identify_framework(self, query: str) -> str:
        """Identify relevant compliance framework from query."""
        query_lower = query.lower()

        if "gdpr" in query_lower or "data protection" in query_lower:
            return "GDPR"
        elif "ccpa" in query_lower or "california" in query_lower:
            return "CCPA"
        elif "hipaa" in query_lower or "healthcare" in query_lower or "phi" in query_lower:
            return "HIPAA"
        elif "sox" in query_lower or "sarbanes" in query_lower:
            return "SOX"
        else:
            return "General"

    async def _search_regulatory_graph(
        self,
        query: str,
        framework: str
    ) -> Dict[str, Any]:
        """Search knowledge graph for regulatory entities."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="graph_search",
                inputs={
                    "query": f"{framework} {query}",
                    "entity_types": ["regulation", "requirement", "clause"]
                },
                agent_name=self.config.name
            )

            return result.get("data", {}) if result["success"] else {}

        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return {}

    async def _verify_compliance(
        self,
        query: str,
        framework: str,
        extractor_output: Dict[str, Any],
        graph_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify compliance using LLM."""
        # Prepare verification prompt
        chunks = extractor_output.get("chunks", [])[:10]
        chunks_text = "\n\n".join([
            f"Chunk {i+1}: {chunk.get('text', '')[:500]}"
            for i, chunk in enumerate(chunks)
        ])

        user_message = f"""Verify compliance with {framework}:

Query: {query}

Extracted Content:
{chunks_text}

Knowledge Graph Results:
{str(graph_results)[:500]}

Provide compliance analysis in the JSON format specified in the system prompt."""

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

            # Parse compliance findings
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                findings = json.loads(json_match.group(0))
                findings["framework"] = framework
                return findings
            else:
                raise ValueError("No JSON found in compliance response")

        except Exception as e:
            logger.error(f"Compliance verification LLM call failed: {e}", exc_info=True)
            return self._get_fallback_compliance(framework)

    def _get_fallback_compliance(self, framework: str) -> Dict[str, Any]:
        """Fallback compliance result."""
        return {
            "framework": framework,
            "violations": [],
            "gaps": [],
            "confidence": 40,
            "recommendations": ["Manual review recommended due to processing error"]
        }
