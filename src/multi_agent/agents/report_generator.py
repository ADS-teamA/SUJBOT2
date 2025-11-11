"""
Report Generator Agent - Final report compilation and synthesis.

Responsibilities:
1. Executive summary creation
2. Detailed findings compilation
3. Compliance matrix generation
4. Risk assessment summary
5. Citations and references consolidation
6. Recommendations prioritization
7. Appendix with metadata
"""

import logging
from typing import Any, Dict
from datetime import datetime

from anthropic import Anthropic

from ..core.agent_base import BaseAgent
from ..core.agent_registry import register_agent
from ..prompts.loader import get_prompt_loader
from ..tools.adapter import get_tool_adapter

logger = logging.getLogger(__name__)


@register_agent("report_generator")
class ReportGeneratorAgent(BaseAgent):
    """
    Report Generator Agent - Synthesizes all agent outputs into final report.

    Creates comprehensive, well-structured Markdown report with:
    - Executive summary
    - Detailed findings
    - Compliance matrix
    - Risk assessment
    - Citations
    - Recommendations
    - Appendix with execution metadata
    """

    def __init__(self, config):
        """Initialize report generator with config."""
        super().__init__(config)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=config.api_key)

        # Load system prompt
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_prompt("report_generator")

        # Initialize tool adapter
        self.tool_adapter = get_tool_adapter()

        logger.info(f"ReportGeneratorAgent initialized with model: {config.model}")

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive final report.

        Args:
            state: Complete workflow state with all agent outputs

        Returns:
            Updated state with final_answer containing Markdown report
        """
        query = state.get("query", "")
        agent_outputs = state.get("agent_outputs", {})

        if not agent_outputs:
            logger.warning("No agent outputs to compile into report")
            state["final_answer"] = "No analysis results available to generate report."
            return state

        logger.info("Generating final report...")

        try:
            # Get tool usage statistics
            tool_stats = await self._get_tool_stats()

            # Generate report using LLM
            report = await self._generate_report(
                query=query,
                agent_outputs=agent_outputs,
                tool_stats=tool_stats,
                state=state
            )

            # Update state with final answer
            state["final_answer"] = report

            # Store report in agent outputs
            state["agent_outputs"]["report_generator"] = {
                "report_generated": True,
                "report_length": len(report),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Report generated successfully ({len(report)} characters)")

            return state

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Report generation error: {str(e)}")

            # Generate fallback report
            state["final_answer"] = self._generate_fallback_report(query, agent_outputs)

            return state

    async def _get_tool_stats(self) -> Dict[str, Any]:
        """Get execution statistics from tool adapter."""
        try:
            result = await self.tool_adapter.execute(
                tool_name="get_stats",
                inputs={},
                agent_name=self.config.name
            )

            return result.get("data", {}) if result["success"] else {}

        except Exception as e:
            logger.warning(f"Failed to get tool stats: {e}")
            return {}

    async def _generate_report(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        tool_stats: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """Generate report using LLM."""
        # Prepare comprehensive report generation prompt
        user_message = f"""Generate a comprehensive final report for the following analysis:

**QUERY:**
{query}

**ORCHESTRATOR OUTPUT:**
- Complexity Score: {state.get('complexity_score', 'N/A')}
- Query Type: {state.get('query_type', 'N/A')}
- Agent Sequence: {', '.join(state.get('agent_sequence', []))}

**EXTRACTOR OUTPUT:**
{self._format_extractor_output(agent_outputs.get('extractor', {}))}

**CLASSIFIER OUTPUT:**
{self._format_classifier_output(agent_outputs.get('classifier', {}))}

**COMPLIANCE OUTPUT:**
{self._format_compliance_output(agent_outputs.get('compliance', {}))}

**RISK VERIFIER OUTPUT:**
{self._format_risk_output(agent_outputs.get('risk_verifier', {}))}

**CITATION AUDITOR OUTPUT:**
{self._format_citation_output(agent_outputs.get('citation_auditor', {}))}

**GAP SYNTHESIZER OUTPUT:**
{self._format_gap_output(agent_outputs.get('gap_synthesizer', {}))}

**EXECUTION METADATA:**
- Total Cost: ${state.get('total_cost_cents', 0) / 100:.2f}
- Documents Analyzed: {len(state.get('documents', []))}
- Citations: {len(state.get('citations', []))}
- Tool Executions: {tool_stats.get('total_executions', 0)}

**CITATIONS:**
{self._format_citations(state.get('citations', []))}

Generate a comprehensive Markdown report following the structure specified in the system prompt."""

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
            report = response.content[0].text

            return report

        except Exception as e:
            logger.error(f"Report generation LLM call failed: {e}", exc_info=True)
            raise

    def _format_extractor_output(self, extractor: Dict[str, Any]) -> str:
        """Format extractor output for report prompt."""
        if not extractor:
            return "No extraction performed"

        return f"""- Chunks Retrieved: {extractor.get('num_chunks_retrieved', 0)}
- Documents: {extractor.get('num_documents', 0)}
- Retrieval Method: {extractor.get('retrieval_method', 'Unknown')}"""

    def _format_classifier_output(self, classifier: Dict[str, Any]) -> str:
        """Format classifier output."""
        if not classifier:
            return "No classification performed"

        return f"""- Document Type: {classifier.get('document_type', 'Unknown')}
- Domain: {classifier.get('domain', 'Unknown')}
- Complexity: {classifier.get('complexity', 'Unknown')}
- Confidence: {classifier.get('confidence', 0)}%"""

    def _format_compliance_output(self, compliance: Dict[str, Any]) -> str:
        """Format compliance output."""
        if not compliance:
            return "No compliance check performed"

        violations = compliance.get('violations', [])
        gaps = compliance.get('gaps', [])

        return f"""- Framework: {compliance.get('framework', 'Unknown')}
- Violations Found: {len(violations)}
- Gaps Identified: {len(gaps)}
- Confidence: {compliance.get('confidence', 0)}%"""

    def _format_risk_output(self, risk: Dict[str, Any]) -> str:
        """Format risk output."""
        if not risk:
            return "No risk assessment performed"

        risks = risk.get('risks', [])

        return f"""- Risks Identified: {len(risks)}
- Overall Risk Score: {risk.get('overall_risk_score', 0)}
- Critical Items: {len(risk.get('critical_items', []))}"""

    def _format_citation_output(self, citation: Dict[str, Any]) -> str:
        """Format citation audit output."""
        if not citation:
            return "No citation audit performed"

        return f"""- Total Citations: {citation.get('total_citations', 0)}
- Verified: {citation.get('verified_citations', 0)}
- Verification Rate: {citation.get('verification_rate', 0)}%
- Quality Score: {citation.get('quality_score', 0)}%"""

    def _format_gap_output(self, gap: Dict[str, Any]) -> str:
        """Format gap analysis output."""
        if not gap:
            return "No gap analysis performed"

        gaps = gap.get('gaps', [])

        return f"""- Gaps Identified: {len(gaps)}
- Completeness Score: {gap.get('completeness_score', 0)}%
- Critical Gaps: {len(gap.get('critical_gaps', []))}"""

    def _format_citations(self, citations: list) -> str:
        """Format citations list."""
        if not citations:
            return "No citations"

        return "\n".join([f"{i+1}. {cite}" for i, cite in enumerate(citations[:20])])

    def _generate_fallback_report(
        self,
        query: str,
        agent_outputs: Dict[str, Any]
    ) -> str:
        """Generate fallback report if LLM generation fails."""
        return f"""# Analysis Report (Fallback)

## Query
{query}

## Error
Report generation encountered an error. Here is a basic summary of agent outputs:

{', '.join(agent_outputs.keys())} agents completed.

Please review logs for detailed information."""
