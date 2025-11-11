"""
Tool Adapter - Bridge between LangGraph agents and existing tools.

Translates LangGraph tool calls to existing SDK tool execution.
Zero changes required to existing tools.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from ...agent.tools.registry import get_registry as get_old_registry
from ...agent.tools.base import ToolResult
from ..core.state import ToolExecution

logger = logging.getLogger(__name__)


class ToolAdapter:
    """
    Adapts existing tool infrastructure for LangGraph agents.

    Handles:
    - Tool lookup in existing registry
    - Input validation (already done by existing tools)
    - Provider selection (already handled by existing providers)
    - Error handling
    - Result formatting
    - Execution tracking
    """

    def __init__(self, tool_registry=None):
        """
        Initialize tool adapter.

        Args:
            tool_registry: Existing tool registry (from src.agent.tools.registry)
                          If None, gets global registry
        """
        if tool_registry is None:
            tool_registry = get_old_registry()

        self.registry = tool_registry
        self.execution_history: List[ToolExecution] = []

        logger.info(f"ToolAdapter initialized with {len(self.registry)} tools")

    async def execute(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Execute a tool with LangGraph interface.

        Args:
            tool_name: Name of tool to execute
            inputs: Validated input dict
            agent_name: Name of agent executing the tool

        Returns:
            Dict with:
                - success: bool
                - data: Any (tool result data)
                - citations: List[str]
                - metadata: Dict
                - error: Optional[str]
        """
        start_time = datetime.now()

        try:
            # Get tool from existing registry
            tool = self.registry.get_tool(tool_name)

            if tool is None:
                logger.error(f"Tool not found: {tool_name}")
                return self._format_error_result(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    error=f"Tool '{tool_name}' not found",
                    start_time=start_time
                )

            # Execute tool using existing infrastructure
            # The tool.execute() method already handles:
            # - Input validation via Pydantic
            # - Error handling
            # - Result formatting
            result: ToolResult = tool.execute(**inputs)

            # Calculate execution time
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Track execution
            execution = ToolExecution(
                tool_name=tool_name,
                agent_name=agent_name,
                timestamp=start_time,
                duration_ms=duration_ms,
                input_tokens=self._estimate_tokens(inputs),
                output_tokens=result.estimated_tokens,
                success=result.success,
                error=result.error,
                result_summary=self._summarize_result(result)
            )
            self.execution_history.append(execution)

            # Convert ToolResult to dict format for LangGraph
            return {
                "success": result.success,
                "data": result.data,
                "citations": result.citations,
                "metadata": {
                    **result.metadata,
                    "agent_name": agent_name,
                    "duration_ms": duration_ms,
                    "estimated_tokens": result.estimated_tokens
                },
                "error": result.error
            }

        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name} by {agent_name}: {e}",
                exc_info=True
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=str(e),
                start_time=start_time
            )

    def _format_error_result(
        self,
        tool_name: str,
        agent_name: str,
        error: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Format error result."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Track failed execution
        execution = ToolExecution(
            tool_name=tool_name,
            agent_name=agent_name,
            timestamp=start_time,
            duration_ms=duration_ms,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=error,
            result_summary=f"Error: {error[:100]}"
        )
        self.execution_history.append(execution)

        return {
            "success": False,
            "data": None,
            "citations": [],
            "metadata": {
                "agent_name": agent_name,
                "duration_ms": duration_ms,
                "error_type": "execution_error"
            },
            "error": error
        }

    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count from input data."""
        try:
            import json
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            # Rough estimate: 4 chars per token
            return len(json_str) // 4
        except (TypeError, ValueError, OverflowError) as e:
            logger.debug(f"Token estimation failed: {type(e).__name__}: {e}")
            return 0

    def _summarize_result(self, result: ToolResult) -> str:
        """Get summary of result data (first 200 chars)."""
        try:
            import json
            data_str = json.dumps(result.data, ensure_ascii=False, default=str)
            return data_str[:200] + ("..." if len(data_str) > 200 else "")
        except (TypeError, ValueError, OverflowError) as e:
            logger.debug(f"Result summarization failed: {type(e).__name__}: {e}")
            return str(result.data)[:200]

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.registry._tools.keys())

    def get_tools_by_tier(self, tiers: List[int]) -> List[str]:
        """
        Get tool names by tier.

        Args:
            tiers: List of tier numbers (e.g., [1, 2])

        Returns:
            List of tool names in those tiers
        """
        tools = []
        for tool_name, tool in self.registry._tools.items():
            if hasattr(tool, 'tier') and tool.tier in tiers:
                tools.append(tool_name)
        return tools

    def validate_tools(self, tool_names: Set[str]) -> bool:
        """
        Validate that all tools are available.

        Args:
            tool_names: Set of tool names to validate

        Returns:
            True if all tools available, False otherwise
        """
        available = set(self.get_available_tools())
        missing = tool_names - available

        if missing:
            logger.error(f"Missing tools: {missing}")
            logger.error(f"Available tools: {available}")
            return False

        return True

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0
            }

        successful = [e for e in self.execution_history if e.success]
        failed = [e for e in self.execution_history if not e.success]

        total_duration = sum(e.duration_ms for e in self.execution_history)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in self.execution_history)

        return {
            "total_executions": len(self.execution_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.execution_history) * 100,
            "avg_duration_ms": total_duration / len(self.execution_history),
            "total_tokens": total_tokens,
            "by_agent": self._get_stats_by_agent(),
            "by_tool": self._get_stats_by_tool()
        }

    def _get_stats_by_agent(self) -> Dict[str, Dict]:
        """Get statistics grouped by agent."""
        stats = {}
        for execution in self.execution_history:
            if execution.agent_name not in stats:
                stats[execution.agent_name] = {
                    "executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0
                }

            stats[execution.agent_name]["executions"] += 1
            if execution.success:
                stats[execution.agent_name]["successful"] += 1
            else:
                stats[execution.agent_name]["failed"] += 1
            stats[execution.agent_name]["total_duration_ms"] += execution.duration_ms

        return stats

    def _get_stats_by_tool(self) -> Dict[str, Dict]:
        """Get statistics grouped by tool."""
        stats = {}
        for execution in self.execution_history:
            if execution.tool_name not in stats:
                stats[execution.tool_name] = {
                    "executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0
                }

            stats[execution.tool_name]["executions"] += 1
            if execution.success:
                stats[execution.tool_name]["successful"] += 1
            else:
                stats[execution.tool_name]["failed"] += 1
            stats[execution.tool_name]["total_duration_ms"] += execution.duration_ms

        return stats

    def clear_history(self) -> None:
        """Clear execution history (for testing)."""
        self.execution_history.clear()


# Global adapter instance
_tool_adapter: Optional[ToolAdapter] = None


def get_tool_adapter(tool_registry=None) -> ToolAdapter:
    """
    Get global tool adapter instance.

    Args:
        tool_registry: Optional tool registry

    Returns:
        ToolAdapter instance
    """
    global _tool_adapter

    if _tool_adapter is None:
        _tool_adapter = ToolAdapter(tool_registry)

    return _tool_adapter
