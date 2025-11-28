"""
Tool Adapter - Bridge between LangGraph agents and existing tools.

Translates LangGraph tool calls to existing SDK tool execution.
Zero changes required to existing tools.

Includes hallucination detection for evaluation:
- Tracks when LLM calls non-existent tools
- Validates inputs against Pydantic schema before execution
- Categorizes errors (hallucination vs validation vs execution)
"""

import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

from pydantic import ValidationError as PydanticValidationError

from ...agent.tools import get_registry as get_old_registry, ToolResult
from ..core.state import ToolExecution, ToolUsageMetrics

logger = logging.getLogger(__name__)


class ToolErrorType(str, Enum):
    """Categorization of tool execution errors."""
    HALLUCINATION = "hallucination"    # Tool doesn't exist in registry
    VALIDATION = "validation"          # Tool exists but input invalid
    EXECUTION = "execution"            # Tool executed but failed
    TIMEOUT = "timeout"                # Tool execution timed out
    SUCCESS = "success"                # No error


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

        # Evaluation metrics tracking
        self.usage_metrics = ToolUsageMetrics()
        self._available_tools_cache: Optional[Set[str]] = None

        logger.info(f"ToolAdapter initialized with {len(self.registry)} tools")

    async def execute(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Execute a tool with LangGraph interface.

        Includes hallucination detection and input validation:
        1. Check if tool exists (hallucination detection)
        2. Validate inputs against Pydantic schema
        3. Execute tool
        4. Track metrics for evaluation

        Args:
            tool_name: Name of tool to execute
            inputs: Input dict (will be validated)
            agent_name: Name of agent executing the tool

        Returns:
            Dict with:
                - success: bool
                - data: Any (tool result data)
                - citations: List[str]
                - metadata: Dict
                - error: Optional[str]
                - error_type: ToolErrorType (for evaluation)
        """
        start_time = datetime.now()

        # Step 1: Check if tool exists (HALLUCINATION DETECTION)
        tool = self.registry.get_tool(tool_name)

        if tool is None:
            logger.warning(
                f"HALLUCINATION DETECTED: Agent '{agent_name}' called "
                f"non-existent tool '{tool_name}'"
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Tool '{tool_name}' does not exist. Available tools: {self._get_available_tools_str()}",
                start_time=start_time,
                error_type=ToolErrorType.HALLUCINATION,
                was_hallucinated=True
            )

        # Step 2: Validate inputs against Pydantic schema
        validation_error = self._validate_inputs(tool, inputs)
        if validation_error:
            logger.warning(
                f"INPUT VALIDATION FAILED: Tool '{tool_name}' by agent '{agent_name}': {validation_error}"
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Invalid inputs for tool '{tool_name}': {validation_error}",
                start_time=start_time,
                error_type=ToolErrorType.VALIDATION,
                validation_error=validation_error
            )

        # Step 3: Execute tool
        try:
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
                result_summary=self._summarize_result(result),
                was_hallucinated=False,
                validation_error=None
            )
            self.execution_history.append(execution)
            self.usage_metrics.record_execution(execution)

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
                "error": result.error,
                "error_type": ToolErrorType.SUCCESS if result.success else ToolErrorType.EXECUTION
            }

        except TimeoutError as e:
            logger.error(f"Tool timeout: {tool_name} by {agent_name}")
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Tool '{tool_name}' timed out: {e}",
                start_time=start_time,
                error_type=ToolErrorType.TIMEOUT
            )

        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name} by {agent_name}: {e}",
                exc_info=True
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=str(e),
                start_time=start_time,
                error_type=ToolErrorType.EXECUTION
            )

    def _validate_inputs(self, tool, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Validate inputs against tool's Pydantic schema.

        Args:
            tool: Tool instance with input_schema
            inputs: Input dict to validate

        Returns:
            Error message if validation failed, None if valid
        """
        if not hasattr(tool, 'input_schema') or tool.input_schema is None:
            return None  # No schema to validate against

        try:
            # Attempt to create Pydantic model instance
            tool.input_schema(**inputs)
            return None
        except PydanticValidationError as e:
            # Extract first error message for clarity
            errors = e.errors()
            if errors:
                first_error = errors[0]
                field = ".".join(str(loc) for loc in first_error.get("loc", []))
                msg = first_error.get("msg", "validation error")
                return f"{field}: {msg}"
            return str(e)
        except TypeError as e:
            # Missing required arguments
            return str(e)

    def _get_available_tools_str(self) -> str:
        """Get comma-separated list of available tools (cached)."""
        if self._available_tools_cache is None:
            self._available_tools_cache = set(self.registry._tools.keys())
        return ", ".join(sorted(self._available_tools_cache)[:10]) + "..."

    def _format_error_result(
        self,
        tool_name: str,
        agent_name: str,
        error: str,
        start_time: datetime,
        error_type: ToolErrorType = ToolErrorType.EXECUTION,
        was_hallucinated: bool = False,
        validation_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error result with evaluation metadata.

        Args:
            tool_name: Name of tool that failed
            agent_name: Agent that called the tool
            error: Error message
            start_time: When execution started
            error_type: Category of error
            was_hallucinated: True if tool doesn't exist
            validation_error: Validation error message if applicable
        """
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Track failed execution with evaluation metadata
        execution = ToolExecution(
            tool_name=tool_name,
            agent_name=agent_name,
            timestamp=start_time,
            duration_ms=duration_ms,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=error,
            result_summary=f"Error ({error_type.value}): {error[:80]}",
            was_hallucinated=was_hallucinated,
            validation_error=validation_error
        )
        self.execution_history.append(execution)
        self.usage_metrics.record_execution(execution)

        return {
            "success": False,
            "data": None,
            "citations": [],
            "metadata": {
                "agent_name": agent_name,
                "duration_ms": duration_ms,
                "error_type": error_type.value,
                "was_hallucinated": was_hallucinated,
                "validation_error": validation_error
            },
            "error": error,
            "error_type": error_type
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
        """Get execution statistics including evaluation metrics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0,
                "hallucination_rate": 0.0,
                "validation_error_rate": 0.0
            }

        successful = [e for e in self.execution_history if e.success]
        failed = [e for e in self.execution_history if not e.success]
        hallucinated = [e for e in self.execution_history if e.was_hallucinated]
        validation_errors = [e for e in self.execution_history if e.validation_error]

        total_duration = sum(e.duration_ms for e in self.execution_history)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in self.execution_history)
        total_count = len(self.execution_history)

        return {
            "total_executions": total_count,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / total_count * 100,
            "avg_duration_ms": total_duration / total_count,
            "total_tokens": total_tokens,
            # Evaluation metrics
            "hallucinated": len(hallucinated),
            "validation_errors": len(validation_errors),
            "hallucination_rate": len(hallucinated) / total_count,
            "validation_error_rate": len(validation_errors) / total_count,
            "by_agent": self._get_stats_by_agent(),
            "by_tool": self._get_stats_by_tool()
        }

    def get_usage_metrics(self) -> ToolUsageMetrics:
        """
        Get aggregated tool usage metrics for evaluation.

        Returns:
            ToolUsageMetrics instance with hallucination rate, success rate, etc.
        """
        return self.usage_metrics

    def reset_metrics(self) -> None:
        """Reset usage metrics (for testing or between runs)."""
        self.usage_metrics = ToolUsageMetrics()

    def _get_stats_by_agent(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by agent."""
        stats: Dict[str, Dict[str, Any]] = {}
        for execution in self.execution_history:
            if execution.agent_name not in stats:
                stats[execution.agent_name] = {
                    "executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0.0
                }

            stats[execution.agent_name]["executions"] += 1
            if execution.success:
                stats[execution.agent_name]["successful"] += 1
            else:
                stats[execution.agent_name]["failed"] += 1
            stats[execution.agent_name]["total_duration_ms"] += execution.duration_ms

        return stats

    def _get_stats_by_tool(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by tool."""
        stats: Dict[str, Dict[str, Any]] = {}
        for execution in self.execution_history:
            if execution.tool_name not in stats:
                stats[execution.tool_name] = {
                    "executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0.0
                }

            stats[execution.tool_name]["executions"] += 1
            if execution.success:
                stats[execution.tool_name]["successful"] += 1
            else:
                stats[execution.tool_name]["failed"] += 1
            stats[execution.tool_name]["total_duration_ms"] += execution.duration_ms

        return stats

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool schema in LLM-compatible format (Anthropic/OpenAI).

        Converts Pydantic input schema to tool calling format.

        Args:
            tool_name: Name of tool

        Returns:
            Tool schema dict with:
                - name: Tool name
                - description: Tool description
                - input_schema: JSON schema from Pydantic model
            Returns None if tool not found
        """
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            logger.warning(f"Tool not found for schema: {tool_name}")
            return None

        try:
            # Get Pydantic schema from input_schema class
            if hasattr(tool, 'input_schema') and tool.input_schema:
                # Pydantic v2 has model_json_schema() method
                if hasattr(tool.input_schema, 'model_json_schema'):
                    pydantic_schema = tool.input_schema.model_json_schema()
                else:
                    # Fallback for older Pydantic versions
                    pydantic_schema = tool.input_schema.schema()

                # Convert to Anthropic/OpenAI format
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": pydantic_schema.get("properties", {}),
                        "required": pydantic_schema.get("required", [])
                    }
                }
            else:
                # Tool has no input schema (shouldn't happen, but handle gracefully)
                logger.warning(f"Tool {tool_name} has no input_schema")
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get schema for tool {tool_name}: {e}", exc_info=True)
            return None

    def clear_history(self) -> None:
        """Clear execution history and reset metrics (for testing)."""
        self.execution_history.clear()
        self.reset_metrics()
        self._available_tools_cache = None


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
