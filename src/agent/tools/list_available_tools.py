"""
List Available Tools Tool

Provides a complete list of all registered tools with their capabilities.
"""

import logging

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class ListAvailableToolsInput(ToolInput):
    """Input for list_available_tools tool."""

    pass  # No parameters needed


@register_tool
class ListAvailableToolsTool(BaseTool):
    """List all available tools."""

    name = "list_available_tools"
    description = "List all available tools"
    detailed_help = """
    Returns a complete list of all available tools with short descriptions.
    For detailed help on a specific tool, use get_tool_help instead.

    **When to use:**
    - Need to see all available tools
    - Understand available capabilities
    - Select the right tool for a task

    **Best practice:** Use get_tool_help for detailed docs on specific tools.
    """
    input_schema = ListAvailableToolsInput

    def execute_impl(self) -> ToolResult:
        """Get list of all available tools with metadata."""
        from ._registry import get_registry

        registry = get_registry()
        all_tools = registry.get_all_tools()

        # Build tool list with metadata
        tools_list = []
        for tool in all_tools:
            # Extract input parameters from schema
            schema = tool.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # Build parameters info
            parameters = []
            for param_name, param_info in properties.items():
                param_desc = param_info.get("description", "No description")
                param_type = param_info.get("type", "unknown")
                is_required = param_name in required

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc,
                        "required": is_required,
                    }
                )

            # Extract "when to use" from description if present
            # Some tools have "Use for:" or "Use when:" in their docstring
            when_to_use = tool.description
            if hasattr(tool.__class__, "__doc__") and tool.__class__.__doc__:
                doc = tool.__class__.__doc__.strip()
                # Look for "Use for:" or "Use when:" lines
                for line in doc.split("\n"):
                    line = line.strip()
                    if line.startswith("Use for:") or line.startswith("Use when:"):
                        when_to_use = line
                        break

            tools_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "when_to_use": when_to_use,
                }
            )

        # Sort by name for consistent ordering
        tools_list.sort(key=lambda t: t["name"])

        return ToolResult(
            success=True,
            data={
                "tools": tools_list,
                "total_count": len(tools_list),
                "best_practices": {
                    "general": [
                        "Use 'search' for most queries (hybrid + optional expansion + rerank = best quality)",
                        "Start with num_expands=0, increase to 1-2 for better recall when needed",
                        "For complex queries, decompose into sub-tasks and use multiple tools",
                        "Try multiple retrieval strategies before giving up",
                    ],
                    "selection_strategy": {
                        "most_queries": "search (with num_expands=0 initially, 1-2 for recall)",
                        "entity_focused": "Use 'search' with entity names, or multi_hop_search if KG available",
                        "specific_document": "Use search with filter_type='document', filter_value=<doc_id>",
                        "multi_hop_reasoning": "multi_hop_search (requires KG)",
                        "comparison": "multi_doc_synthesizer",
                        "temporal_info": "search with filter_type='temporal' or timeline_view",
                    },
                },
            },
            metadata={
                "total_tools": len(tools_list),
            },
        )
