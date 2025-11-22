"""
Get Tool Help Tool

Provides detailed documentation for any registered tool.
"""

import logging
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetToolHelpInput(ToolInput):
    """Input for get_tool_help tool."""

    tool_name: str = Field(
        ..., description="Name of tool to get help for (e.g., 'search', 'multi_doc_synthesizer')"
    )


@register_tool
class GetToolHelpTool(BaseTool):
    """Get detailed documentation for a specific tool."""

    name = "get_tool_help"
    description = "Get detailed help for any tool"
    detailed_help = """
    Returns comprehensive documentation for a specific tool including:
    - Full description and use cases
    - All parameters with types and defaults
    - Examples of when to use this tool

    Use this whenever you need to understand a tool's capabilities or parameters
    before using it for the first time.
    """
    input_schema = GetToolHelpInput

    def execute_impl(self, tool_name: str) -> ToolResult:
        """Get detailed help for a tool."""
        from ._registry import get_registry

        registry = get_registry()

        # Check if tool exists
        if tool_name not in registry._tool_classes:
            available_tools = sorted(registry._tool_classes.keys())
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools[:10])}...",
                metadata={"requested_tool": tool_name, "available_count": len(available_tools)},
            )

        # Get tool class
        tool_class = registry._tool_classes[tool_name]

        # Build detailed help
        help_text = f"# {tool_class.name}\n\n"

        # Description
        help_text += f"**Description:** {tool_class.description}\n\n"

        # Detailed help if available
        if tool_class.detailed_help:
            help_text += f"**Details:**\n{tool_class.detailed_help.strip()}\n\n"

        # Parameters from Pydantic schema
        if tool_class.input_schema and tool_class.input_schema != ToolInput:
            help_text += "**Parameters:**\n"
            schema = tool_class.input_schema.model_json_schema()

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "No description")
                is_required = "✓ Required" if param_name in required else "Optional"
                default = param_info.get("default", "N/A")

                help_text += f"- `{param_name}` ({param_type}) - {is_required}\n"
                help_text += f"  {param_desc}\n"
                if default != "N/A":
                    help_text += f"  Default: {default}\n"
                help_text += "\n"

        # Requirements
        requirements = []
        if tool_class.requires_kg:
            requirements.append("Knowledge Graph")
        if tool_class.requires_reranker:
            requirements.append("Reranker")

        if requirements:
            help_text += f"**Requires:** {', '.join(requirements)}\n\n"

        return ToolResult(
            success=True,
            data={
                "tool_name": tool_name,
                "help_text": help_text,
                "short_description": tool_class.description,
                "requires_kg": tool_class.requires_kg,
                "requires_reranker": tool_class.requires_reranker,
            },
            metadata={"tool": tool_name},
        )
