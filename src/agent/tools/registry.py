"""
Tool Registry

Manages registration and discovery of all RAG tools.
Provides Claude SDK tool definitions.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BaseTool, ToolResult
from src.storage import VectorStoreAdapter

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all RAG tools.

    Pattern: Registry + Dependency Injection
    - Tools register themselves (via initialization or decorator)
    - Registry manages tool lifecycle
    - Provides Claude SDK tool definitions
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._unavailable_tools: Dict[str, str] = {}  # tool_name -> reason

    def register_tool_class(self, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """
        Register a tool class (for later instantiation).

        Args:
            tool_class: BaseTool subclass

        Returns:
            Same tool class (for decorator pattern)
        """
        tool_name = tool_class.name
        self._tool_classes[tool_name] = tool_class
        logger.debug(f"Registered tool class: {tool_name}")
        return tool_class

    def initialize_tools(
        self,
        vector_store: VectorStoreAdapter,
        embedder,
        reranker=None,
        graph_retriever=None,
        knowledge_graph=None,
        context_assembler=None,
        config=None,
    ) -> None:
        """
        Initialize all registered tool classes with dependencies.

        This creates tool instances by injecting pipeline components.

        Args:
            vector_store: VectorStoreAdapter (FAISS or PostgreSQL backend)
            embedder: EmbeddingGenerator
            reranker: CrossEncoderReranker (optional)
            graph_retriever: GraphEnhancedRetriever (optional)
            knowledge_graph: KnowledgeGraph (optional)
            context_assembler: ContextAssembler (optional)
            config: ToolConfig
        """
        for tool_name, tool_class in self._tool_classes.items():
            # Check requirements
            if tool_class.requires_kg and not knowledge_graph:
                reason = "Requires knowledge graph (use --kg option)"
                logger.warning(
                    f"Tool '{tool_name}' requires knowledge graph but none provided. Skipping. "
                    f"Run indexing with ENABLE_KNOWLEDGE_GRAPH=true to generate: "
                    f"uv run python run_pipeline.py data/your_docs/"
                )
                self._unavailable_tools[tool_name] = reason
                continue

            if tool_class.requires_reranker and not reranker:
                logger.warning(
                    f"Tool '{tool_name}' requires reranker but none provided. "
                    f"Tool will use base retrieval."
                )

            # Instantiate tool
            tool_instance = tool_class(
                vector_store=vector_store,
                embedder=embedder,
                reranker=reranker,
                graph_retriever=graph_retriever,
                knowledge_graph=knowledge_graph,
                context_assembler=context_assembler,
                config=config,
            )

            self._tools[tool_name] = tool_instance
            logger.info(f"Initialized tool: {tool_name} (Tier {tool_class.tier})")

        # Log summary
        available_count = len(self._tools)
        unavailable_count = len(self._unavailable_tools)
        total_count = len(self._tool_classes)

        logger.info(f"Tool registry initialized: {available_count}/{total_count} tools available")

        if unavailable_count > 0:
            logger.warning(
                f"{unavailable_count} tools unavailable: {list(self._unavailable_tools.keys())}"
            )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get initialized tool instance by name.

        Args:
            name: Tool name

        Returns:
            BaseTool instance or None if not found
        """
        return self._tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name with input validation.

        Args:
            name: Tool name
            **kwargs: Tool input parameters

        Returns:
            ToolResult
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {name}",
                metadata={"available_tools": list(self._tools.keys())},
            )

        return tool.execute(**kwargs)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all initialized tool instances."""
        return list(self._tools.values())

    def get_tools_by_tier(self, tier: int) -> List[BaseTool]:
        """
        Get tools by tier level.

        Args:
            tier: 1=basic, 2=advanced, 3=analysis

        Returns:
            List of tools in that tier
        """
        return [tool for tool in self._tools.values() if tool.tier == tier]

    def get_unavailable_tools(self) -> Dict[str, str]:
        """
        Get list of unavailable tools and reasons.

        Returns:
            Dict mapping tool names to unavailability reasons
        """
        return self._unavailable_tools.copy()

    def get_claude_sdk_tools(self) -> List[Dict[str, Any]]:
        """
        Get Claude SDK tool definitions for all tools.

        Returns:
            List of tool definition dicts for Claude SDK
        """
        return [tool.get_claude_sdk_definition() for tool in self._tools.values()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for all tools.

        Returns:
            Dict with statistics per tool and overall stats
        """
        tool_stats = [tool.get_stats() for tool in self._tools.values()]

        total_calls = sum(s["execution_count"] for s in tool_stats)
        total_errors = sum(s["error_count"] for s in tool_stats)
        total_time = sum(s["total_time_ms"] for s in tool_stats)

        # Group by tier
        tier_stats = {}
        for tier in [1, 2, 3]:
            tier_tools = self.get_tools_by_tier(tier)
            tier_calls = sum(t.execution_count for t in tier_tools)
            tier_stats[f"tier{tier}_calls"] = tier_calls

        return {
            "total_tools": len(self._tools),
            "total_registered": len(self._tool_classes),
            "unavailable_tools": len(self._unavailable_tools),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(total_time / total_calls, 2) if total_calls > 0 else 0,
            "success_rate": (
                round((total_calls - total_errors) / total_calls * 100, 1)
                if total_calls > 0
                else 100.0
            ),
            "tier_distribution": tier_stats,
            "tools": tool_stats,
            "unavailable": self._unavailable_tools,
        }

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    return _registry


def register_tool(tool_class: Type[BaseTool]) -> Type[BaseTool]:
    """
    Decorator to register a tool class.

    Usage:
        @register_tool
        class MyTool(BaseTool):
            name = "my_tool"
            ...
    """
    return _registry.register_tool_class(tool_class)
