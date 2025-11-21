"""
Base Tool Abstraction

Provides lightweight abstraction for all RAG tools with:
- Input validation via Pydantic
- Error handling
- Execution statistics
- Result formatting
"""

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


def _convert_enums_recursive(obj: Any) -> Any:
    """
    Recursively convert enum keys and values to strings for JSON serialization.

    Handles the case where dict keys are enum types (which json.dumps can't handle).
    Also converts enum values to their string representation.

    Args:
        obj: Object to convert (dict, list, enum, or primitive)

    Returns:
        Object with all enums converted to strings
    """
    from enum import Enum

    if isinstance(obj, Enum):
        # Convert enum to its string value
        return obj.value
    elif isinstance(obj, dict):
        # Convert both keys and values recursively
        return {
            (k.value if isinstance(k, Enum) else k): _convert_enums_recursive(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        # Convert list/tuple elements recursively
        return type(obj)(_convert_enums_recursive(item) for item in obj)
    else:
        # Return primitives as-is
        return obj


def estimate_tokens_from_result(result_data: Any) -> int:
    """
    Estimate token count from tool result data.

    Uses JSON serialization + character count / 4 heuristic.
    This is an approximation - actual tokenization depends on the model.

    Args:
        result_data: Tool result data (any JSON-serializable type)

    Returns:
        Estimated token count
    """
    try:
        # Convert enums to strings recursively (defensive - handles enum keys/values)
        # This prevents "keys must be str, int, float, bool or None, not EntityType" errors
        safe_data = _convert_enums_recursive(result_data)

        # Serialize to JSON string
        json_str = json.dumps(safe_data, ensure_ascii=False, default=str)

        # Estimate tokens: ~4 chars per token (approximation)
        # Actual ratio varies: 3-4 for English, 4-6 for code/JSON
        # Using ceil for conservative estimate (rounds up)
        estimated_tokens = math.ceil(len(json_str) / 4.0)

        return max(estimated_tokens, 1)  # Minimum 1 token
    except (TypeError, ValueError) as e:
        # Only catch serialization errors, not programming bugs
        logger.error(f"Failed to estimate tokens from result: {e}")
        return 0


class ToolInput(BaseModel):
    """
    Base input validation using Pydantic.

    All tool inputs inherit from this for automatic validation.
    """

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields


@dataclass
class ToolResult:
    """
    Standardized tool execution result.

    All tools return this format for consistency.
    """

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    estimated_tokens: int = 0  # Estimated token count of result data
    api_cost_usd: float = 0.0  # Estimated API cost in USD

    def __post_init__(self):
        """Validate ToolResult invariants."""
        # Validate execution time
        if self.execution_time_ms < 0:
            raise ValueError(f"Execution time cannot be negative: {self.execution_time_ms}")

        # Validate API cost
        if self.api_cost_usd < 0:
            raise ValueError(f"API cost cannot be negative: {self.api_cost_usd}")

        # Validate success/error relationship
        if self.success and self.error is not None:
            raise ValueError("Successful results cannot have errors")
        if not self.success and not self.error:
            raise ValueError("Failed results must have an error message")


class BaseTool(ABC):
    """
    Lightweight base class for all RAG tools.

    Provides:
    - Input validation (via Pydantic schemas)
    - Error handling (try/catch with graceful degradation)
    - Execution statistics (call count, avg time)
    - Result formatting (consistent ToolResult structure)

    Subclasses implement:
    - name: Tool identifier
    - description: What the tool does
    - tier: 1=basic, 2=advanced, 3=analysis
    - input_schema: Pydantic model for input validation
    - execute_impl(): Tool-specific logic
    """

    # Class attributes (override in subclasses)
    name: str = "base_tool"
    description: str = "Base tool (override in subclass)"  # Short description (API)
    detailed_help: str = ""  # Detailed help text (for get_tool_help)
    tier: int = 1
    input_schema: type[ToolInput] = ToolInput

    # Metadata flags
    requires_kg: bool = False  # Requires knowledge graph
    requires_reranker: bool = False  # Requires reranker

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        reranker: Any = None,
        graph_retriever: Any = None,
        knowledge_graph: Any = None,
        context_assembler: Any = None,
        llm_provider: Any = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize tool with dependencies.

        Args:
            vector_store: Vector store adapter
            embedder: Embedding generator
            reranker: Reranker (optional)
            graph_retriever: Graph retriever (optional)
            knowledge_graph: Knowledge graph (optional)
            context_assembler: Context assembler (optional)
            llm_provider: LLM provider for synthesis/HyDE (optional)
            config: Tool configuration (optional)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.graph_retriever = graph_retriever
        self.knowledge_graph = knowledge_graph
        self.context_assembler = context_assembler
        self.llm_provider = llm_provider
        self.config = config or {}

        # Statistics
        self.execution_count = 0
        self.total_time_ms = 0.0
        self.error_count = 0

    @abstractmethod
    def execute_impl(self, **kwargs) -> ToolResult:
        """
        Tool-specific execution logic.

        Args:
            **kwargs: Validated input parameters

        Returns:
            ToolResult with execution results
        """
        pass

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with validation and error handling.

        Flow:
        1. Validate inputs via Pydantic schema
        2. Execute tool logic with timing
        3. Track statistics and API costs
        4. Handle errors gracefully

        Args:
            **kwargs: Tool input parameters

        Returns:
            ToolResult
        """
        start_time = time.time()

        # Track API cost delta (before tool execution)
        from ...cost_tracker import get_global_tracker
        cost_tracker = get_global_tracker()
        cost_before = cost_tracker.get_total_cost()

        try:
            # Track which parameters were explicitly provided by the model (before validation)
            explicit_params = list(kwargs.keys())

            # Validate inputs
            validated_input = self.input_schema(**kwargs)
            validated_dict = validated_input.model_dump()

            # Execute tool logic
            result = self.execute_impl(**validated_dict)

            # Calculate API cost delta (after tool execution)
            cost_after = cost_tracker.get_total_cost()
            api_cost_delta = max(0.0, cost_after - cost_before)
            result.api_cost_usd = api_cost_delta

            # Track statistics
            elapsed_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_time_ms += elapsed_ms
            result.execution_time_ms = elapsed_ms
            result.metadata["tool_name"] = self.name
            result.metadata["tier"] = self.tier
            result.metadata["explicit_params"] = explicit_params  # Track model-specified params
            result.metadata["api_cost_usd"] = api_cost_delta

            # Estimate token count from result data
            result.estimated_tokens = estimate_tokens_from_result(result.data)
            result.metadata["estimated_tokens"] = result.estimated_tokens

            logger.info(
                f"Tool '{self.name}' executed in {elapsed_ms:.0f}ms "
                f"(success={result.success}, ~{result.estimated_tokens} tokens, ${api_cost_delta:.6f})"
            )

            return result

        except ValidationError as e:
            # Pydantic validation errors - user input issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.warning(f"Tool '{self.name}' validation failed: {e}")

            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid input: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "validation",
                },
            )

        except (KeyError, AttributeError, IndexError, TypeError) as e:
            # Programming errors - these are bugs in tool implementation
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(
                f"Tool '{self.name}' implementation error: {e}",
                exc_info=True,
                extra={"kwargs": kwargs},
            )

            return ToolResult(
                success=False,
                data=None,
                error=f"Internal tool error - this is a bug. {type(e).__name__}: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "programming",
                },
            )

        except (OSError, RuntimeError, MemoryError) as e:
            # System errors - resource issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(f"Tool '{self.name}' system error: {e}", exc_info=True)

            return ToolResult(
                success=False,
                data=None,
                error=f"System error: {type(e).__name__}: {str(e)}. Try again or contact administrator.",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "system",
                },
            )

        except Exception as e:
            # Unexpected errors - catch-all for unknown issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(
                f"Tool '{self.name}' unexpected error: {type(e).__name__}: {e}",
                exc_info=True,
                extra={"kwargs": kwargs},
            )

            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "unexpected",
                },
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        avg_time = self.total_time_ms / self.execution_count if self.execution_count > 0 else 0
        success_rate = (
            (self.execution_count - self.error_count) / self.execution_count
            if self.execution_count > 0
            else 0
        )

        return {
            "name": self.name,
            "tier": self.tier,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "success_rate": round(success_rate * 100, 1),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
        }

    def get_claude_sdk_definition(self) -> Dict[str, Any]:
        """
        Get Claude SDK tool definition.

        Returns:
            Tool definition dict for Claude SDK
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }

