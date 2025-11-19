"""
LangGraph State Schema

Comprehensive state definition for multi-agent workflow.
Ensures type safety, validation, and consistency across all 8 agents.
"""

from typing import Annotated, Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import operator

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# REDUCER FUNCTIONS (for parallel execution / fan-in)
# ============================================================================

def keep_first(existing: Any, new: Any) -> Any:
    """
    Reducer that keeps the first non-empty value.

    Used for immutable fields like 'query' that should not change after initial set.
    Treats None, empty strings, and default Enum values as "no value".
    """
    from enum import Enum as EnumBase

    # Check if existing is a default Enum value (UNKNOWN or ROUTING)
    if isinstance(existing, EnumBase):
        # For Enums, check if it's a default/placeholder value
        if existing.value in ('unknown', 'routing'):
            return new if new != existing else existing
        # Otherwise keep existing value (already set by previous agent)
        return existing

    # For strings, None, etc: treat None and empty string as "no value"
    if not existing:
        return new
    return existing


def take_max(existing: Any, new: Any) -> Any:
    """
    Reducer that returns the maximum of two values.

    Handles None values by treating them as negative infinity.
    Used for numeric fields where we want to keep the highest value (e.g., complexity_score).
    """
    if existing is None and new is None:
        return None
    if existing is None:
        return new
    if new is None:
        return existing
    return max(existing, new)


def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reducer that merges dictionaries (new values override existing).

    Used for agent_outputs, shared_context, etc.
    """
    if existing is None:
        return new
    if new is None:
        return existing
    return {**existing, **new}


def merge_lists_unique(existing: List[Any], new: List[Any]) -> List[Any]:
    """
    Reducer that merges lists and removes duplicates (preserves order).

    Used for agent_sequence to avoid duplicate agent names.
    """
    if existing is None:
        existing = []
    if new is None:
        new = []

    # Combine and deduplicate while preserving order
    seen = set()
    result = []
    for item in existing + new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class QueryType(str, Enum):
    """Classification of user query."""
    SIMPLE_SEARCH = "simple_search"       # Single doc lookup
    CROSS_DOC_ANALYSIS = "cross_doc"      # Compare multiple docs
    COMPLIANCE_CHECK = "compliance"        # Regulatory compliance (always includes requirement extraction → verification)
    RISK_ASSESSMENT = "risk"               # Risk analysis
    SYNTHESIS = "synthesis"                # Knowledge synthesis
    REPORTING = "reporting"                # Generate report
    UNKNOWN = "unknown"                    # Uncategorized


class ExecutionPhase(str, Enum):
    """Workflow execution phase."""
    ROUTING = "routing"              # Determine complexity
    AGENT_EXECUTION = "agent_execution"  # Agent executing task
    EXTRACTION = "extraction"        # Extract from docs
    CLASSIFICATION = "classification" # Classify content
    REQUIREMENT_EXTRACTION = "requirement_extraction"  # Extract atomic legal requirements
    VERIFICATION = "verification"    # Verify accuracy (compliance checking)
    SYNTHESIS = "synthesis"          # Synthesize results
    REPORTING = "reporting"          # Generate report
    COMPLETE = "complete"            # Workflow complete
    ERROR = "error"                  # Error state


class DocumentMetadata(BaseModel):
    """Metadata for a retrieved document."""
    doc_id: str
    filename: str
    layer: int                       # Layer from FAISS (1-3)
    relevance_score: float
    chunk_index: Optional[int] = None
    section_path: Optional[str] = None


class ToolExecution(BaseModel):
    """Record of a single tool execution."""
    tool_name: str
    agent_name: str
    timestamp: datetime
    duration_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    error: Optional[str] = None
    result_summary: str             # First 200 chars of result


class AgentState(BaseModel):
    """
    Base state shared with all agents via LangGraph.

    This is the minimal state that flows through the graph.
    Agents can extend this with agent-specific fields.
    """

    # === Routing ===
    query: str                      # Original user query
    complexity_score: int = 0       # 0-100 score of query complexity
    agent_path: List[str] = []      # Agent execution path
    current_agent: Optional[str] = None

    # === Context ===
    context: Dict[str, Any] = {}    # Shared context across agents
    documents: List[str] = []       # Retrieved documents
    sections: List[Dict] = []       # Extracted sections

    # === Results ===
    tool_results: Dict[str, Any] = {}  # Tool execution results
    agent_outputs: Dict[str, Any] = {} # Per-agent outputs

    # === Metadata ===
    created_at: datetime = Field(default_factory=datetime.now)
    checkpoints: List[Dict] = []    # State checkpoints for recovery
    errors: List[str] = []          # Error tracking
    cost_tokens: Dict[str, int] = {} # Cost tracking per agent


class MultiAgentState(BaseModel):
    """
    Comprehensive state for multi-agent workflow.

    Passed through LangGraph and updated by each agent.
    This is the complete state with all fields needed for the full pipeline.

    IMPORTANT: Fields use Annotated with reducer functions to support parallel execution.
    When multiple agents run in parallel (fan-out/fan-in), LangGraph needs to know how
    to merge their state updates. Reducers define the merge strategy per field.
    """

    # === INPUT (immutable after initial set) ===
    # NOTE: Default empty string allows LangGraph to accept partial state updates
    # The initial query will be set by the first state update via keep_first reducer
    query: Annotated[str, keep_first] = Field(default="", description="Original user query")

    # === ROUTING ===
    query_type: Annotated[QueryType, keep_first] = QueryType.UNKNOWN  # Orchestrator sets this once
    complexity_score: Annotated[int, take_max] = Field(default=0, ge=0, le=100)  # Keep highest score
    execution_phase: Annotated[ExecutionPhase, keep_first] = ExecutionPhase.ROUTING  # Set once, don't override
    agent_sequence: Annotated[List[str], merge_lists_unique] = Field(default_factory=list)

    # === EXECUTION ===
    current_agent: Annotated[Optional[str], keep_first] = None
    agent_outputs: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    tool_executions: Annotated[List[ToolExecution], operator.add] = Field(default_factory=list)

    # === RETRIEVAL ===
    documents: Annotated[List[DocumentMetadata], operator.add] = Field(default_factory=list)
    retrieved_text: Annotated[str, operator.add] = ""  # Concatenate text from multiple agents

    # === CONTEXT ===
    shared_context: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)

    # === RESULTS ===
    final_answer: Annotated[Optional[str], keep_first] = None  # First agent to set final answer wins
    structured_output: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    citations: Annotated[List[str], operator.add] = Field(default_factory=list)
    confidence_score: Annotated[Optional[float], take_max] = None  # Keep highest confidence

    # === COST TRACKING ===
    total_cost_cents: Annotated[float, operator.add] = 0.0  # Sum costs from all agents
    cost_breakdown: Annotated[Dict[str, float], merge_dicts] = Field(default_factory=dict)

    # === ERROR HANDLING ===
    errors: Annotated[List[str], operator.add] = Field(default_factory=list)

    # === METADATA ===
    session_id: Annotated[str, keep_first] = Field(default="default")  # Immutable session identifier
    user_id: Annotated[str, keep_first] = Field(default="default")  # Immutable user identifier
    created_at: Annotated[datetime, keep_first] = Field(default_factory=datetime.now)  # Immutable timestamp
    checkpoints: Annotated[List[str], operator.add] = Field(default_factory=list)  # Checkpoint IDs accumulate

    # === HUMAN-IN-THE-LOOP (CLARIFICATIONS) ===
    quality_check_required: Annotated[bool, operator.or_] = False  # True if ANY agent needs clarification
    quality_issues: Annotated[List[str], operator.add] = Field(default_factory=list)
    quality_metrics: Annotated[Optional[Dict[str, float]], merge_dicts] = None
    clarifying_questions: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    original_query: Annotated[Optional[str], keep_first] = None
    user_clarification: Annotated[Optional[str], keep_first] = None
    enriched_query: Annotated[Optional[str], keep_first] = None
    clarification_round: Annotated[int, take_max] = 0  # Track multi-round clarifications
    awaiting_user_input: Annotated[bool, operator.or_] = False  # Signal frontend to pause

    # === INTERNAL INFRASTRUCTURE (excluded from serialization) ===
    # EventBus for real-time progress streaming (NOT persisted to checkpoints)
    # Note: Pydantic V2 doesn't allow field names with leading underscores,
    # so we use "event_bus" (no underscore) consistently throughout the codebase.
    event_bus: Annotated[Optional[Any], keep_first] = Field(default=None, exclude=True)

    @field_validator("complexity_score")
    @classmethod
    def validate_complexity(cls, v):
        """Validate complexity score is in valid range."""
        if not 0 <= v <= 100:
            raise ValueError("Complexity score must be 0-100")
        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence score is in valid range."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be 0.0-1.0")
        return v

    def add_error(self, error_message: str) -> None:
        """Add an error to the error list."""
        self.errors.append(f"[{datetime.now().isoformat()}] {error_message}")

    def add_agent_output(self, agent_name: str, output: Any) -> None:
        """Record output from an agent."""
        self.agent_outputs[agent_name] = output
        if agent_name not in self.agent_sequence:
            self.agent_sequence.append(agent_name)

    def add_tool_execution(self, execution: ToolExecution) -> None:
        """
        Record a tool execution.

        NOTE: Cost tracking removed from this method (2025-11).
        Use global CostTracker instead for accurate model-specific pricing.
        The total_cost_cents field is now populated by runner/backend from CostTracker.
        """
        self.tool_executions.append(execution)

    def update_execution_phase(self, phase: ExecutionPhase) -> None:
        """Update the current execution phase."""
        self.execution_phase = phase

    def get_agent_history(self) -> List[str]:
        """Get list of agents that have processed this query."""
        return self.agent_sequence

    def is_error_state(self) -> bool:
        """Check if workflow is in error state."""
        return len(self.errors) > 0 or self.execution_phase == ExecutionPhase.ERROR


# Type aliases for cleaner code
State = MultiAgentState  # Shorthand alias
