"""
LangGraph State Schema

Comprehensive state definition for multi-agent workflow.
Ensures type safety, validation, and consistency across all 8 agents.
"""

from typing import Annotated, Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    """Classification of user query."""
    SIMPLE_SEARCH = "simple_search"       # Single doc lookup
    CROSS_DOC_ANALYSIS = "cross_doc"      # Compare multiple docs
    COMPLIANCE_CHECK = "compliance"        # Regulatory compliance
    RISK_ASSESSMENT = "risk"               # Risk analysis
    SYNTHESIS = "synthesis"                # Knowledge synthesis
    REPORTING = "reporting"                # Generate report
    UNKNOWN = "unknown"                    # Uncategorized


class ExecutionPhase(str, Enum):
    """Workflow execution phase."""
    ROUTING = "routing"              # Determine complexity
    EXTRACTION = "extraction"        # Extract from docs
    CLASSIFICATION = "classification" # Classify content
    VERIFICATION = "verification"    # Verify accuracy
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
    """

    # === INPUT ===
    query: str = Field(..., description="Original user query")

    # === ROUTING ===
    query_type: QueryType = QueryType.UNKNOWN
    complexity_score: int = Field(default=0, ge=0, le=100)
    execution_phase: ExecutionPhase = ExecutionPhase.ROUTING
    agent_sequence: List[str] = Field(default_factory=list)

    # === EXECUTION ===
    current_agent: Optional[str] = None
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    tool_executions: List[ToolExecution] = Field(default_factory=list)

    # === RETRIEVAL ===
    documents: List[DocumentMetadata] = Field(default_factory=list)
    retrieved_text: str = ""

    # === CONTEXT ===
    shared_context: Dict[str, Any] = Field(default_factory=dict)

    # === RESULTS ===
    final_answer: Optional[str] = None
    structured_output: Dict[str, Any] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None

    # === COST TRACKING ===
    total_cost_cents: float = 0.0
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)

    # === ERROR HANDLING ===
    errors: List[str] = Field(default_factory=list)

    # === METADATA ===
    session_id: str = Field(default="default")
    user_id: str = Field(default="default")
    created_at: datetime = Field(default_factory=datetime.now)
    checkpoints: List[str] = Field(default_factory=list)  # Checkpoint IDs

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
        self.agent_path.append(agent_name) if agent_name not in self.agent_path else None

    def add_tool_execution(self, execution: ToolExecution) -> None:
        """Record a tool execution."""
        self.tool_executions.append(execution)

        # Update cost tracking
        agent_cost = self.cost_breakdown.get(execution.agent_name, 0.0)
        # Estimate cost: rough approximation for tracking
        token_cost = (execution.input_tokens + execution.output_tokens) * 0.000003  # $3/1M tokens
        self.cost_breakdown[execution.agent_name] = agent_cost + token_cost
        self.total_cost_cents += token_cost * 100

    def update_execution_phase(self, phase: ExecutionPhase) -> None:
        """Update the current execution phase."""
        self.execution_phase = phase

    def get_agent_history(self) -> List[str]:
        """Get list of agents that have processed this query."""
        return self.agent_path

    def is_error_state(self) -> bool:
        """Check if workflow is in error state."""
        return len(self.errors) > 0 or self.execution_phase == ExecutionPhase.ERROR


# Type aliases for cleaner code
State = MultiAgentState  # Shorthand alias
