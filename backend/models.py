"""
Pydantic models for API request/response validation.

These models define the contract between frontend and backend.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field


class ChatMessage(BaseModel):
    """Single chat message (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class SelectedContext(BaseModel):
    """Selected text from PDF for agent context.

    When user selects text in the PDF viewer, this context is passed
    to the agent to provide additional relevant information.
    """

    text: str = Field(
        ...,
        max_length=10000,
        description="Selected text content (max 10K chars)",
    )
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Human-readable document name")
    page_start: int = Field(..., ge=1, description="Starting page number (1-indexed)")
    page_end: int = Field(..., ge=1, description="Ending page number (1-indexed)")


class ChatRequest(BaseModel):
    """Request to send a message to the agent."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="User message (max 50K chars to prevent abuse)",
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history")
    skip_save_user_message: bool = Field(
        False,
        description="Skip saving user message to database (for regenerate where message already exists)",
    )
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="Conversation history for context (last N messages)",
    )
    selected_context: Optional[SelectedContext] = Field(
        None,
        description="Selected text from PDF viewer for additional context",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "error"]
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Available model information."""

    id: str
    name: str
    provider: Literal["anthropic", "openai", "google"]
    description: str


class ModelsResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo]
    default_model: str


class ClarificationRequest(BaseModel):
    """Request to provide clarification for interrupted workflow."""

    thread_id: str = Field(..., description="Thread ID from clarification_needed event")
    response: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's free-form clarification response",
    )


class AgentVariantRequest(BaseModel):
    """Request to update agent variant preference."""

    variant: Literal["premium", "cheap", "local"] = Field(
        ...,
        description="Agent variant: 'premium' (Opus+Sonnet), 'cheap' (Haiku), or 'local' (Llama 3.1 70B)"
    )


class AgentVariantResponse(BaseModel):
    """Response with agent variant information."""

    variant: Literal["premium", "cheap", "local"] = Field(..., description="Current variant")
    display_name: str = Field(..., description="Human-readable variant name")
    model: str = Field(..., description="Default model identifier for this variant")


class SpendingResponse(BaseModel):
    """User spending information."""

    total_spent_czk: float = Field(..., ge=0, description="Total amount spent in CZK")
    spending_limit_czk: float = Field(..., ge=0, description="Spending limit in CZK")
    remaining_czk: float = Field(..., ge=0, description="Remaining budget in CZK")
    reset_at: Optional[str] = Field(None, description="ISO timestamp of last reset")


# =============================================================================
# Admin API Models
# =============================================================================


class AdminUserResponse(BaseModel):
    """Admin view of user (includes all fields)."""

    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    agent_variant: Optional[str] = None
    created_at: str
    updated_at: str
    last_login_at: Optional[str] = None
    # Spending tracking
    spending_limit_czk: Optional[float] = Field(default=500.0, ge=0)
    total_spent_czk: float = Field(default=0.0, ge=0)
    spending_reset_at: Optional[str] = None


class AdminUserListResponse(BaseModel):
    """Paginated list of users for admin."""

    users: List[AdminUserResponse]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=1000)
    offset: int = Field(..., ge=0)


class AdminUserCreateRequest(BaseModel):
    """Request to create a new user (admin endpoint)."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password (min 8 chars)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")
    is_admin: bool = Field(False, description="Grant admin privileges")
    is_active: bool = Field(True, description="Account active status")


class AdminUserUpdateRequest(BaseModel):
    """Request to update user (admin endpoint)."""

    email: Optional[EmailStr] = Field(None, description="New email address")
    password: Optional[str] = Field(None, min_length=8, max_length=128, description="New password (min 8 chars)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")
    is_admin: Optional[bool] = Field(None, description="Admin privileges")
    is_active: Optional[bool] = Field(None, description="Account active status")
    agent_variant: Optional[Literal["premium", "cheap", "local"]] = Field(
        None, description="Model preference"
    )
    spending_limit_czk: Optional[float] = Field(
        None, ge=0, description="Spending limit in CZK"
    )


class AdminLoginRequest(BaseModel):
    """Admin login credentials."""

    email: EmailStr = Field(..., description="Admin email address")
    password: str = Field(..., min_length=8, description="Admin password")


class ServiceHealthDetail(BaseModel):
    """Individual service health status."""

    name: str = Field(..., min_length=1)
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: Optional[float] = Field(None, ge=0)
    message: Optional[str] = None


class AdminHealthResponse(BaseModel):
    """Detailed health check for admin dashboard."""

    status: Literal["healthy", "degraded", "unhealthy"]
    services: List[ServiceHealthDetail]
    timestamp: str


class AdminStatsResponse(BaseModel):
    """System statistics for admin dashboard."""

    total_users: int = Field(..., ge=0)
    active_users: int = Field(..., ge=0)
    admin_users: int = Field(..., ge=0)
    total_conversations: int = Field(..., ge=0)
    total_messages: int = Field(..., ge=0)
    users_last_24h: int = Field(..., ge=0)
    # Spending statistics
    total_spent_czk: float = Field(default=0.0, ge=0)
    avg_spent_per_message_czk: float = Field(default=0.0, ge=0)
    avg_spent_per_conversation_czk: float = Field(default=0.0, ge=0)
    timestamp: str
