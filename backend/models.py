"""
Pydantic models for API request/response validation.

These models define the contract between frontend and backend.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


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

    variant: Literal["premium", "local"] = Field(
        ...,
        description="Agent variant: 'premium' (Claude Haiku) or 'local' (Llama 3.1 70B)"
    )


class AgentVariantResponse(BaseModel):
    """Response with agent variant information."""

    variant: Literal["premium", "local"] = Field(..., description="Current variant")
    display_name: str = Field(..., description="Human-readable variant name")
    model: str = Field(..., description="Model identifier for this variant")
