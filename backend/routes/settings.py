"""
User settings API endpoints.

Provides endpoints for managing user-specific settings:
- Agent variant preference (premium/local)
"""

from fastapi import APIRouter, Depends, HTTPException
from ..models import AgentVariantRequest, AgentVariantResponse
from ..middleware.auth import get_current_user
from .auth import get_auth_queries

router = APIRouter(prefix="/settings", tags=["settings"])

# Variant configuration
VARIANT_INFO = {
    "premium": {
        "display_name": "Premium (Claude Haiku)",
        "model": "claude-haiku-4-5"
    },
    "local": {
        "display_name": "Local (Llama 3.1 70B)",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }
}


@router.get("/agent-variant", response_model=AgentVariantResponse)
async def get_agent_variant(current_user: dict = Depends(get_current_user)):
    """
    Get current user's agent variant preference.

    Returns:
        AgentVariantResponse with variant, display_name, and model

    Example response:
        {
            "variant": "premium",
            "display_name": "Premium (Claude Haiku)",
            "model": "claude-haiku-4-5"
        }
    """
    queries = get_auth_queries()
    variant = await queries.get_agent_variant(current_user["id"])

    return AgentVariantResponse(
        variant=variant,
        **VARIANT_INFO[variant]
    )


@router.post("/agent-variant", response_model=AgentVariantResponse)
async def update_agent_variant(
    request: AgentVariantRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user's agent variant preference.

    Args:
        request: AgentVariantRequest with variant field

    Returns:
        AgentVariantResponse with updated variant information

    Raises:
        HTTPException(400): If variant is invalid

    Example request:
        {"variant": "local"}

    Example response:
        {
            "variant": "local",
            "display_name": "Local (Qwen 72B)",
            "model": "Qwen/Qwen2.5-72B-Instruct"
        }
    """
    queries = get_auth_queries()

    try:
        await queries.update_agent_variant(current_user["id"], request.variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AgentVariantResponse(
        variant=request.variant,
        **VARIANT_INFO[request.variant]
    )
