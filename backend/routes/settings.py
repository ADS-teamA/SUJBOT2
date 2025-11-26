"""
User settings API endpoints.

Provides endpoints for managing user-specific settings:
- Agent variant preference (premium/local)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from ..models import AgentVariantRequest, AgentVariantResponse
from ..middleware.auth import get_current_user
from ..constants import VARIANT_CONFIG, DEFAULT_VARIANT, is_valid_variant
from .auth import get_auth_queries

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)


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

    # Defensive check: fall back to default if variant unknown
    if not is_valid_variant(variant):
        logger.warning(f"Unknown variant '{variant}' for user {current_user['id']}, using default")
        variant = DEFAULT_VARIANT

    return AgentVariantResponse(
        variant=variant,
        **VARIANT_CONFIG[variant]
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
            "display_name": "Local (Llama 3.1 70B)",
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"
        }
    """
    queries = get_auth_queries()

    try:
        await queries.update_agent_variant(current_user["id"], request.variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AgentVariantResponse(
        variant=request.variant,
        **VARIANT_CONFIG[request.variant]
    )
