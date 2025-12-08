"""
User settings API endpoints.

Provides endpoints for managing user-specific settings:
- Agent variant preference (premium/cheap/local)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from ..models import AgentVariantRequest, AgentVariantResponse
from ..middleware.auth import get_current_user
from ..constants import VARIANT_CONFIG, DEFAULT_VARIANT, is_valid_variant
from .auth import get_auth_queries

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)


def _build_variant_response(variant: str) -> AgentVariantResponse:
    """Build AgentVariantResponse from variant config."""
    config = VARIANT_CONFIG[variant]
    return AgentVariantResponse(
        variant=variant,
        display_name=config["display_name"],
        model=config["default_model"],  # Return default tier model for display
    )


@router.get("/agent-variant", response_model=AgentVariantResponse)
async def get_agent_variant(current_user: dict = Depends(get_current_user)):
    """
    Get current user's agent variant preference.

    Returns:
        AgentVariantResponse with variant, display_name, and model

    Example response:
        {
            "variant": "cheap",
            "display_name": "Cheap (Haiku 4.5)",
            "model": "claude-haiku-4-5-20251001"
        }
    """
    queries = get_auth_queries()
    variant = await queries.get_agent_variant(current_user["id"])

    # Defensive check: fall back to default if variant unknown
    if not is_valid_variant(variant):
        logger.warning(f"Unknown variant '{variant}' for user {current_user['id']}, using default")
        variant = DEFAULT_VARIANT

    return _build_variant_response(variant)


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
        HTTPException(403): If non-admin tries to select premium

    Example request:
        {"variant": "premium"}

    Example response:
        {
            "variant": "premium",
            "display_name": "Premium (Opus + Sonnet)",
            "model": "claude-sonnet-4-5-20250929"
        }
    """
    # Premium variant requires admin privileges
    if request.variant == "premium" and not current_user.get("is_admin", False):
        logger.warning(f"Non-admin user {current_user['id']} attempted to select premium variant")
        raise HTTPException(
            status_code=403,
            detail="Premium models are only available to admin users"
        )

    queries = get_auth_queries()

    try:
        await queries.update_agent_variant(current_user["id"], request.variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _build_variant_response(request.variant)
