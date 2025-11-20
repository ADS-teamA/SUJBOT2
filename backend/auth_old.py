"""
Backend Authentication Module for SUJBOT2

JWT-based authentication with configurable credentials.
In production, integrate with your organization's SSO/LDAP.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_USE_RANDOM_STRING")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Hardcoded credentials (for demo/development)
# In production: Replace with LDAP/SSO/database validation
VALID_CREDENTIALS = {
    "username": os.getenv("AUTH_USERNAME", "admin"),
    "password": os.getenv("AUTH_PASSWORD", "adssujbot"),
}

# HTTP Bearer token scheme
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request payload."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """JWT token payload."""
    username: str
    exp: datetime


def create_access_token(username: str) -> tuple[str, datetime]:
    """
    Create JWT access token.

    Args:
        username: Username to encode in token

    Returns:
        Tuple of (token, expiration_datetime)
    """
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    payload = {
        "sub": username,  # "sub" (subject) is standard JWT claim
        "exp": expires_at,  # "exp" (expiration) is standard JWT claim
        "iat": datetime.now(timezone.utc),  # "iat" (issued at)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Created access token for user: {username}, expires: {expires_at}")

    return token, expires_at


def verify_password(username: str, password: str) -> bool:
    """
    Verify username and password.

    In production: Replace with LDAP/SSO/database validation.

    Args:
        username: Username
        password: Password

    Returns:
        True if credentials are valid
    """
    return (
        username == VALID_CREDENTIALS["username"] and
        password == VALID_CREDENTIALS["password"]
    )


def decode_token(token: str) -> str:
    """
    Decode and verify JWT token.

    Args:
        token: JWT token string

    Returns:
        Username from token

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return username

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    FastAPI dependency to extract and validate JWT token from request.

    Usage:
        @app.get("/protected")
        async def protected_route(username: str = Depends(get_current_user)):
            return {"user": username}

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        Username from valid token

    Raises:
        HTTPException: If token is missing, invalid, or expired
    """
    token = credentials.credentials
    username = decode_token(token)
    logger.debug(f"Authenticated user: {username}")
    return username
