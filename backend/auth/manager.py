"""
Authentication Manager - Single Source of Truth for Auth Operations

Handles:
- JWT token generation/validation (stateless auth)
- Argon2 password hashing (OWASP recommended)
- Token expiration management (24h default)

Usage:
    auth_manager = AuthManager(secret_key=os.getenv("AUTH_SECRET_KEY"))

    # Hash password on registration
    password_hash = auth_manager.hash_password("user_password")

    # Verify password on login
    is_valid = auth_manager.verify_password("user_password", password_hash)

    # Create JWT token
    token = auth_manager.create_token(user_id=123, email="user@example.com")

    # Validate JWT token
    payload = auth_manager.validate_token(token)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import jwt
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from pydantic import BaseModel


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    user_id: int
    email: str
    exp: datetime
    iat: datetime


class AuthManager:
    """
    Authentication manager using JWT (stateless) + Argon2 (password hashing).

    Security features:
    - Argon2id algorithm (PHC winner, GPU-resistant)
    - JWT with HS256 algorithm (HMAC-SHA256)
    - Configurable token expiry (default: 24 hours)
    - Secure parameter defaults (m=65536, t=3, p=4)
    """

    def __init__(
        self,
        secret_key: str,
        token_expiry_hours: int = 24,
        algorithm: str = "HS256"
    ):
        """
        Initialize authentication manager.

        Args:
            secret_key: Secret key for JWT signing (min 32 chars recommended)
            token_expiry_hours: Token validity duration (default: 24 hours)
            algorithm: JWT algorithm (default: HS256)
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")

        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=token_expiry_hours)
        self.algorithm = algorithm

        # Initialize Argon2 password hasher with secure defaults
        # Parameters follow OWASP recommendations (2024)
        self.password_hasher = PasswordHasher(
            time_cost=3,        # Number of iterations (recommended: 2-4)
            memory_cost=65536,  # Memory usage in KiB (64 MB, recommended: 19-65536)
            parallelism=4,      # Number of parallel threads (recommended: 1-4)
            hash_len=32,        # Hash output length in bytes
            salt_len=16,        # Salt length in bytes
            type=Type.ID        # Argon2id variant (hybrid: data-independent + data-dependent)
        )

    # =========================================================================
    # Password Hashing (Argon2)
    # =========================================================================

    def hash_password(self, password: str) -> str:
        """
        Hash password using Argon2id.

        Args:
            password: Plain text password

        Returns:
            Argon2 hash string (includes algorithm, params, salt, and hash)
            Format: $argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>

        Example:
            >>> hash_password("SecurePass123!")
            '$argon2id$v=19$m=65536,t=3,p=4$...'
        """
        if not password:
            raise ValueError("Password cannot be empty")

        return self.password_hasher.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against Argon2 hash.

        Args:
            password: Plain text password to verify
            password_hash: Argon2 hash from database

        Returns:
            True if password matches, False otherwise

        Example:
            >>> verify_password("SecurePass123!", stored_hash)
            True
        """
        if not password or not password_hash:
            return False

        try:
            # verify() raises exception if password doesn't match
            self.password_hasher.verify(password_hash, password)
            return True
        except Exception:
            # Password doesn't match or hash is invalid
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if password hash needs to be rehashed with current parameters.

        Useful for upgrading hash parameters after security updates.

        Args:
            password_hash: Argon2 hash to check

        Returns:
            True if rehashing recommended, False otherwise
        """
        try:
            return self.password_hasher.check_needs_rehash(password_hash)
        except (InvalidHashError, ValueError):
            # Invalid hash format - needs rehash
            return True

    # =========================================================================
    # JWT Token Management
    # =========================================================================

    def create_token(self, user_id: int, email: str) -> str:
        """
        Generate JWT access token.

        Args:
            user_id: User's database ID
            email: User's email address

        Returns:
            JWT token string

        Token structure:
            {
                "user_id": 123,
                "email": "user@example.com",
                "iat": 1700000000,  # Issued at (Unix timestamp)
                "exp": 1700086400   # Expires at (Unix timestamp)
            }

        Example:
            >>> create_token(123, "user@example.com")
            'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
        """
        now = datetime.now(timezone.utc)
        expires_at = now + self.token_expiry

        payload = {
            "user_id": user_id,
            "email": email,
            "iat": now,
            "exp": expires_at
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def validate_token(self, token: str) -> Optional[TokenPayload]:
        """
        Validate and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenPayload if valid, None if invalid/expired

        Example:
            >>> payload = validate_token(token)
            >>> if payload:
            ...     print(f"User ID: {payload.user_id}")
        """
        if not token:
            return None

        try:
            # Decode token (automatically verifies signature and expiration)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            # Convert to Pydantic model for type safety
            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Invalid token (malformed, wrong signature, etc.)
            return None
        except Exception:
            # Unexpected error (e.g., invalid payload structure)
            return None

    def decode_token_unsafe(self, token: str) -> Optional[Dict]:
        """
        Decode token WITHOUT verification (for debugging only).

        ⚠️ WARNING: Does NOT verify signature or expiration!
        Use only for debugging/logging purposes.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict (unverified) or None if malformed
        """
        try:
            return jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
                algorithms=[self.algorithm]
            )
        except Exception:
            return None

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get token expiration time.

        Args:
            token: JWT token string

        Returns:
            Expiration datetime or None if invalid
        """
        payload = self.decode_token_unsafe(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        return None
