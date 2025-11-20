"""
Tests for AuthManager - JWT tokens and password hashing.

Critical security tests to prevent:
- Authentication bypass (expired tokens accepted)
- Password hash leakage (weak hashing)
- Token tampering (signature not verified)
- Privilege escalation (user_id modification)
"""

import pytest
from datetime import datetime, timedelta, timezone
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from backend.auth.manager import AuthManager, TokenPayload


@pytest.fixture
def auth_manager():
    """Create AuthManager instance with test secret key."""
    return AuthManager(
        secret_key="test_secret_key_minimum_32_characters_long_for_security",
        token_expiry_hours=24
    )


class TestPasswordHashing:
    """Tests for Argon2 password hashing."""

    def test_hash_password_creates_valid_argon2_hash(self, auth_manager):
        """Verify hash format starts with $argon2id$"""
        password = "SecurePassword123!"
        password_hash = auth_manager.hash_password(password)

        assert password_hash.startswith("$argon2id$")
        assert "v=19" in password_hash  # Argon2 version
        assert "m=65536" in password_hash  # Memory cost
        assert "t=3" in password_hash  # Time cost
        assert "p=4" in password_hash  # Parallelism

    def test_hash_password_different_for_same_password(self, auth_manager):
        """Different salts produce different hashes."""
        password = "SamePassword123!"
        hash1 = auth_manager.hash_password(password)
        hash2 = auth_manager.hash_password(password)

        assert hash1 != hash2  # Different salts

    def test_verify_password_accepts_correct_password(self, auth_manager):
        """Correct password returns True."""
        password = "CorrectPassword123!"
        password_hash = auth_manager.hash_password(password)

        assert auth_manager.verify_password(password, password_hash) is True

    def test_verify_password_rejects_wrong_password(self, auth_manager):
        """Wrong password returns False."""
        password = "CorrectPassword123!"
        password_hash = auth_manager.hash_password(password)

        assert auth_manager.verify_password("WrongPassword123!", password_hash) is False

    def test_verify_password_rejects_empty_password(self, auth_manager):
        """Empty password returns False, not exception."""
        password_hash = auth_manager.hash_password("ValidPassword123!")

        assert auth_manager.verify_password("", password_hash) is False

    def test_verify_password_rejects_empty_hash(self, auth_manager):
        """Empty hash returns False, not exception."""
        assert auth_manager.verify_password("ValidPassword123!", "") is False

    def test_hash_password_raises_on_empty_password(self, auth_manager):
        """Empty password raises ValueError."""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            auth_manager.hash_password("")

    def test_verify_password_handles_corrupted_hash(self, auth_manager):
        """Corrupted hash returns False instead of crashing."""
        corrupted_hash = "$argon2id$v=19$corrupted_data_here"

        assert auth_manager.verify_password("password", corrupted_hash) is False


class TestJWTTokens:
    """Tests for JWT token generation and validation."""

    def test_create_token_generates_valid_jwt(self, auth_manager):
        """Token can be decoded with jwt.decode()."""
        token = auth_manager.create_token(user_id=123, email="test@example.com")

        # Decode without verification to check structure
        payload = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])

        assert payload["user_id"] == 123
        assert payload["email"] == "test@example.com"
        assert "iat" in payload  # Issued at
        assert "exp" in payload  # Expiration

    def test_validate_token_accepts_valid_token(self, auth_manager):
        """Freshly created token validates successfully."""
        token = auth_manager.create_token(user_id=456, email="user@example.com")

        payload = auth_manager.validate_token(token)

        assert payload is not None
        assert payload.user_id == 456
        assert payload.email == "user@example.com"

    def test_validate_token_rejects_expired_token(self, auth_manager):
        """Token past expiry time returns None."""
        # Create manager with very short expiry
        short_expiry_manager = AuthManager(
            secret_key="test_secret_key_minimum_32_characters_long",
            token_expiry_hours=0  # Expires immediately
        )

        token = short_expiry_manager.create_token(user_id=789, email="expired@example.com")

        # Wait a bit to ensure expiration
        import time
        time.sleep(0.1)

        payload = short_expiry_manager.validate_token(token)
        assert payload is None

    def test_validate_token_rejects_tampered_signature(self, auth_manager):
        """Token with modified payload returns None."""
        token = auth_manager.create_token(user_id=100, email="user@example.com")

        # Tamper with token (change user_id in payload)
        parts = token.split(".")
        import base64
        import json

        # Decode payload
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "==")
        payload_dict = json.loads(payload_bytes)

        # Modify user_id (privilege escalation attempt)
        payload_dict["user_id"] = 999

        # Re-encode
        tampered_payload = base64.urlsafe_b64encode(
            json.dumps(payload_dict).encode()
        ).decode().rstrip("=")

        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        # Should reject tampered token
        result = auth_manager.validate_token(tampered_token)
        assert result is None

    def test_validate_token_rejects_wrong_secret_key(self, auth_manager):
        """Token signed with different secret returns None."""
        wrong_manager = AuthManager(
            secret_key="different_secret_key_minimum_32_chars_long",
            token_expiry_hours=24
        )

        token = wrong_manager.create_token(user_id=111, email="test@example.com")

        # Try to validate with original manager (different secret)
        payload = auth_manager.validate_token(token)
        assert payload is None

    def test_create_token_includes_all_required_claims(self, auth_manager):
        """Payload contains user_id, email, iat, exp."""
        token = auth_manager.create_token(user_id=222, email="claims@example.com")

        payload_dict = jwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256"]
        )

        assert "user_id" in payload_dict
        assert "email" in payload_dict
        assert "iat" in payload_dict  # Issued at
        assert "exp" in payload_dict  # Expiration

    def test_token_expiry_respects_configured_duration(self, auth_manager):
        """Token expiry = now + token_expiry_hours."""
        token = auth_manager.create_token(user_id=333, email="expiry@example.com")

        payload_dict = jwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256"]
        )

        iat = datetime.fromtimestamp(payload_dict["iat"], tz=timezone.utc)
        exp = datetime.fromtimestamp(payload_dict["exp"], tz=timezone.utc)

        expected_exp = iat + timedelta(hours=24)

        # Allow 1 second tolerance for test execution time
        assert abs((exp - expected_exp).total_seconds()) < 1

    def test_validate_token_handles_malformed_token(self, auth_manager):
        """Malformed token returns None instead of crashing."""
        malformed_tokens = [
            "not.a.jwt",
            "only_one_part",
            "two.parts",
            "",
            "header.payload.signature.extra",
        ]

        for token in malformed_tokens:
            assert auth_manager.validate_token(token) is None


class TestSecurityConstraints:
    """Security-focused edge cases."""

    def test_auth_manager_rejects_short_secret_key(self):
        """__init__ raises ValueError if secret < 32 chars."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            AuthManager(secret_key="short_key", token_expiry_hours=24)

    def test_auth_manager_accepts_32_char_secret(self):
        """Exactly 32 characters is valid."""
        secret = "a" * 32
        manager = AuthManager(secret_key=secret, token_expiry_hours=24)
        assert manager is not None

    def test_decode_token_unsafe_returns_payload_without_verification(self, auth_manager):
        """decode_token_unsafe works for debugging."""
        token = auth_manager.create_token(user_id=444, email="unsafe@example.com")

        payload = auth_manager.decode_token_unsafe(token)

        assert payload is not None
        assert payload["user_id"] == 444
        assert payload["email"] == "unsafe@example.com"

    def test_decode_token_unsafe_handles_expired_token(self, auth_manager):
        """Unsafe decode works even for expired tokens."""
        short_manager = AuthManager(
            secret_key="test_secret_key_minimum_32_characters",
            token_expiry_hours=0
        )

        token = short_manager.create_token(user_id=555, email="expired@example.com")

        import time
        time.sleep(0.1)

        # Regular validation should fail
        assert short_manager.validate_token(token) is None

        # Unsafe decode should still work
        payload = short_manager.decode_token_unsafe(token)
        assert payload is not None
        assert payload["user_id"] == 555
