"""
Authentication package for SUJBOT2.

Provides JWT token management and password hashing using Argon2.
"""

from .manager import AuthManager

__all__ = ["AuthManager"]
