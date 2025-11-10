"""
Neo4j-specific exception types for error handling.

This module defines custom exceptions for Neo4j operations, allowing
for precise error handling and automatic retry logic.
"""


class Neo4jError(Exception):
    """Base exception for all Neo4j-related errors."""

    pass


class Neo4jConnectionError(Neo4jError):
    """
    Failed to connect to Neo4j or maintain connection.

    This is a transient error that can be retried with exponential backoff.
    Common causes:
    - Network timeout
    - Server temporarily unavailable
    - Connection pool exhausted
    """

    pass


class Neo4jQueryError(Neo4jError):
    """
    Cypher query error or constraint violation.

    This is a permanent error that should NOT be retried.
    Common causes:
    - Cypher syntax error
    - Constraint violation (duplicate entity ID)
    - Type mismatch in query parameters
    """

    pass


class Neo4jTimeoutError(Neo4jError):
    """
    Query execution exceeded timeout threshold.

    This is treated as a PERMANENT error and will NOT be automatically retried.

    Common causes:
    - Query too complex (needs optimization)
    - Missing indexes on frequently queried properties
    - Server under heavy load

    Action: Optimize query or increase timeout threshold in config.
    """

    pass


class Neo4jAuthenticationError(Neo4jError):
    """
    Authentication failed (invalid credentials).

    This is a permanent error - check NEO4J_USERNAME and NEO4J_PASSWORD.
    """

    pass
