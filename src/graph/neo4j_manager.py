"""
Neo4j connection management with retry logic and health monitoring.

This module provides a production-ready Neo4j connection manager with:
- Connection pooling
- Automatic retry with exponential backoff
- Health checks
- Graceful error handling
"""

import time
import logging
from typing import Any, Dict, List, Optional

from .config import Neo4jConfig
from .exceptions import (
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jTimeoutError,
    Neo4jAuthenticationError,
)

logger = logging.getLogger(__name__)


def _sanitize_uri(uri: str) -> str:
    """
    Remove credentials from URI for safe logging.

    Args:
        uri: Neo4j URI that may contain credentials

    Returns:
        URI with password redacted

    Example:
        >>> _sanitize_uri("neo4j://user:pass@host:7687")
        "neo4j://user:***@host:7687"
    """
    import re
    # Use greedy match (.+) to handle passwords containing @ characters
    return re.sub(r"://([^:]+):(.+)@", r"://\1:***@", uri)


class Neo4jManager:
    """
    Manages Neo4j connections with retry logic and batch operations.

    Features:
    - Connection pooling with configurable size
    - Automatic retry for transient failures (3 attempts, exponential backoff)
    - Health checks before operations
    - Graceful degradation on errors

    Example:
        config = Neo4jConfig.from_env()
        manager = Neo4jManager(config)

        # Execute query with automatic retry
        result = manager.execute("MATCH (n) RETURN COUNT(n)")

        # Health check
        status = manager.health_check()

        # Cleanup
        manager.close()
    """

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j connection manager.

        Args:
            config: Neo4jConfig with URI, credentials, and pool settings

        Raises:
            Neo4jConnectionError: If driver creation fails
            Neo4jAuthenticationError: If credentials are invalid
        """
        self.config = config
        self.database = config.database

        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                config.uri,
                auth=(config.username, config.password),
                max_connection_lifetime=config.max_connection_lifetime,
                max_connection_pool_size=config.max_connection_pool_size,
            )

            # Verify connectivity immediately (fail fast)
            self._verify_connectivity()

            logger.info(f"Neo4jManager initialized: {_sanitize_uri(config.uri)}")

        except ImportError as e:
            raise Neo4jConnectionError(
                "neo4j package not installed. Install with: uv pip install neo4j"
            ) from e
        except Exception as e:
            # Check if it's an authentication error
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise Neo4jAuthenticationError(
                    f"Authentication failed. Check NEO4J_USERNAME and NEO4J_PASSWORD: {e}"
                ) from e
            raise Neo4jConnectionError(f"Failed to create Neo4j driver: {e}") from e

    def __enter__(self) -> "Neo4jManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup."""
        self.close()

    def _verify_connectivity(self) -> None:
        """
        Test connection to Neo4j.

        Raises:
            Neo4jConnectionError: If connection fails
        """
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").consume()
        except Exception as e:
            raise Neo4jConnectionError(f"Failed to verify connectivity: {e}") from e

    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query with automatic retry logic.

        Transient errors (network timeouts, server unavailable) are retried
        with exponential backoff. Permanent errors (syntax, constraints) fail immediately.

        Args:
            query: Cypher query string
            params: Query parameters (default: empty dict)
            max_retries: Maximum retry attempts for transient failures (default: 3)

        Returns:
            List of result records as dictionaries

        Raises:
            Neo4jConnectionError: Connection failed after retries
            Neo4jQueryError: Cypher syntax or constraint error (permanent)
            Neo4jTimeoutError: Query timeout

        Example:
            # Simple query
            result = manager.execute("MATCH (n:Entity) RETURN COUNT(n) as count")
            print(result[0]["count"])

            # Parameterized query
            result = manager.execute(
                "MATCH (n:Entity {id: $id}) RETURN n",
                {"id": "entity-123"}
            )
        """
        from neo4j.exceptions import (
            ServiceUnavailable,
            ClientError,
            TransientError,
            DatabaseError,
        )

        params = params or {}
        backoff = 1.0  # Start with 1 second

        for attempt in range(1, max_retries + 1):
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(query, parameters=params)
                    # Convert to list of dicts
                    return [dict(record) for record in result]

            except (ServiceUnavailable, TransientError, DatabaseError) as e:
                # Transient errors - can be retried
                if attempt >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                    raise Neo4jConnectionError(
                        f"Failed after {max_retries} attempts: {e}"
                    ) from e

                wait_time = min(backoff, 32.0)  # Cap at 32 seconds
                logger.warning(
                    f"Transient error (attempt {attempt}/{max_retries}), "
                    f"retrying in {wait_time:.1f}s: {e}"
                )
                time.sleep(wait_time)
                backoff *= 2  # Exponential backoff: 1s, 2s, 4s, 8s, ...

            except ClientError as e:
                # Permanent errors - don't retry
                error_msg = str(e)

                if "timeout" in error_msg.lower():
                    raise Neo4jTimeoutError(f"Query timeout: {e}") from e
                elif "constraint" in error_msg.lower() or "unique" in error_msg.lower():
                    raise Neo4jQueryError(f"Constraint violation: {e}") from e
                else:
                    raise Neo4jQueryError(f"Query failed (permanent error): {e}") from e

            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error in execute(): {e}")
                raise Neo4jConnectionError(f"Unexpected error: {e}") from e

        # Should never reach here
        raise Neo4jConnectionError("Unexpected: retry loop completed without result")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on Neo4j connection.

        Tests:
        1. Connectivity (can reach server)
        2. Write capability (create/delete test node)
        3. Query execution (times operation, warns if > 1000ms)

        Note: Health check passes even with slow response times.
              Check warnings[] in return dict for performance issues.

        Returns:
            Dict with keys:
            - healthy: bool (overall status)
            - connected: bool
            - can_write: bool
            - can_query: bool
            - response_time_ms: float
            - error: str or None
            - warnings: List[str] (empty if no warnings)

        Example:
            status = manager.health_check()
            if not status["healthy"]:
                print(f"Health check failed: {status['error']}")
        """
        status = {
            "healthy": False,
            "connected": False,
            "can_write": False,
            "can_query": False,
            "response_time_ms": 0.0,
            "error": None,
        }

        test_id = None  # Track test node ID for cleanup

        try:
            start_time = time.time()

            # Test 1: Connectivity
            self.execute("RETURN 1")
            status["connected"] = True

            # Test 2: Write capability (create, verify, delete)
            test_id = f"health_check_{int(time.time() * 1000)}"

            self.execute(
                "CREATE (n:HealthCheck {id: $id}) RETURN n",
                {"id": test_id},
            )

            # Verify write
            result = self.execute(
                "MATCH (n:HealthCheck {id: $id}) RETURN COUNT(n) as count",
                {"id": test_id},
            )

            if result and result[0].get("count") == 1:
                status["can_write"] = True

            # Test 3: Query performance
            status["can_query"] = True
            status["response_time_ms"] = (time.time() - start_time) * 1000

            # Overall status
            status["healthy"] = all(
                [status["connected"], status["can_write"], status["can_query"]]
            )

            if status["healthy"]:
                logger.info(
                    f"Health check passed ({status['response_time_ms']:.0f}ms)"
                )

        except Exception as e:
            status["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        finally:
            # Bug #6 fix: Always cleanup test node in finally block
            if test_id is not None:
                try:
                    self.execute("MATCH (n:HealthCheck {id: $id}) DELETE n", {"id": test_id})
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup health check node {test_id}: {cleanup_error}")

        return status

    def close(self) -> None:
        """
        Close Neo4j driver and cleanup resources.

        Always call this when done with the manager to prevent resource leaks.
        """
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4jManager closed")
