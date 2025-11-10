"""
Health check utility for Neo4j connectivity and capability verification.

This module provides a simple function to verify Neo4j health before
starting operations. Useful for startup validation and monitoring.
"""

import logging
from typing import Dict, Any

from .config import Neo4jConfig
from .neo4j_manager import Neo4jManager
from .exceptions import Neo4jConnectionError

logger = logging.getLogger(__name__)


def check_neo4j_health(config: Neo4jConfig) -> Dict[str, Any]:
    """
    Perform comprehensive health check on Neo4j instance.

    This function creates a temporary connection, runs health checks,
    and returns detailed status information.

    Args:
        config: Neo4jConfig with connection details

    Returns:
        Dict with keys:
        - connected: bool - Can reach Neo4j server
        - can_write: bool - Can create/delete nodes
        - can_query: bool - Can execute queries
        - response_time_ms: float - Health check duration
        - error: str or None - Error message if failed
        - warnings: List[str] - Non-critical issues

    Example:
        from src.graph.config import Neo4jConfig
        from src.graph.health_check import check_neo4j_health

        config = Neo4jConfig.from_env()
        status = check_neo4j_health(config)

        if status["connected"]:
            print(f"✓ Neo4j healthy ({status['response_time_ms']:.0f}ms)")
        else:
            print(f"✗ Neo4j unhealthy: {status['error']}")
    """
    status = {
        "connected": False,
        "can_write": False,
        "can_query": False,
        "response_time_ms": 0.0,
        "error": None,
        "warnings": [],
    }

    manager = None

    try:
        # Create manager (this tests connectivity)
        manager = Neo4jManager(config)

        # Run health checks
        health = manager.health_check()

        # Copy results
        status["connected"] = health.get("connected", False)
        status["can_write"] = health.get("can_write", False)
        status["can_query"] = health.get("can_query", False)
        status["response_time_ms"] = health.get("response_time_ms", 0.0)

        if not health.get("healthy", False):
            status["error"] = health.get("error", "Unknown health check failure")

        # Add warnings for slow response
        if status["response_time_ms"] > 1000:
            status["warnings"].append(
                f"Slow response time: {status['response_time_ms']:.0f}ms (expected <1000ms)"
            )

        logger.info(f"Health check completed: {status}")

    except Neo4jConnectionError as e:
        status["error"] = str(e)
        logger.error(f"Health check failed (connection): {e}")

    except Exception as e:
        status["error"] = f"Unexpected error: {e}"
        logger.error(f"Health check failed (unexpected): {e}")

    finally:
        # Always cleanup
        if manager:
            manager.close()

    return status
