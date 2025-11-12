"""
PostgreSQL Checkpointer - State persistence for LangGraph workflows.

Integrates with LangGraph's checkpointing system to provide:
1. Automatic state snapshots
2. Workflow pause/resume
3. Error recovery with state rollback
4. Multi-session state management
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


class PostgresCheckpointer:
    """
    PostgreSQL-backed checkpointer for multi-agent workflows.

    Wraps LangGraph's PostgresSaver with additional features:
    - Connection pooling
    - Automatic schema creation
    - State snapshot management
    - Recovery window enforcement
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL checkpointer.

        Args:
            config: Configuration dict with PostgreSQL connection params
        """
        self.config = config

        # Check if using connection_string_env (Docker-friendly approach)
        connection_string_env = config.get("connection_string_env")
        if connection_string_env:
            import os
            self.connection_string = os.getenv(connection_string_env)
            if not self.connection_string:
                raise ValueError(
                    f"Environment variable '{connection_string_env}' not set. "
                    f"Please configure DATABASE_URL in your environment."
                )
            logger.info(f"Using connection string from environment variable: {connection_string_env}")
        else:
            # Extract connection parameters (legacy approach)
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 5432)
            self.user = config.get("user", "postgres")
            self.password = config.get("password")
            self.database = config.get("database", "sujbot_agents")

            # Connection string
            self.connection_string = self._build_connection_string()

        self.table_name = config.get("table_name", "agent_checkpoints")

        # State snapshot configuration
        self.enable_snapshots = config.get("enable_state_snapshots", True)
        self.snapshot_interval = config.get("snapshot_interval_queries", 5)
        self.recovery_window_hours = config.get("recovery_window_hours", 24)

        # LangGraph PostgresSaver (will be initialized lazily)
        self._saver: Optional[PostgresSaver] = None
        self._connection: Optional[Connection] = None

        logger.info(f"PostgresCheckpointer initialized (table={self.table_name})")

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = None
        try:
            conn = psycopg.connect(
                self.connection_string, row_factory=dict_row, autocommit=False
            )
            yield conn
        except Exception as e:
            logger.error(
                f"Database connection failed: {type(e).__name__}: {e}. "
                f"Check: (1) PostgreSQL is running, (2) credentials are correct, (3) database exists."
            )
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def initialize(self) -> None:
        """
        Initialize checkpointer (create tables if needed).

        Creates:
        - Main checkpoint table
        - State snapshot table (if enabled)
        """
        logger.info("Initializing PostgreSQL checkpointer...")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create checkpoint table (LangGraph schema)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        thread_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        checkpoint JSONB NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (thread_id, checkpoint_id)
                    )
                """)

                # Create index for faster lookups
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_thread_id
                    ON {self.table_name}(thread_id, created_at DESC)
                """)

                # Create state snapshot table (if enabled)
                if self.enable_snapshots:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name}_snapshots (
                            snapshot_id SERIAL PRIMARY KEY,
                            thread_id TEXT NOT NULL,
                            checkpoint_id TEXT NOT NULL,
                            query TEXT,
                            state JSONB NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (thread_id, checkpoint_id)
                                REFERENCES {self.table_name}(thread_id, checkpoint_id)
                                ON DELETE CASCADE
                        )
                    """)

                    # Create index for snapshot lookups
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_snapshots_thread
                        ON {self.table_name}_snapshots(thread_id, created_at DESC)
                    """)

                conn.commit()

            logger.info("PostgreSQL checkpointer initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize checkpointer tables in {self.database}.{self.table_name}: "
                f"{type(e).__name__}: {e}. "
                f"Check: (1) User {self.user} has CREATE TABLE permission, "
                f"(2) Database {self.database} exists, (3) No conflicting table schemas.",
                exc_info=True
            )
            raise

    def get_saver(self) -> PostgresSaver:
        """
        Get LangGraph PostgresSaver instance.

        Returns:
            PostgresSaver for LangGraph integration
        """
        if self._saver is None:
            # Initialize database tables first
            self.initialize()

            # Create PostgresSaver with connection
            self._connection = psycopg.connect(
                self.connection_string, autocommit=True, prepare_threshold=0
            )

            self._saver = PostgresSaver(self._connection)
            self._saver.setup()

            logger.info("PostgresSaver created and set up")

        return self._saver

    def save_snapshot(
        self, thread_id: str, checkpoint_id: str, query: str, state: Dict[str, Any]
    ) -> None:
        """
        Save state snapshot for debugging/recovery.

        Args:
            thread_id: Workflow thread ID
            checkpoint_id: Checkpoint ID
            query: Original query
            state: Current state dict
        """
        if not self.enable_snapshots:
            return

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name}_snapshots
                        (thread_id, checkpoint_id, query, state)
                    VALUES (%s, %s, %s, %s)
                """,
                    (thread_id, checkpoint_id, query, psycopg.types.json.Jsonb(state)),
                )

                conn.commit()

            logger.debug(f"Snapshot saved: thread={thread_id}, checkpoint={checkpoint_id}")

        except Exception as e:
            logger.warning(
                f"Failed to save snapshot for thread {thread_id}: {type(e).__name__}: {e}. "
                f"Non-critical error, continuing execution."
            )

    def get_latest_snapshot(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latest state snapshot for thread.

        Args:
            thread_id: Workflow thread ID

        Returns:
            Latest snapshot dict or None
        """
        if not self.enable_snapshots:
            return None

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    SELECT snapshot_id, checkpoint_id, query, state, created_at
                    FROM {self.table_name}_snapshots
                    WHERE thread_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (thread_id,),
                )

                result = cursor.fetchone()

                if result:
                    return dict(result)

                return None

        except Exception as e:
            logger.error(
                f"Failed to retrieve latest snapshot for thread {thread_id}: "
                f"{type(e).__name__}: {e}. Returning None."
            )
            return None

    def cleanup_old_snapshots(self) -> int:
        """
        Clean up snapshots older than recovery window.

        Returns:
            Number of snapshots deleted
        """
        if not self.enable_snapshots:
            return 0

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    DELETE FROM {self.table_name}_snapshots
                    WHERE created_at < NOW() - INTERVAL '%s hours'
                """,
                    (self.recovery_window_hours,),
                )

                deleted_count = cursor.rowcount
                conn.commit()

            logger.info(f"Cleaned up {deleted_count} old snapshots")

            return deleted_count

        except Exception as e:
            logger.error(
                f"Failed to cleanup snapshots older than {self.recovery_window_hours}h: "
                f"{type(e).__name__}: {e}. Returning 0 (no snapshots deleted)."
            )
            return 0

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._saver = None

            logger.info("PostgreSQL checkpointer closed")


def create_checkpointer(config: Dict[str, Any]) -> Optional[PostgresCheckpointer]:
    """
    Create PostgreSQL checkpointer from configuration.

    Args:
        config: Multi-agent config dict

    Returns:
        PostgresCheckpointer instance or None if disabled
    """
    checkpointing_config = config.get("checkpointing", {})

    # Check if checkpointing is enabled and backend is PostgreSQL
    backend = checkpointing_config.get("backend", "none")

    if backend != "postgresql":
        logger.info(f"PostgreSQL checkpointing disabled (backend={backend})")
        return None

    # Extract PostgreSQL configuration
    postgres_config = checkpointing_config.get("postgresql", {})

    if not postgres_config:
        logger.warning("PostgreSQL checkpointing enabled but no configuration provided")
        return None

    # Create checkpointer
    try:
        checkpointer = PostgresCheckpointer(postgres_config)
        checkpointer.initialize()

        logger.info("PostgreSQL checkpointer created successfully")

        return checkpointer

    except Exception as e:
        logger.error(f"Failed to create PostgreSQL checkpointer: {e}", exc_info=True)
        return None
