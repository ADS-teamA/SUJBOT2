"""
Logging Configuration for SUJBOT2

This module sets up comprehensive logging for the entire system including:
- File and console handlers
- Structured logging with context
- Performance logging
- Integration with loguru for advanced features
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import os


# ============================================================================
# Standard Logging Configuration
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """
    Setup standard Python logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log message format
        enable_console: Enable console output
        enable_file: Enable file output
    """
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


# ============================================================================
# Loguru Configuration (Advanced Logging)
# ============================================================================

def setup_loguru(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 month",
    compression: str = "zip",
    enable_console: bool = True,
    enable_file: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup loguru for advanced logging.

    Args:
        level: Logging level
        log_file: Path to log file
        rotation: When to rotate logs (e.g., "10 MB", "1 day")
        retention: How long to keep logs (e.g., "1 month", "10 days")
        compression: Compression format (zip, gz, etc.)
        enable_console: Enable console output
        enable_file: Enable file output
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()

    # Default format with colors for console
    if format_string is None:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    else:
        console_format = file_format = format_string

    # Console handler with colors
    if enable_console:
        logger.add(
            sys.stdout,
            level=level.upper(),
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File handler with rotation
    if enable_file and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=level.upper(),
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
            encoding='utf-8',
        )


# ============================================================================
# Integration: Bridge Standard Logging to Loguru
# ============================================================================

class InterceptHandler(logging.Handler):
    """
    Intercept standard logging and redirect to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where logged
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_integrated_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_loguru: bool = True,
) -> None:
    """
    Setup integrated logging that bridges standard logging to loguru.

    Args:
        level: Logging level
        log_file: Path to log file
        use_loguru: Use loguru for advanced features
    """
    if use_loguru:
        # Setup loguru
        setup_loguru(
            level=level,
            log_file=log_file,
            enable_console=True,
            enable_file=log_file is not None,
        )

        # Intercept standard logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Intercept specific loggers
        for logger_name in ["uvicorn", "uvicorn.access", "fastapi"]:
            logging.getLogger(logger_name).handlers = [InterceptHandler()]
    else:
        # Use standard logging
        setup_logging(
            level=level,
            log_file=log_file,
            enable_console=True,
            enable_file=log_file is not None,
        )


# ============================================================================
# Context Logging
# ============================================================================

def log_with_context(
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Log message with additional context.

    Args:
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        context: Context dictionary
        **kwargs: Additional context as keyword arguments
    """
    ctx = context or {}
    ctx.update(kwargs)

    log_func = getattr(logger, level.lower())
    log_func(message, **ctx)


# ============================================================================
# Performance Logging
# ============================================================================

class LoggingTimer:
    """
    Context manager for logging operation duration.
    """

    def __init__(self, operation: str, level: str = "INFO"):
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            logger.log(
                self.level,
                f"Completed: {self.operation} in {duration:.2f}s"
            )
        else:
            logger.error(
                f"Failed: {self.operation} after {duration:.2f}s - {exc_val}"
            )


# ============================================================================
# Configuration from Config Object
# ============================================================================

def setup_logging_from_config(config: Dict[str, Any]) -> None:
    """
    Setup logging from configuration dictionary.

    Args:
        config: Configuration dictionary with logging section
    """
    # Get logging config
    log_config = config.get("logging", {})

    # Get environment overrides
    level = os.getenv("LOG_LEVEL", log_config.get("level", "INFO"))
    verbose = os.getenv("VERBOSE_LOGGING", "false").lower() in ["true", "1", "yes"]

    if verbose:
        level = "DEBUG"

    log_file = log_config.get("file", "logs/sujbot2.log")
    enable_console = log_config.get("console", True)

    # Setup integrated logging
    setup_integrated_logging(
        level=level,
        log_file=log_file if enable_console else None,
        use_loguru=True,
    )

    logger.info(f"Logging configured at level: {level}")


# ============================================================================
# Suppress Noisy Loggers
# ============================================================================

def suppress_noisy_loggers():
    """
    Suppress verbose logging from third-party libraries.
    """
    noisy_loggers = [
        "urllib3",
        "httpx",
        "httpcore",
        "asyncio",
        "PIL",
        "matplotlib",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'setup_logging',
    'setup_loguru',
    'setup_integrated_logging',
    'setup_logging_from_config',
    'log_with_context',
    'LoggingTimer',
    'suppress_noisy_loggers',
]
