"""
Backend Configuration Constants

SSOT (Single Source of Truth) for backend configuration.
All routes should import from here instead of defining their own constants.
"""

from pathlib import Path

# Project root directory (SUJBOT2/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# PDF document storage directory
PDF_BASE_DIR = PROJECT_ROOT / "data"
