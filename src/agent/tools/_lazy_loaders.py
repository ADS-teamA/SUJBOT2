"""
Lazy Loaders for Expensive Components

Shared mixins and helpers for lazy initialization of expensive components
like QueryExpander and HyDEGenerator to avoid code duplication.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LazyHyDEMixin:
    """Mixin for lazy HyDE generator initialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyde_generator = None

    def _get_hyde_generator(self):
        """
        Lazy initialization of HyDEGenerator.

        Returns None if initialization fails for any reason (missing API key,
        missing package, or unexpected error).
        """
        if self._hyde_generator is None:
            from ..hyde_generator import HyDEGenerator

            # Get provider and model from config
            provider = self.config.query_expansion_provider
            model = self.config.query_expansion_model

            # Get API keys from environment
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            try:
                self._hyde_generator = HyDEGenerator(
                    provider=provider,
                    model=model,
                    anthropic_api_key=anthropic_key,
                    openai_api_key=openai_key,
                    num_hypotheses=self.config.hyde_num_hypotheses,
                )
                logger.info(f"HyDEGenerator initialized: provider={provider}, model={model}")
            except ValueError as e:
                logger.warning(
                    f"HyDEGenerator configuration error: {e}. "
                    f"HyDE will be disabled. "
                    f"Common causes: missing API key or unsupported provider."
                )
                self._hyde_generator = None
            except ImportError as e:
                package_name = "openai" if provider == "openai" else "anthropic"
                logger.warning(
                    f"HyDEGenerator package missing: {e}. "
                    f"HyDE will be disabled. Install: 'uv pip install {package_name}'"
                )
                self._hyde_generator = None
            except Exception as e:
                logger.error(
                    f"Unexpected error initializing HyDEGenerator ({type(e).__name__}): {e}. "
                    f"HyDE will be disabled."
                )
                self._hyde_generator = None

        return self._hyde_generator


class LazyQueryExpanderMixin:
    """Mixin for lazy QueryExpander initialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_expander = None

    def _get_query_expander(self):
        """
        Lazy initialization of QueryExpander.

        Returns None if initialization fails for any reason.
        """
        if self._query_expander is None:
            from ..query_expander import QueryExpander

            # Get provider and model from config
            provider = self.config.query_expansion_provider
            model = self.config.query_expansion_model

            # Get API keys from environment
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            try:
                self._query_expander = QueryExpander(
                    provider=provider,
                    model=model,
                    anthropic_api_key=anthropic_key,
                    openai_api_key=openai_key,
                )
                logger.info(f"QueryExpander initialized: provider={provider}, model={model}")
            except ValueError as e:
                logger.warning(
                    f"QueryExpander configuration error: {e}. "
                    f"Query expansion will be disabled. "
                    f"Common causes: missing API key or unsupported provider."
                )
                self._query_expander = None
            except ImportError as e:
                package_name = "openai" if provider == "openai" else "anthropic"
                logger.warning(
                    f"QueryExpander package missing: {e}. "
                    f"Query expansion will be disabled. Install: 'uv pip install {package_name}'"
                )
                self._query_expander = None
            except Exception as e:
                logger.error(
                    f"Unexpected error initializing QueryExpander ({type(e).__name__}): {e}. "
                    f"Query expansion will be disabled."
                )
                self._query_expander = None

        return self._query_expander