"""
Conversation Title Generator using Qwen 2.5 7B via DeepInfra.

Generates concise, descriptive titles from the first user message.
Uses async OpenAI-compatible API for non-blocking operation.
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generate conversation titles using Qwen 2.5 7B via DeepInfra.

    Uses the lighter 7B model for fast, cost-effective title generation.
    Thread-safe and designed for concurrent usage across multiple workers.
    """

    def __init__(self):
        """Initialize the title generator with DeepInfra client."""
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=15.0,
                max_retries=2
            )
            self.model = "Qwen/Qwen2.5-7B-Instruct"
            logger.info("TitleGenerator initialized with DeepInfra Qwen 2.5 7B")
        else:
            self.client = None
            self.model = None
            logger.warning("DEEPINFRA_API_KEY not set - title generation will use fallback")

    async def generate_title(self, first_message: str) -> Optional[str]:
        """
        Generate a conversation title from the first user message.

        Args:
            first_message: The user's first message in the conversation

        Returns:
            Generated title (max 60 chars), or fallback to truncated message
        """
        if not first_message or not first_message.strip():
            return None

        # Fallback if DeepInfra not configured
        if not self.client:
            return self._fallback_title(first_message)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate a very short title (max 6 words) for this conversation. "
                            "The title should capture the main topic or question. "
                            "Return ONLY the title, no quotes, no explanation, no punctuation at the end."
                        )
                    },
                    {"role": "user", "content": first_message[:500]}  # Limit input
                ],
                max_tokens=30,
                temperature=0.3
            )

            title = response.choices[0].message.content
            if title:
                # Cleanup: remove quotes, trailing punctuation, limit length
                title = title.strip().strip('"\'').rstrip('.')
                return title[:60] if len(title) > 60 else title

            return self._fallback_title(first_message)

        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return self._fallback_title(first_message)

    def _fallback_title(self, message: str) -> str:
        """Generate fallback title by truncating the message."""
        # Clean up whitespace and truncate
        clean = " ".join(message.split())
        if len(clean) <= 50:
            return clean
        return clean[:47] + "..."


# Singleton instance for use across the application
title_generator = TitleGenerator()
